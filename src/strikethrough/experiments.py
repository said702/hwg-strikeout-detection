from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import load_yaml, resolve_path
from .constants import LABEL_CLEAN, LABEL_STRIKE, STRIKE_TYPES
from .datasets import Sample
from .metrics import compute_metrics, save_metric_bundle
from .sampling import (
    balanced_type_vs_clean,
    bootstrap_seeds,
    expand_sample_sizes,
    fixed_dev_test_split,
    select_clean_prefix,
    stratified_kfold_splits,
    stratified_three_way_kfold,
    subset_by_size,
    subset_by_size_with_meta,
    three_way_split_by_label,
)
from .train_dino import predict_dino, train_dino_classifier
from .train_yolo import predict_yolo, train_yolo_classifier
from .utils import (
    assert_disjoint,
    cleanup_run_artifacts,
    config_hash,
    ensure_dir,
    manifest_hash,
    sample_id_set,
    stable_sample_id,
    write_csv,
    write_json,
)


CROSS_DATASET_MANIFEST = "cross_dataset_checkpoints.json"

# Required metadata fields that must match before a cross_dataset checkpoint
# can be reused by strike_type_analysis. Old manifests that only stored a
# checkpoint path (plain string) are treated as invalid and force a fresh
# training.
REUSE_REQUIRED_FIELDS = (
    "model",
    "train_dataset",
    "seed",
    "epochs",
    "batch_size",
    "config_hash",
    "manifest_hash",
)


def _write_cross_dataset_manifest(base: Path, manifest: dict) -> None:
    ensure_dir(base)
    with open(base / CROSS_DATASET_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _load_cross_dataset_manifest(output_root: Path) -> dict:
    path = output_root / "cross_dataset" / CROSS_DATASET_MANIFEST
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_key(model: str, train_dataset: str) -> str:
    return f"{model}|{train_dataset}"


def _run_manifest_entry(
    model: str,
    train_dataset: str,
    seed: int,
    config: dict,
    train_samples: list[Sample],
) -> dict:
    training = config.get("training", {})
    return {
        "model": model,
        "train_dataset": train_dataset,
        "seed": int(seed),
        "epochs": int(training.get("epochs", 0)),
        "batch_size": int(training.get("batch_size", 0)),
        "config_hash": config_hash(config),
        "manifest_hash": manifest_hash(train_samples),
    }


def _checkpoint_reusable(
    entry, expected: dict, expected_checkpoint: Path | None
) -> tuple[Path | None, str]:
    """Validate a manifest entry against the expected reuse metadata.

    Returns ``(checkpoint_path_or_None, source)``. Old entries that only
    store a plain checkpoint path string are rejected (no metadata to
    validate against) and force fresh training.
    """
    if not isinstance(entry, dict):
        return None, "fresh"
    checkpoint_path = entry.get("checkpoint")
    if not checkpoint_path:
        return None, "fresh"
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return None, "fresh"
    for field in REUSE_REQUIRED_FIELDS:
        if entry.get(field) != expected.get(field):
            return None, "fresh"
    if expected_checkpoint is not None and checkpoint != expected_checkpoint:
        return None, "fresh"
    return checkpoint, "cross_dataset"


def _apply_val_ratio(config: dict, exp_cfg: dict) -> None:
    val_ratio = exp_cfg.get("val_ratio")
    if val_ratio is not None:
        config.setdefault("training", {})["val_ratio"] = float(val_ratio)


def _model_config(name: str) -> dict:
    return load_yaml(f"configs/{name}.yaml")


def _stratified_val_from_pool(
    pool: list[Sample], n_val: int, seed: int
) -> tuple[list[Sample], list[Sample]]:
    """Split ``pool`` into (remaining, val) stratified by label.

    Returns the validation subset (of size up to ``n_val``) plus the
    remaining samples. Used by ``learning_curve`` to draw a validation set
    from the development pool complement that is disjoint from the training
    subset.
    """
    from .sampling import stratified_split

    if not pool or n_val <= 0:
        return list(pool), []
    train_part, val_part = stratified_split(pool, n_val / max(len(pool), 1), seed)
    return train_part, val_part


def _available(datasets: dict[str, list[Sample]], name: str) -> list[Sample]:
    return datasets.get(name, [])


def _train_and_predict(
    model_name: str,
    train_samples: list[Sample],
    eval_samples: list[Sample],
    output_dir: Path,
    seed: int,
    dry_run: bool,
    epochs: int | None = None,
    val_ratio: float | None = None,
    val_samples: list[Sample] | None = None,
) -> pd.DataFrame | None:
    if not train_samples or not eval_samples:
        return None
    if dry_run:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "train_samples": len(train_samples),
                    "eval_samples": len(eval_samples),
                    "dry_run": True,
                }
            ]
        )
    print(f"      [{model_name}] training on {len(train_samples)} samples ...", flush=True)
    config = _model_config(model_name)
    if epochs is not None:
        config.setdefault("training", {})["epochs"] = epochs
    if val_ratio is not None:
        config.setdefault("training", {})["val_ratio"] = float(val_ratio)
    if model_name == "yolo":
        checkpoint = train_yolo_classifier(train_samples, output_dir / "model", config, seed, val_samples=val_samples)
        print(f"      [{model_name}] predicting on {len(eval_samples)} samples ...", flush=True)
        predictions = predict_yolo(checkpoint, eval_samples, config)
    elif model_name == "dino":
        checkpoint = train_dino_classifier(train_samples, output_dir / "model", config, seed, val_samples=val_samples)
        print(f"      [{model_name}] predicting on {len(eval_samples)} samples ...", flush=True)
        predictions = predict_dino(checkpoint, eval_samples, config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    predictions.insert(0, "model", model_name)
    return predictions


def run_cross_dataset(cfg: dict, datasets: dict[str, list[Sample]], output_root: Path, dry_run: bool) -> None:
    exp = cfg["experiments"]["cross_dataset"]
    models = exp.get("models", ["yolo"])
    train_sets = exp.get("train_datasets", [])
    eval_sets = exp.get("eval_datasets", [])
    base = output_root / "cross_dataset"
    rows = []
    seed = int(cfg["run"].get("seed", 42))
    epochs = exp.get("epochs")
    val_ratio = exp.get("val_ratio")
    manifest: dict[str, dict] = {}
    for model in models:
        active_trains = [(n, _available(datasets, n)) for n in train_sets]
        for train_name, train_samples in active_trains:
            if not train_samples:
                for eval_name in eval_sets:
                    if eval_name == train_name:
                        continue
                    rows.append({"model": model, "train_dataset": train_name, "eval_dataset": eval_name, "status": "skipped_missing_samples"})
        active_trains = [(n, s) for n, s in active_trains if s]
        print(f"    [{model}] starting {len(active_trains)} trainings ...", flush=True)
        training_count = 0
        for train_name, train_samples in active_trains:
            training_count += 1
            print(f"      training {training_count}/{len(active_trains)} on {train_name} ({len(train_samples)} samples)", flush=True)
            model_dir = base / model / f"train_{train_name}"

            config = _model_config(model)
            if epochs is not None:
                config.setdefault("training", {})["epochs"] = epochs
            if val_ratio is not None:
                config.setdefault("training", {})["val_ratio"] = float(val_ratio)

            checkpoint = None
            if not dry_run:
                if model == "yolo":
                    checkpoint = train_yolo_classifier(train_samples, model_dir / "model", config, seed)
                elif model == "dino":
                    checkpoint = train_dino_classifier(train_samples, model_dir / "model", config, seed)
                if checkpoint is not None:
                    entry = _run_manifest_entry(model, train_name, seed, config, train_samples)
                    entry["checkpoint"] = str(checkpoint)
                    manifest[_manifest_key(model, train_name)] = entry
                    write_json(model_dir / "run_manifest.json", entry)

            active_evals = [(n, _available(datasets, n)) for n in eval_sets if n != train_name]
            for eval_name, eval_samples in active_evals:
                if not eval_samples:
                    rows.append({"model": model, "train_dataset": train_name, "eval_dataset": eval_name, "status": "skipped_missing_samples"})
            active_evals = [(n, s) for n, s in active_evals if s]
            if not active_evals:
                continue
            print(f"        starting {len(active_evals)} predicts ...", flush=True)
            for eval_name, eval_samples in active_evals:
                assert train_name != eval_name, (
                    f"cross_dataset must not self-evaluate: train={train_name} eval={eval_name}"
                )
                if dry_run:
                    rows.append({"model": model, "train_dataset": train_name, "eval_dataset": eval_name, "status": "dry_run"})
                    continue

                print(f"        eval on {eval_name} ({len(eval_samples)} samples) ...", flush=True)

                run_dir = model_dir / f"eval_{eval_name}"
                if model == "yolo":
                    predictions = predict_yolo(checkpoint, eval_samples, config)
                elif model == "dino":
                    predictions = predict_dino(checkpoint, eval_samples, config)

                predictions.insert(0, "model", model)
                predictions["train_dataset"] = train_name
                predictions["eval_dataset"] = eval_name
                save_metric_bundle(run_dir, predictions)
                rows.append({"model": model, "train_dataset": train_name, "eval_dataset": eval_name, "status": "done"})

    if manifest and not dry_run:
        _write_cross_dataset_manifest(base, manifest)
    write_csv(base / "summary.csv", rows)


def run_intra_cross_validation(cfg: dict, datasets: dict[str, list[Sample]], output_root: Path, dry_run: bool) -> None:
    from tqdm import tqdm

    exp = cfg["experiments"]["intra"]
    models = exp.get("models", ["yolo"])
    n_folds = int(exp.get("folds", 10))
    val_ratio_of_train = float(exp.get("val_ratio_of_train", exp.get("val_ratio", 0.2)))
    intra_datasets = exp.get("datasets", [])
    seed = int(cfg["run"].get("seed", 42))
    epochs = exp.get("epochs")
    val_ratio = exp.get("val_ratio")
    base = output_root / "intra"
    print(f"\n    [intra] starting {n_folds}-fold cross-validation ...", flush=True)
    for model in models:
        model_base = base / model
        for dataset_name in intra_datasets:
            samples = _available(datasets, dataset_name)
            if not samples:
                print(f"    [{model}] {dataset_name}: skipped (no samples)", flush=True)
                continue
            try:
                folds = stratified_three_way_kfold(samples, n_folds, val_ratio_of_train, seed)
            except ValueError as exc:
                print(f"    [{model}] {dataset_name}: skipped ({exc})", flush=True)
                continue
            dataset_dir = model_base / dataset_name
            print(f"    [{model}] {dataset_name}: {len(samples)} samples, {n_folds} folds", flush=True)
            fold_rows = []
            for fold_idx, (train_samples, val_samples, test_samples) in enumerate(tqdm(folds, desc=f"    [{model}] {dataset_name} folds", bar_format='{desc}: {n_fmt}/{total_fmt}')):
                fold_dir = dataset_dir / f"fold_{fold_idx}"
                assert_disjoint(f"intra {dataset_name} fold{fold_idx} train/val", train_samples, val_samples)
                assert_disjoint(f"intra {dataset_name} fold{fold_idx} train/test", train_samples, test_samples)
                assert_disjoint(f"intra {dataset_name} fold{fold_idx} val/test", val_samples, test_samples)
                if dry_run:
                    fold_rows.append({"dataset": dataset_name, "fold": fold_idx, "train_samples": len(train_samples), "val_samples": len(val_samples), "test_samples": len(test_samples), "status": "dry_run"})
                    continue
                config = _model_config(model)
                if epochs is not None:
                    config.setdefault("training", {})["epochs"] = epochs
                if val_ratio is not None:
                    config.setdefault("training", {})["val_ratio"] = float(val_ratio)
                fold_seed = (seed + fold_idx) % (2**31 - 1)
                print(f"      [{model}] {dataset_name} fold {fold_idx+1}/{n_folds}: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}", flush=True)
                if model == "yolo":
                    checkpoint = train_yolo_classifier(train_samples, fold_dir / "model", config, fold_seed, val_samples=val_samples)
                    predictions = predict_yolo(checkpoint, test_samples, config)
                elif model == "dino":
                    checkpoint = train_dino_classifier(train_samples, fold_dir / "model", config, fold_seed, val_samples=val_samples)
                    predictions = predict_dino(checkpoint, test_samples, config)
                else:
                    raise ValueError(f"Unknown model: {model}")
                predictions.insert(0, "model", model)
                predictions["train_dataset"] = dataset_name
                predictions["eval_dataset"] = dataset_name
                predictions["fold"] = fold_idx
                save_metric_bundle(fold_dir, predictions)
                from .metrics import compute_metrics
                overall, _, _ = compute_metrics(predictions)
                row = {"dataset": dataset_name, "fold": fold_idx, "train_samples": len(train_samples), "val_samples": len(val_samples), "test_samples": len(test_samples), "status": "done"}
                row.update(overall.iloc[0].to_dict())
                fold_rows.append(row)
            if fold_rows:
                df = pd.DataFrame(fold_rows)
                numeric_cols = df.select_dtypes(include="number").columns
                agg = df[numeric_cols].agg(["mean", "std"]).reset_index()
                agg["dataset"] = dataset_name
                agg["fold"] = agg["index"]
                agg = agg.drop(columns=["index"])
                ensure_dir(dataset_dir)
                pd.concat([df, agg], ignore_index=True).to_csv(dataset_dir / "summary.csv", index=False)
    print(f"    [intra] done.", flush=True)


def _load_or_train_checkpoint_for_analysis(
    model: str,
    train_name: str,
    train_samples: list[Sample],
    manifest: dict,
    reuse: bool,
    seed: int,
    epochs: int | None,
    val_ratio: float | None,
    dry_run: bool,
    model_dir: Path,
) -> tuple[Path | None, str]:
    """Return (checkpoint_path_or_None, checkpoint_source).

    A cross_dataset checkpoint is reused only if every field in
    ``REUSE_REQUIRED_FIELDS`` matches the expected metadata computed from the
    current run configuration and the training sample manifest. Old manifest
    entries that only stored a plain checkpoint path are rejected and force
    fresh training.
    """
    config = _model_config(model)
    if epochs is not None:
        config.setdefault("training", {})["epochs"] = epochs
    if val_ratio is not None:
        config.setdefault("training", {})["val_ratio"] = float(val_ratio)
    expected = _run_manifest_entry(model, train_name, seed, config, train_samples)

    checkpoint = None
    source = "fresh"
    if reuse:
        entry = manifest.get(_manifest_key(model, train_name))
        if entry is not None:
            checkpoint, source = _checkpoint_reusable(entry, expected, None)
            if checkpoint is not None:
                print(f"    [strike_type_analysis] reusing cross_dataset checkpoint for {train_name}: {checkpoint}", flush=True)
                return checkpoint, source
            print(f"    [strike_type_analysis] cross_dataset manifest entry for {train_name} failed reuse validation; retraining.", flush=True)
        elif not dry_run:
            print(f"    [strike_type_analysis] no cross_dataset manifest entry for {train_name}; training fresh.", flush=True)

    if dry_run:
        return None, source

    print(f"    [{model}] training fresh checkpoint on {train_name} ({len(train_samples)} samples) ...", flush=True)
    if model == "yolo":
        checkpoint = train_yolo_classifier(train_samples, model_dir / "model", config, seed)
    elif model == "dino":
        checkpoint = train_dino_classifier(train_samples, model_dir / "model", config, seed)
    else:
        raise ValueError(f"Unknown model: {model}")
    return checkpoint, "fresh"


def _predict_for_analysis(model: str, checkpoint: Path, samples: list[Sample], config: dict) -> pd.DataFrame:
    if model == "yolo":
        predictions = predict_yolo(checkpoint, samples, config)
    elif model == "dino":
        predictions = predict_dino(checkpoint, samples, config)
    else:
        raise ValueError(f"Unknown model: {model}")
    predictions.insert(0, "model", model)
    return predictions


def run_strike_type_analysis(cfg: dict, datasets: dict[str, list[Sample]], output_root: Path, dry_run: bool) -> None:
    exp = cfg["experiments"]["strike_type_analysis"]
    base = output_root / "strike_type_analysis"
    rows = []
    seed = int(cfg["run"].get("seed", 42))
    epochs = exp.get("epochs")
    val_ratio = exp.get("val_ratio")
    model = exp.get("model", "yolo")
    reuse = bool(exp.get("reuse_cross_dataset_models", False))
    manifest = _load_cross_dataset_manifest(output_root) if reuse else {}
    if reuse and not manifest and not dry_run:
        print("    [strike_type_analysis] reuse_cross_dataset_models=true but no manifest found; falling back to fresh training.", flush=True)
    strike_types = exp.get("strike_types", STRIKE_TYPES)
    evaluation_map = exp.get("evaluation_by_train_dataset", {})

    for train_name in exp.get("train_datasets", []):
        train_samples = _available(datasets, train_name)
        eval_names = [n for n in evaluation_map.get(train_name, []) if n != train_name]
        if not train_samples and not dry_run:
            for eval_name in eval_names:
                for strike_type in strike_types:
                    rows.append({"train_dataset": train_name, "eval_dataset": eval_name, "strike_type": strike_type, "status": "skipped_missing_samples"})
            continue

        model_dir = base / f"{model}_train_{train_name}"
        checkpoint, checkpoint_source = _load_or_train_checkpoint_for_analysis(
            model, train_name, train_samples, manifest, reuse, seed, epochs, val_ratio, dry_run, model_dir,
        )
        config = _model_config(model)
        if epochs is not None:
            config.setdefault("training", {})["epochs"] = epochs
        if val_ratio is not None:
            config.setdefault("training", {})["val_ratio"] = float(val_ratio)

        for eval_name in eval_names:
            eval_samples = _available(datasets, eval_name)
            if not eval_samples:
                for strike_type in strike_types:
                    rows.append({"train_dataset": train_name, "eval_dataset": eval_name, "strike_type": strike_type, "status": "skipped_missing_samples"})
                continue
            clean_pool = [s for s in eval_samples if s.label == LABEL_CLEAN]
            if not clean_pool:
                print(f"    [strike_type_analysis] {eval_name}: no clean samples; recording NaN for all strike types.", flush=True)
                for strike_type in strike_types:
                    rows.append({
                        "train_dataset": train_name,
                        "eval_dataset": eval_name,
                        "strike_type": strike_type,
                        "n_strike": 0,
                        "n_clean": 0,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "macro_f1": None,
                        "status": "skipped_no_clean",
                        "checkpoint_source": checkpoint_source,
                    })
                continue

            for strike_type in strike_types:
                strikes = [s for s in eval_samples if s.label == LABEL_STRIKE and s.strike_type == strike_type]
                n_strike = len(strikes)
                if n_strike == 0:
                    rows.append({
                        "train_dataset": train_name,
                        "eval_dataset": eval_name,
                        "strike_type": strike_type,
                        "n_strike": 0,
                        "n_clean": 0,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "macro_f1": None,
                        "status": "missing_strike_type",
                        "checkpoint_source": checkpoint_source,
                    })
                    continue
                balanced = balanced_type_vs_clean(strikes, clean_pool, seed, replacement=True)
                n_clean = len(balanced) - n_strike

                if dry_run:
                    rows.append({
                        "train_dataset": train_name,
                        "eval_dataset": eval_name,
                        "strike_type": strike_type,
                        "n_strike": n_strike,
                        "n_clean": n_clean,
                        "status": "dry_run",
                        "checkpoint_source": checkpoint_source,
                    })
                    continue

                print(f"    [{model}] train_{train_name} eval on {eval_name} type={strike_type} ({n_strike} strike + {n_clean} clean)", flush=True)
                run_dir = model_dir / f"eval_{eval_name}" / f"type_{strike_type}"
                predictions = _predict_for_analysis(model, checkpoint, balanced, config)
                predictions["train_dataset"] = train_name
                predictions["eval_dataset"] = eval_name
                predictions["checkpoint_source"] = checkpoint_source
                save_metric_bundle(run_dir, predictions)
                overall, _, _ = compute_metrics(predictions)
                overall_row = overall.iloc[0].to_dict()
                rows.append({
                    "train_dataset": train_name,
                    "eval_dataset": eval_name,
                    "strike_type": strike_type,
                    "n_strike": n_strike,
                    "n_clean": n_clean,
                    "precision": overall_row.get("macro_precision"),
                    "recall": overall_row.get("macro_recall"),
                    "f1": overall_row.get("struck_out_f1"),
                    "macro_f1": overall_row.get("macro_f1"),
                    "status": "done",
                    "checkpoint_source": checkpoint_source,
                })
    write_csv(base / "summary.csv", rows)


def run_leave_one_type_out(cfg: dict, datasets: dict[str, list[Sample]], output_root: Path, dry_run: bool) -> None:
    from tqdm import tqdm

    exp = cfg["experiments"]["leave_one_type_out"]
    base = output_root / "leave_one_type_out"
    all_train = _available(datasets, exp["train_dataset"])
    clean_pool = [s for s in _available(datasets, exp["eval_clean_dataset"]) if s.label == LABEL_CLEAN]
    seed = int(cfg["run"].get("seed", 42))
    epochs = exp.get("epochs")
    val_ratio = exp.get("val_ratio")
    model = exp.get("model", "yolo")
    rows = []
    for strike_type in tqdm(exp.get("strike_types", []), desc="    leave-one-type-out", bar_format='{desc}: {n_fmt}/{total_fmt}'):
        train = [s for s in all_train if not (s.label == LABEL_STRIKE and s.strike_type == strike_type)]
        held_out = [s for s in all_train if s.label == LABEL_STRIKE and s.strike_type == strike_type]
        eval_samples = balanced_type_vs_clean(held_out, clean_pool, seed)
        run_dir = base / model / strike_type
        pred = _train_and_predict(model, train, eval_samples, run_dir, seed, dry_run, epochs, val_ratio)
        if pred is None:
            rows.append({"held_out_type": strike_type, "status": "skipped_missing_samples"})
            continue
        if not dry_run:
            save_metric_bundle(run_dir, pred)
        rows.append({"held_out_type": strike_type, "train_samples": len(train), "eval_samples": len(eval_samples), "status": "dry_run" if dry_run else "done"})
    write_csv(base / "summary.csv", rows)


def run_single_type_training(cfg: dict, datasets: dict[str, list[Sample]], output_root: Path, dry_run: bool) -> None:
    from tqdm import tqdm

    exp = cfg["experiments"]["single_type_training"]
    base = output_root / "single_type_training"
    train_dataset = exp["train_dataset"]
    train_clean_dataset = exp["train_clean_dataset"]
    eval_clean_dataset = exp["eval_clean_dataset"]
    source = _available(datasets, train_dataset)
    clean_train_pool = [s for s in _available(datasets, train_clean_dataset) if s.label == LABEL_CLEAN]
    eval_clean_pool = [s for s in _available(datasets, eval_clean_dataset) if s.label == LABEL_CLEAN]
    seed = int(cfg["run"].get("seed", 42))
    epochs = exp.get("epochs")
    val_ratio = exp.get("val_ratio")
    model = exp.get("model", "yolo")
    split_ratios = tuple(exp.get("split_ratios", (0.70, 0.15, 0.15)))
    strike_types = exp.get("strike_types", [])
    rows = []
    matrix_rows = []
    for train_type in tqdm(strike_types, desc="    single-type-training", bar_format='{desc}: {n_fmt}/{total_fmt}'):
        all_strikes_of_type = [s for s in source if s.label == LABEL_STRIKE and s.strike_type == train_type]
        if not all_strikes_of_type:
            for target_type in strike_types:
                rows.append({"train_type": train_type, "target_type": target_type, "status": "skipped_missing_samples"})
                matrix_rows.append({
                    "train_type": train_type,
                    "target_type": target_type,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "macro_f1": None,
                    "n_strike": 0,
                    "n_clean": 0,
                    "status": "skipped_missing_samples",
                })
            continue

        strike_tr, strike_val, diagonal_test = three_way_split_by_label(all_strikes_of_type, split_ratios, seed)
        n_clean_train = len(strike_tr)
        n_clean_val = len(strike_val)
        clean_trainval = select_clean_prefix(clean_train_pool, n_clean_train + n_clean_val, seed)
        clean_tr = clean_trainval[:n_clean_train]
        clean_va = clean_trainval[n_clean_train : n_clean_train + n_clean_val]
        train_samples = strike_tr + clean_tr
        val_samples = strike_val + clean_va

        assert_disjoint(f"single_type {train_type} train/val", train_samples, val_samples)
        assert_disjoint(f"single_type {train_type} train/diagonal_test", train_samples, diagonal_test)
        assert_disjoint(f"single_type {train_type} val/diagonal_test", val_samples, diagonal_test)

        train_dir = base / model / f"train_{train_type}"
        if dry_run:
            for target_type in strike_types:
                if target_type == train_type:
                    eval_strikes = diagonal_test
                else:
                    eval_strikes = [s for s in source if s.label == LABEL_STRIKE and s.strike_type == target_type]
                rows.append({
                    "train_type": train_type,
                    "target_type": target_type,
                    "train_samples": len(train_samples),
                    "n_strike": len(eval_strikes),
                    "status": "dry_run",
                })
            continue

        print(f"    [{model}] train_type={train_type}: train={len(train_samples)} val={len(val_samples)} diagonal_test={len(diagonal_test)}", flush=True)
        config = _model_config(model)
        if epochs is not None:
            config.setdefault("training", {})["epochs"] = epochs
        if val_ratio is not None:
            config.setdefault("training", {})["val_ratio"] = float(val_ratio)
        if model == "yolo":
            checkpoint = train_yolo_classifier(train_samples, train_dir / "model", config, seed, val_samples=val_samples)
            predict_fn = predict_yolo
        elif model == "dino":
            checkpoint = train_dino_classifier(train_samples, train_dir / "model", config, seed, val_samples=val_samples)
            predict_fn = predict_dino
        else:
            raise ValueError(f"Unknown model: {model}")

        for target_type in strike_types:
            if target_type == train_type:
                eval_strikes = diagonal_test
            else:
                eval_strikes = [s for s in source if s.label == LABEL_STRIKE and s.strike_type == target_type]
            n_strike = len(eval_strikes)
            if n_strike == 0:
                rows.append({"train_type": train_type, "target_type": target_type, "n_strike": 0, "n_clean": 0, "status": "missing_strike_type"})
                matrix_rows.append({
                    "train_type": train_type,
                    "target_type": target_type,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "macro_f1": None,
                    "n_strike": 0,
                    "n_clean": 0,
                    "status": "missing_strike_type",
                })
                continue
            eval_clean = select_clean_prefix(eval_clean_pool, n_strike, seed)
            eval_samples = eval_strikes + eval_clean
            run_dir = train_dir / f"eval_{target_type}"
            print(f"      [{model}] train_{train_type} eval on target={target_type} ({n_strike} strike + {len(eval_clean)} clean)", flush=True)
            predictions = predict_fn(checkpoint, eval_samples, config)
            predictions.insert(0, "model", model)
            predictions["train_type"] = train_type
            predictions["target_type"] = target_type
            predictions["evaluation_target_type"] = target_type
            save_metric_bundle(run_dir, predictions)
            overall, _, _ = compute_metrics(predictions)
            overall_row = overall.iloc[0].to_dict()
            rows.append({
                "train_type": train_type,
                "target_type": target_type,
                "train_samples": len(train_samples),
                "n_strike": n_strike,
                "n_clean": len(eval_clean),
                "precision": overall_row.get("macro_precision"),
                "recall": overall_row.get("macro_recall"),
                "f1": overall_row.get("struck_out_f1"),
                "macro_f1": overall_row.get("macro_f1"),
                "status": "done",
            })
            matrix_rows.append({
                "train_type": train_type,
                "target_type": target_type,
                "precision": overall_row.get("macro_precision"),
                "recall": overall_row.get("macro_recall"),
                "f1": overall_row.get("struck_out_f1"),
                "macro_f1": overall_row.get("macro_f1"),
                "n_strike": n_strike,
                "n_clean": len(eval_clean),
                "status": "done",
            })
    write_csv(base / "summary.csv", rows)
    ensure_dir(base / model)
    write_csv(base / model / "single_type_matrix.csv", matrix_rows)


def run_learning_curve(cfg: dict, datasets: dict[str, list[Sample]], output_root: Path, dry_run: bool) -> None:
    from tqdm import tqdm

    exp = cfg["experiments"]["learning_curve"]
    base = output_root / "learning_curve"
    train_dataset = exp["train_dataset"]
    source = _available(datasets, train_dataset)
    fixed_test_ratio = float(exp.get("fixed_test_ratio", 0.15))
    seed = int(cfg["run"].get("seed", 42))
    seeds = bootstrap_seeds(seed, int(exp.get("repetitions", 10)))
    epochs = exp.get("epochs")
    val_ratio = exp.get("val_ratio")
    model = exp.get("model", "yolo")
    sample_sizes = expand_sample_sizes(exp.get("sample_sizes"))
    eval_dataset_names = list(exp.get("eval_datasets", []))
    keep_checkpoint = bool(exp.get("keep_checkpoint", False))

    dev_pool, fixed_test = fixed_dev_test_split(source, fixed_test_ratio, seed)
    fixed_test_ids = sorted(stable_sample_id(s) for s in fixed_test)
    ensure_dir(base)
    write_json(base / "_fixed_split.json", {
        "train_dataset": train_dataset,
        "fixed_test_ratio": fixed_test_ratio,
        "n_dev": len(dev_pool),
        "n_fixed_test": len(fixed_test),
        "fixed_test_ids": fixed_test_ids,
    })

    rows = []
    size_iter = tqdm(sample_sizes, desc="    learning_curve sizes", bar_format='{desc}: {n_fmt}/{total_fmt}')
    for n in size_iter:
        for run_idx, run_seed in enumerate(seeds):
            train_subset, meta = subset_by_size_with_meta(dev_pool, int(n), run_seed)
            complement = [s for s in dev_pool if s not in train_subset]
            n_val = max(1, int(len(train_subset) * float(val_ratio or 0.2)))
            _, val_subset = _stratified_val_from_pool(complement, n_val, run_seed)
            assert_disjoint(f"learning_curve n={n} run={run_idx} train/fixed_test", train_subset, fixed_test)
            assert_disjoint(f"learning_curve n={n} run={run_idx} val/fixed_test", val_subset, fixed_test)
            assert_disjoint(f"learning_curve n={n} run={run_idx} train/val", train_subset, val_subset)

            run_dir = base / model / f"n_{n}" / f"run_{run_idx}"
            if dry_run:
                for eval_name in eval_dataset_names:
                    if eval_name == "HWG-written-test":
                        eval_samples = fixed_test
                    else:
                        eval_samples = _available(datasets, eval_name)
                    rows.append({
                        "sample_size": n,
                        "repetition": run_idx,
                        "eval_dataset": eval_name,
                        "n_train_actual": len(train_subset),
                        "n_strike": meta["n_strike"],
                        "n_clean": meta["n_clean"],
                        "sampling_strategy": meta["sampling_strategy"],
                        "class_ratio": meta["class_ratio"],
                        "train_ids": len(train_subset),
                        "validation_ids": len(val_subset),
                        "eval_samples": len(eval_samples),
                        "status": "dry_run",
                    })
                continue

            print(f"    [{model}] n={n} run={run_idx}: training once on {len(train_subset)} samples ...", flush=True)
            config = _model_config(model)
            if epochs is not None:
                config.setdefault("training", {})["epochs"] = epochs
            if val_ratio is not None:
                config.setdefault("training", {})["val_ratio"] = float(val_ratio)
            if model == "yolo":
                checkpoint = train_yolo_classifier(train_subset, run_dir / "model", config, run_seed, val_samples=val_subset)
                predict_fn = predict_yolo
            elif model == "dino":
                checkpoint = train_dino_classifier(train_subset, run_dir / "model", config, run_seed, val_samples=val_subset)
                predict_fn = predict_dino
            else:
                raise ValueError(f"Unknown model: {model}")

            for eval_name in eval_dataset_names:
                if eval_name == "HWG-written-test":
                    eval_samples = fixed_test
                else:
                    eval_samples = _available(datasets, eval_name)
                if not eval_samples:
                    rows.append({
                        "sample_size": n,
                        "repetition": run_idx,
                        "eval_dataset": eval_name,
                        "n_train_actual": len(train_subset),
                        "n_strike": meta["n_strike"],
                        "n_clean": meta["n_clean"],
                        "sampling_strategy": meta["sampling_strategy"],
                        "class_ratio": meta["class_ratio"],
                        "train_ids": len(train_subset),
                        "validation_ids": len(val_subset),
                        "eval_samples": 0,
                        "status": "skipped_missing_samples",
                    })
                    continue
                print(f"      [{model}] n={n} run={run_idx} eval on {eval_name} ({len(eval_samples)} samples) ...", flush=True)
                eval_run_dir = run_dir / f"eval_{eval_name}"
                predictions = predict_fn(checkpoint, eval_samples, config)
                predictions.insert(0, "model", model)
                predictions["sample_size"] = n
                predictions["repetition"] = run_idx
                predictions["eval_dataset"] = eval_name
                predictions["n_train_actual"] = len(train_subset)
                save_metric_bundle(eval_run_dir, predictions)
                rows.append({
                    "sample_size": n,
                    "repetition": run_idx,
                    "eval_dataset": eval_name,
                    "n_train_actual": len(train_subset),
                    "n_strike": meta["n_strike"],
                    "n_clean": meta["n_clean"],
                    "sampling_strategy": meta["sampling_strategy"],
                    "class_ratio": meta["class_ratio"],
                    "train_ids": len(train_subset),
                    "validation_ids": len(val_subset),
                    "eval_samples": len(eval_samples),
                    "status": "done",
                })
            if not dry_run and not keep_checkpoint:
                cleanup_run_artifacts(run_dir / "model", keep_checkpoint)
    write_csv(base / "summary.csv", rows)


EXPERIMENT_RUNNERS = {
    "cross_dataset": run_cross_dataset,
    "intra": run_intra_cross_validation,
    "strike_type_analysis": run_strike_type_analysis,
    "leave_one_type_out": run_leave_one_type_out,
    "single_type_training": run_single_type_training,
    "learning_curve": run_learning_curve,
}


def run_selected_experiments(cfg: dict, datasets: dict[str, list[Sample]], dry_run: bool = False) -> None:
    output_root = resolve_path(cfg.get("run", {}).get("output_dir", "results"))
    assert output_root is not None
    for name, runner in EXPERIMENT_RUNNERS.items():
        exp_cfg = cfg.get("experiments", {}).get(name, {})
        if not exp_cfg.get("enabled", False):
            continue
        print(f"\n{'='*60}", flush=True)
        print(f"[experiment] STARTING: {name}", flush=True)
        print(f"{'='*60}", flush=True)
        runner(cfg, datasets, output_root, dry_run)
        print(f"[experiment] DONE: {name}", flush=True)
        if not dry_run:
            from .summary import aggregate_experiment

            print(f"[summary] aggregating {name} ...", flush=True)
            aggregate_experiment(name, output_root / name)
    if not dry_run:
        from .summary import aggregate_results

        print(f"\n{'='*60}", flush=True)
        print("[summary] aggregating all results ...", flush=True)
        print(f"{'='*60}", flush=True)
        aggregate_results(output_root)
