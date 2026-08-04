from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .utils import ensure_dir


EXPERIMENT_NAMES = (
    "cross_dataset",
    "intra",
    "strike_type_analysis",
    "leave_one_type_out",
    "single_type_training",
    "learning_curve",
)


def _read_overall(path: Path) -> dict:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _read_per_class(path: Path) -> dict:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    out = {}
    for _, row in df.iterrows():
        label = int(row["label"])
        prefix = "non_struck_out" if label == 0 else "struck_out"
        out[f"{prefix}_precision"] = float(row["precision"])
        out[f"{prefix}_recall"] = float(row["recall"])
        out[f"{prefix}_f1"] = float(row["f1"])
        out[f"{prefix}_support"] = float(row["support"])
    return out


def _parse_cross_dataset(rel: Path) -> dict:
    # <model>/train_<train>/eval_<eval>
    parts = rel.parts
    out = {"experiment": "cross_dataset"}
    if len(parts) >= 1:
        out["model"] = parts[0]
    if len(parts) >= 2 and parts[1].startswith("train_"):
        out["train_dataset"] = parts[1][len("train_"):]
    if len(parts) >= 3 and parts[2].startswith("eval_"):
        out["eval_dataset"] = parts[2][len("eval_"):]
    return out


def _parse_strike_type_analysis(rel: Path) -> dict:
    # <model>_train_<train>/eval_<eval>/type_<type>
    parts = rel.parts
    out = {"experiment": "strike_type_analysis"}
    if len(parts) >= 1:
        head = parts[0]
        idx = head.find("_train_")
        if idx > 0:
            out["model"] = head[:idx]
            out["train_dataset"] = head[idx + len("_train_"):]
    if len(parts) >= 2 and parts[1].startswith("eval_"):
        out["eval_dataset"] = parts[1][len("eval_"):]
    if len(parts) >= 3 and parts[2].startswith("type_"):
        out["strike_type"] = parts[2][len("type_"):]
    return out


def _parse_intra(rel: Path) -> dict:
    # <model>/<dataset>/fold_<idx>
    parts = rel.parts
    out = {"experiment": "intra"}
    if len(parts) >= 1:
        out["model"] = parts[0]
    if len(parts) >= 2:
        out["train_dataset"] = parts[1]
        out["eval_dataset"] = parts[1]
    if len(parts) >= 3 and parts[2].startswith("fold_"):
        out["fold"] = parts[2][len("fold_"):]
    return out


def _parse_leave_one_type_out(rel: Path) -> dict:
    # <model>/<strike_type>
    parts = rel.parts
    out = {"experiment": "leave_one_type_out"}
    if len(parts) >= 1:
        out["model"] = parts[0]
    if len(parts) >= 2:
        out["held_out_type"] = parts[1]
    return out


def _parse_single_type_training(rel: Path) -> dict:
    # <model>/train_<train_type>/eval_<target_type>
    parts = rel.parts
    out = {"experiment": "single_type_training"}
    if len(parts) >= 1:
        out["model"] = parts[0]
    if len(parts) >= 2 and parts[1].startswith("train_"):
        out["train_type"] = parts[1][len("train_"):]
    if len(parts) >= 3 and parts[2].startswith("eval_"):
        out["target_type"] = parts[2][len("eval_"):]
        out["eval_split"] = "target_type_eval"
    return out


def _parse_learning_curve(rel: Path) -> dict:
    # <model>/n_<n>/run_<run>/eval_<eval>
    parts = rel.parts
    out = {"experiment": "learning_curve"}
    if len(parts) >= 1:
        out["model"] = parts[0]
    if len(parts) >= 2 and parts[1].startswith("n_"):
        out["sample_size"] = parts[1][len("n_"):]
    if len(parts) >= 3 and parts[2].startswith("run_"):
        out["run"] = parts[2][len("run_"):]
    if len(parts) >= 4 and parts[3].startswith("eval_"):
        out["eval_dataset"] = parts[3][len("eval_"):]
    return out


PARSERS = {
    "cross_dataset": _parse_cross_dataset,
    "strike_type_analysis": _parse_strike_type_analysis,
    "intra": _parse_intra,
    "leave_one_type_out": _parse_leave_one_type_out,
    "single_type_training": _parse_single_type_training,
    "learning_curve": _parse_learning_curve,
}

COLUMNS = [
    "experiment",
    "model",
    "train_dataset",
    "eval_dataset",
    "strike_type",
    "fold",
    "sample_size",
    "run",
    "repetition",
    "held_out_type",
    "train_type",
    "target_type",
    "eval_split",
    "n_samples",
    "n_strike",
    "n_clean",
    "precision",
    "recall",
    "f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "non_struck_out_precision",
    "non_struck_out_recall",
    "non_struck_out_f1",
    "non_struck_out_support",
    "struck_out_precision",
    "struck_out_recall",
    "struck_out_f1",
    "struck_out_support",
]


def _collect_run_rows(exp_name: str, exp_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not exp_dir.exists():
        return rows
    parser = PARSERS[exp_name]
    for overall_path in sorted(exp_dir.rglob("overall_metrics.csv")):
        run_dir = overall_path.parent
        try:
            rel = run_dir.relative_to(exp_dir)
        except ValueError:
            continue
        base = parser(rel)
        # cross_dataset must never report self-evaluation rows.
        if exp_name == "cross_dataset":
            train_ds = base.get("train_dataset")
            eval_ds = base.get("eval_dataset")
            if train_ds and eval_ds and train_ds == eval_ds:
                continue
        overall = _read_overall(overall_path)
        per_class = _read_per_class(run_dir / "per_class_metrics.csv")
        row = {col: "" for col in COLUMNS}
        row.update(base)
        row.update(overall)
        row.update(per_class)
        rows.append(row)
    return rows


def _write_rows(summary_path: Path, rows: list[dict]) -> None:
    ensure_dir(summary_path.parent)
    if rows:
        df = pd.DataFrame(rows)
        for col in COLUMNS:
            if col in df.columns:
                df[col] = df.get(col, "")
        df = df[COLUMNS]
        df.to_csv(summary_path, index=False)
    else:
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()


def aggregate_experiment(exp_name: str, exp_dir: Path) -> Path:
    """Aggregate metrics for a single experiment folder.

    Walks ``exp_dir`` for ``overall_metrics.csv`` files, parses each run's
    metadata from its relative path and writes ``exp_dir/summary_metrics.csv``
    with one row per run including macro / per-class precision, recall, f1.
    Returns the path of the written summary file.
    """
    rows = _collect_run_rows(exp_name, Path(exp_dir))
    summary_path = Path(exp_dir) / "summary_metrics.csv"
    _write_rows(summary_path, rows)
    print(f"    [{exp_name}] aggregated {len(rows)} runs -> {summary_path}", flush=True)
    return summary_path


def aggregate_results(output_root: Path) -> Path:
    output_root = Path(output_root)
    all_rows: list[dict] = []
    for exp_name in EXPERIMENT_NAMES:
        exp_dir = output_root / exp_name
        if not exp_dir.exists():
            continue
        rows = _collect_run_rows(exp_name, exp_dir)
        all_rows.extend(rows)
    summary_path = output_root / "summary_all.csv"
    _write_rows(summary_path, all_rows)
    print(f"    aggregated {len(all_rows)} runs -> {summary_path}", flush=True)

    combined_path = _write_combined_overview(output_root)
    print(f"    combined cross/intra overview -> {combined_path}", flush=True)
    return summary_path


def _write_combined_overview(output_root: Path) -> Path:
    """Combine intra (same train/eval dataset) and cross_dataset (different
    datasets) results into a single overview.

    For identical train/eval dataset pairs only intra rows are used; for
    differing pairs only cross_dataset rows are used. cross_dataset
    self-evaluation is never substituted for intra.
    """
    output_root = Path(output_root)
    combined: list[dict] = []
    intra_rows = _collect_run_rows("intra", output_root / "intra")
    cross_rows = _collect_run_rows("cross_dataset", output_root / "cross_dataset")
    for row in intra_rows:
        train_ds = row.get("train_dataset")
        eval_ds = row.get("eval_dataset")
        if train_ds and eval_ds and train_ds == eval_ds:
            combined.append(row)
    for row in cross_rows:
        train_ds = row.get("train_dataset")
        eval_ds = row.get("eval_dataset")
        if train_ds and eval_ds and train_ds != eval_ds:
            combined.append(row)
    path = output_root / "summary_combined.csv"
    _write_rows(path, combined)
    return path
