from __future__ import annotations

import json
from pathlib import Path

from .config import resolve_path
from .datasets import build_hwg_sow_from_local, discover_hwg_datasets, load_manifest, load_sws, save_manifests
from .download import ensure_hwg_dataset, ensure_sws_dataset
from .utils import ensure_dir, write_json


def prepare_datasets(config: dict, dry_run: bool = False) -> dict:
    data_cfg = config["data_sources"]["datasets"]
    run_cfg = config.get("run", {})
    output_dir = resolve_path(run_cfg.get("output_dir", "results"))
    assert output_dir is not None
    prepared_root = ensure_dir(output_dir / "_prepared")
    report = {"datasets": {}, "messages": []}

    hwg_cfg = data_cfg.get("hwg", {})
    hwg_root = resolve_path(hwg_cfg.get("target_dir", "data/HWG-Dataset"))
    assert hwg_root is not None
    if hwg_cfg.get("record_id") and not dry_run:
        try:
            ensure_hwg_dataset(hwg_cfg["record_id"], hwg_root)
        except Exception as exc:
            report["messages"].append(f"HWG download failed: {exc}")
    elif dry_run:
        report["messages"].append(f"Dry-run: would ensure HWG dataset from Zenodo record {hwg_cfg.get('record_id')}")

    sws_cfg = data_cfg.get("sws", {})
    sws_root = resolve_path(sws_cfg.get("target_dir", "data/SWS"))
    assert sws_root is not None
    if sws_cfg.get("enabled", True) and not dry_run:
        try:
            ensure_sws_dataset(sws_cfg.get("record_id", 4765063), sws_root)
        except Exception as exc:
            report["messages"].append(f"SWS download failed: {exc}")
    elif dry_run:
        report["messages"].append("Dry-run: would ensure SWS dataset from Zenodo record 4765063")

    iam_cfg = data_cfg.get("iam", {})
    iam_words_root = resolve_path(iam_cfg.get("words_root"))

    sow_cfg = data_cfg.get("sow", {})
    sow_dataset_root = resolve_path(sow_cfg.get("dataset_root"))

    current_source_paths = {
        "iam_words_root": str(iam_words_root) if iam_words_root else None,
        "sow_dataset_root": str(sow_dataset_root) if sow_dataset_root else None,
        "hwg_target_dir": str(hwg_root) if hwg_root else None,
        "iam_resolution": "targeted-v1",
    }

    manifest_dir = prepared_root / "manifests"
    report_path = prepared_root / "prepare_report.json"
    if report_path.exists() and not dry_run:
        try:
            old_report = json.loads(report_path.read_text())
            old_paths = old_report.get("source_paths", {})
            if old_paths != current_source_paths:
                print("    source paths changed — invalidating cached manifests ...", flush=True)
                for mf in manifest_dir.glob("*.csv"):
                    mf.unlink()
        except (json.JSONDecodeError, OSError):
            pass

    if iam_words_root is None or not iam_words_root.exists():
        report["messages"].append("IAM words_root is not set or missing; IAM-derived HWG-collected samples will be skipped.")

    print("[prepare] discovering HWG datasets ...", flush=True)
    known_manifest_names = {"HWG-synthetic", "HWG-written", "HWG-collected", "HWG-SOW", "SWS"}
    all_manifests_exist = all((manifest_dir / f"{name}.csv").exists() for name in known_manifest_names if name != "SWS")
    if all_manifests_exist and not dry_run:
        print("    loading from cached manifests ...", flush=True)
        datasets = {}
        for name in known_manifest_names:
            if name == "SWS":
                continue
            samples = load_manifest(manifest_dir / f"{name}.csv")
            datasets[name] = samples
            print(f"  {name}: {len(samples)} samples", flush=True)
    else:
        datasets = discover_hwg_datasets(hwg_root, iam_words_root)
        for name, samples in datasets.items():
            print(f"  {name}: {len(samples)} samples", flush=True)

    print("[prepare] building HWG-SOW from local images ...", flush=True)
    sow_samples = []
    if sow_dataset_root and sow_dataset_root.exists():
        sow_output_root = ensure_dir(resolve_path("data/HWG-SOW") or Path("data/HWG-SOW"))
        if not dry_run:
            sow_samples = build_hwg_sow_from_local(sow_cfg, sow_output_root)
        print(f"  HWG-SOW: {len(sow_samples)} samples", flush=True)
    else:
        report["messages"].append("SOW dataset_root is not set or missing; HWG-SOW image crops will be skipped.")
        print("  HWG-SOW: skipped (dataset_root not configured)", flush=True)
    datasets["HWG-SOW"] = sow_samples or datasets.get("HWG-SOW", [])

    print("[prepare] loading SWS dataset ...", flush=True)
    sws_manifest = manifest_dir / "SWS.csv"
    sws_cached_valid = False
    if sws_manifest.exists() and not dry_run:
        cached_sws = load_manifest(sws_manifest)
        # Invalidate cached manifest if it still contains struck_gt samples (pre-fix).
        if not any("struck_gt" in s.image_path for s in cached_sws):
            datasets["SWS"] = cached_sws
            sws_cached_valid = True
            print(f"  SWS: {len(datasets['SWS'])} samples (cached)", flush=True)
    if not sws_cached_valid:
        datasets["SWS"] = load_sws(sws_root)
        print(f"  SWS: {len(datasets['SWS'])} samples", flush=True)

    if not dry_run and (not all_manifests_exist or not sws_cached_valid):
        print("[prepare] saving manifests ...", flush=True)
        save_manifests(prepared_root, datasets)
    for name, samples in datasets.items():
        report["datasets"][name] = {"n_samples": len(samples)}
    report["source_paths"] = current_source_paths
    write_json(report_path, report)
    print("[prepare] done.", flush=True)
    return {"root": prepared_root, "datasets": datasets, "report": report}
