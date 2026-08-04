from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HWG struck-out word detection experiments.")
    parser.add_argument("--config", default="configs/experiments.yaml", help="Path to the experiment YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and data availability without training.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare/download datasets and write manifests without training.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .config import load_experiment_config
    from .environment import print_environment_report
    from .experiments import run_selected_experiments
    from .prepare import prepare_datasets
    from .utils import set_seed

    cfg = load_experiment_config(args.config)
    set_seed(int(cfg.get("run", {}).get("seed", 42)))
    print("Starting environment check ...", flush=True)
    print_environment_report()
    prepared = prepare_datasets(cfg, dry_run=args.dry_run)
    print("Dataset status")
    for name, info in prepared["report"]["datasets"].items():
        print(f"  {name}: {info['n_samples']} samples")
    for message in prepared["report"].get("messages", []):
        print(f"  warning: {message}")

    datasets = prepared["datasets"]
    data_cfg = cfg.get("data_sources", {}).get("datasets", {})
    missing_eval = []
    if not datasets.get("HWG-SOW"):
        sow_root = data_cfg.get("sow", {}).get("dataset_root")
        if not sow_root:
            missing_eval.append("HWG-SOW (SOW dataset_root is not configured)")
    iam_root = data_cfg.get("iam", {}).get("words_root")
    if not iam_root and datasets.get("HWG-collected") is not None:
        missing_eval.append("HWG-collected (IAM words_root is not configured; only a subset of samples is available)")

    if missing_eval:
        if len(missing_eval) == 2:
            print(f"\nWarning: {missing_eval[0]} and {missing_eval[1]} may not be fully evaluated because their required external paths are missing.")
        else:
            print(f"\nWarning: {missing_eval[0]} may not be fully evaluated because the required external path is missing.")
    if args.prepare_only:
        return
    run_selected_experiments(cfg, prepared["datasets"], dry_run=args.dry_run)
