from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path | None, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or project_root()) / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    run_cfg = cfg.setdefault("run", {})
    data_sources_path = run_cfg.get("data_sources", "configs/data_sources.yaml")
    cfg["data_sources"] = load_yaml(data_sources_path)
    cfg["_config_path"] = str(resolve_path(path))
    return cfg
