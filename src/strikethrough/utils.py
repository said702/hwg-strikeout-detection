from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import IMAGE_EXTENSIONS, TYPE_ALIASES


class ProgressReporter:
    """Adaptive progress reporter.

    Uses a live tqdm progress bar when stdout is attached to a real TTY,
    otherwise falls back to plain flushed prints every ~1% of the work.
    This keeps progress visible in Docker, CI, captured logs and the
    OpenCode browser terminal where tqdm cannot render a live bar.
    """

    def __init__(self, total: int, desc: str = "        predict", unit: str = "img"):
        self.total = max(0, int(total))
        self.desc = desc
        self.count = 0
        self.use_tqdm = bool(getattr(sys.stdout, "isatty", lambda: False)()) and self.total > 0
        self.log_every = max(1, self.total // 100) if self.total > 0 else 1
        self._next_log = self.log_every
        self.pbar = None
        if self.use_tqdm:
            from tqdm import tqdm

            self.pbar = tqdm(total=self.total, desc=desc, unit=unit)

    def update(self, n: int = 1) -> None:
        if self.total <= 0:
            return
        self.count += n
        if self.pbar is not None:
            self.pbar.update(n)
            return
        if self.count >= self.total or self.count >= self._next_log:
            print(f"{self.desc}: {self.count}/{self.total}", flush=True)
            while self.count >= self._next_log:
                self._next_log += self.log_every

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
        elif 0 < self.count < self.total:
            print(f"{self.desc}: {self.count}/{self.total}", flush=True)

    def write(self, msg: str) -> None:
        """Print a message without disturbing the live progress bar."""
        if self.pbar is not None:
            self.pbar.write(msg)
        else:
            print(msg, flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if is_image_file(path))


def normalize_type(value: str | None) -> str:
    if value is None:
        return "unknown"
    key = str(value).strip().lower().replace(" ", "-")
    return TYPE_ALIASES.get(key, key or "unknown")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def write_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def copy_or_link(src: Path, dst: Path, prefer_link: bool = False) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    if prefer_link:
        try:
            dst.symlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def stable_sample_id(sample) -> str:
    """Stable, dataset-unique identifier for a sample: ``dataset|source_id``."""
    return f"{sample.dataset}|{sample.source_id}"


def sample_id_set(samples: Iterable) -> set[str]:
    return {stable_sample_id(s) for s in samples}


def assert_disjoint(label: str, a: Iterable, b: Iterable) -> None:
    a_ids = sample_id_set(a)
    b_ids = sample_id_set(b)
    overlap = a_ids & b_ids
    assert not overlap, f"{label}: overlap of {len(overlap)} sample ids: {sorted(overlap)[:5]}"


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def config_hash(config: dict) -> str:
    """Stable short hash of a config dict (used for checkpoint reuse checks)."""
    return hashlib.sha1(_canonical_json(config).encode("utf-8")).hexdigest()[:16]


def manifest_hash(samples: Iterable) -> str:
    """Stable short hash of a sample collection based on sorted stable ids."""
    ids = sorted(stable_sample_id(s) for s in samples)
    return hashlib.sha1("|".join(ids).encode("utf-8")).hexdigest()[:16]


def cleanup_run_artifacts(model_dir: Path, keep_checkpoint: bool) -> None:
    """Remove a run's model directory (checkpoint + materialized YOLO dataset).

    Called after a model has been trained and evaluated on all evaluation
    datasets when ``keep_checkpoint`` is False. No-op if the directory is
    missing. Eval outputs (``eval_*`` folders) live outside ``model_dir``
    and are never touched.
    """
    if keep_checkpoint:
        return
    if model_dir.exists():
        shutil.rmtree(model_dir)
