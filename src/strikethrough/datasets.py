from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from .config import resolve_path
from .constants import LABEL_CLEAN, LABEL_NAMES, LABEL_STRIKE, NAME_TO_LABEL, STRIKE_TYPES
from .utils import ensure_dir, is_image_file, iter_images, normalize_type, write_csv, write_json


@dataclass(frozen=True)
class Sample:
    image_path: str
    label: int
    dataset: str
    split: str = "all"
    source_id: str = ""
    strike_type: str = "unknown"

    @property
    def label_name(self) -> str:
        return LABEL_NAMES[self.label]


def sample_to_row(sample: Sample) -> dict:
    row = asdict(sample)
    row["label_name"] = sample.label_name
    return row


def _find_first(root: Path, names: Iterable[str]) -> Path | None:
    wanted = {name.lower() for name in names}
    for path in root.rglob("*"):
        if path.name.lower() in wanted:
            return path
    return None


def _annotation_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows


def _label_from_gold(gold: str) -> int:
    normalized = normalize_type(gold)
    if normalized == "none":
        return LABEL_CLEAN
    return NAME_TO_LABEL.get(gold.lower(), LABEL_STRIKE)


def _image_index(root: Path) -> dict[str, Path]:
    index = {}
    for path in iter_images(root):
        index[path.name] = path
        index[path.stem] = path
        try:
            index[path.relative_to(root).as_posix()] = path
        except ValueError:
            pass
    return index


def _match_image(root: Path, rel_path: str, external_index: dict[str, Path] | None = None) -> Path | None:
    direct = root / rel_path
    if direct.exists():
        return direct
    index = _image_index(root)
    candidates = [
        rel_path,
        Path(rel_path).name,
        Path(rel_path).stem,
        rel_path.replace("\\", "/"),
    ]
    if external_index:
        index = {**index, **external_index}
    for candidate in candidates:
        if candidate in index:
            return index[candidate]
    return None


def _match_iam_image(words_root: Path, rel_path: str) -> Path | None:
    """Resolve one IAM word path without scanning the complete words directory."""
    normalized = rel_path.replace("\\", "/")
    prefix = "datasets/IAM/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix):]
    normalized = normalized.removeprefix("clean/").removeprefix("struck-out/")

    filename = Path(normalized).name
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None

    # IAM words are stored as words/<writer>/<form>/<word-image>.
    writer = parts[0]
    form = "-".join(parts[:2])
    candidates = [
        words_root / writer / form / filename,
        words_root / writer / filename,
        words_root / normalized,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_annotation_dataset(root: Path, dataset_name: str, annotation_names: list[str], external_image_root: Path | None = None) -> list[Sample]:
    annotation = _find_first(root, annotation_names)
    if annotation is None:
        return []
    samples = []
    for row in _annotation_rows(annotation):
        if len(row) < 3:
            continue
        rel_path, status, gold = row[0], row[1], row[2]
        if status.lower() != "ok":
            continue
        image_path = _match_image(root, rel_path)
        if image_path is None and external_image_root and external_image_root.exists():
            image_path = _match_iam_image(external_image_root, rel_path)
        if image_path is None:
            continue
        strike_type = normalize_type(gold)
        samples.append(
            Sample(
                image_path=str(image_path),
                label=_label_from_gold(gold),
                dataset=dataset_name,
                source_id=rel_path,
                strike_type=strike_type,
            )
        )
    return samples


def load_folder_dataset(root: Path, dataset_name: str) -> list[Sample]:
    samples = []
    if not root.exists():
        return samples
    for path in iter_images(root):
        parts = [part.lower() for part in path.parts]
        label = None
        if any(part in {"clean", "none", "no", "non-struck-out", "no_strike"} for part in parts):
            label = LABEL_CLEAN
        elif any(part in {"strike", "struck-out", "struck", "genuine"} for part in parts):
            label = LABEL_STRIKE
        if label is None:
            continue
        strike_type = "none" if label == LABEL_CLEAN else "unknown"
        for part in path.parts:
            normalized = normalize_type(part)
            if normalized in STRIKE_TYPES:
                strike_type = normalized
                break
        samples.append(
            Sample(
                image_path=str(path),
                label=label,
                dataset=dataset_name,
                source_id=str(path.relative_to(root)),
                strike_type=strike_type,
            )
        )
    return samples


def _load_csv_chunk(idxs, image_names, labels_col, strike_types, dataset_name, root, external_image_root):
    samples = []
    iam_total = 0
    iam_found = 0
    for i in idxs:
        rel_path = image_names[i]
        image_path = root / rel_path
        is_iam = rel_path.replace("\\", "/").startswith("datasets/IAM/")
        if not image_path.is_file() and external_image_root and is_iam:
            iam_total += 1
            image_path = _match_iam_image(external_image_root, rel_path)
            if image_path is not None and image_path.is_file():
                iam_found += 1
        if image_path is None or not image_path.is_file():
            continue
        label_name = labels_col[i].strip().lower()
        label = NAME_TO_LABEL.get(label_name)
        if label is None:
            label = LABEL_STRIKE if "struck" in label_name and "non" not in label_name else LABEL_CLEAN
        strike_type_raw = strike_types[i] if strike_types is not None else ""
        strike_type = "none" if label == LABEL_CLEAN else normalize_type(strike_type_raw)
        samples.append(
            Sample(
                image_path=str(image_path),
                label=label,
                dataset=dataset_name,
                source_id=rel_path,
                strike_type=strike_type,
            )
        )
    return samples, iam_total, iam_found


def load_gt_csv_dataset(root: Path, dataset_name: str, external_image_root: Path | None = None, workers: int = 1) -> list[Sample]:
    from .utils import ProgressReporter

    gt_path = root / "gt.csv"
    if not gt_path.exists():
        return []
    frame = pd.read_csv(gt_path)
    required = {"image_name", "label"}
    if not required.issubset(frame.columns):
        return []
    image_names = frame["image_name"].to_numpy(dtype=str)
    labels_col = frame["label"].to_numpy(dtype=str)
    has_strike_type = "strikeout_type" in frame.columns
    strike_types = frame["strikeout_type"].to_numpy(dtype=str) if has_strike_type else None

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunks = np.array_split(np.arange(len(image_names)), workers)
        samples = []
        iam_total = 0
        iam_found = 0
        progress = ProgressReporter(len(chunks), desc=f"    {dataset_name}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _load_csv_chunk,
                    idxs, image_names, labels_col,
                    strike_types if has_strike_type else None,
                    dataset_name, root, external_image_root,
                ): idxs
                for idxs in chunks
            }
            for future in as_completed(futures):
                chunk_samples, ct, cf = future.result()
                samples.extend(chunk_samples)
                iam_total += ct
                iam_found += cf
                progress.update(1)
        progress.close()
    else:
        samples = []
        iam_total = 0
        iam_found = 0
        iam_printed_header = False
        progress = ProgressReporter(len(image_names), desc=f"    {dataset_name}")
        for i in range(len(image_names)):
            rel_path = image_names[i]
            image_path = root / rel_path
            is_iam = rel_path.replace("\\", "/").startswith("datasets/IAM/")
            if not image_path.is_file() and external_image_root and is_iam:
                iam_total += 1
                if not iam_printed_header:
                    progress.write(f"    looking up IAM images under {external_image_root} ...")
                    iam_printed_header = True
                image_path = _match_iam_image(external_image_root, rel_path)
                if image_path is not None and image_path.is_file():
                    iam_found += 1
            if image_path is None or not image_path.is_file():
                continue
            label_name = labels_col[i].strip().lower()
            label = NAME_TO_LABEL.get(label_name)
            if label is None:
                label = LABEL_STRIKE if "struck" in label_name and "non" not in label_name else LABEL_CLEAN
            strike_type_raw = strike_types[i] if has_strike_type else ""
            strike_type = "none" if label == LABEL_CLEAN else normalize_type(strike_type_raw)
            samples.append(
                Sample(
                    image_path=str(image_path),
                    label=label,
                    dataset=dataset_name,
                    source_id=rel_path,
                    strike_type=strike_type,
                )
            )
            progress.update(1)
        progress.close()
    if iam_total > 0:
        iam_missing = iam_total - iam_found
        if iam_missing == 0:
            print(f"    IAM images: {iam_found}/{iam_total} all found", flush=True)
        else:
            print(f"    IAM images: {iam_found}/{iam_total} found, {iam_missing} missing", flush=True)
    return samples


def _parse_sow_annotation(path: Path) -> dict[str, tuple[str, str]]:
    """Parse the SOW gt.csv annotation file.

    Returns a mapping from image_name (e.g. ``train_sow/11_strike_17.jpg``)
    to a ``(label, strikeout_type)`` tuple, where ``label`` is the raw
    string from the CSV (e.g. ``struck-out`` / ``non-struck-out``) and
    ``strikeout_type`` is the strike type string (or empty string).
    """
    if not path.exists():
        return {}
    mapping: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = (row.get("image_name") or "").strip()
            if not image_name:
                continue
            label = (row.get("label") or "").strip()
            strike_type = (row.get("strikeout_type") or "").strip()
            mapping[image_name] = (label, strike_type)
    return mapping


def _parse_yolo_line(line: str, width: int, height: int) -> tuple[int, int, int, int, int] | None:
    parts = line.strip().replace(",", " ").split()
    if len(parts) < 5:
        return None
    # SOW label files use the format: x1,y1,x2,y2 <label> (label is the last token).
    # A YOLO label-first fallback (label x y w h) is also supported.
    raw_label = None
    nums = None
    try:
        candidate = int(float(parts[-1]))
        if candidate >= 0:
            raw_label = candidate
            nums = [float(x) for x in parts[:4]]
    except ValueError:
        pass
    if raw_label is None:
        try:
            raw_label = int(float(parts[0]))
            nums = [float(x) for x in parts[1:5]]
        except ValueError:
            return None
    x1, y1, x2, y2 = nums
    if max(nums) <= 1.5:
        # Normalized YOLO format: cx, cy, w, h
        cx, cy, w, h = nums
        left = int(max(0, cx - w / 2) * width)
        top = int(max(0, cy - h / 2) * height)
        right = int(min(width, cx + w / 2) * width)
        bottom = int(min(height, cy + h / 2) * height)
    else:
        # Absolute pixel coordinates: x1, y1, x2, y2
        left, top, right, bottom = map(int, [x1, y1, x2, y2])
    if right <= left or bottom <= top:
        return None
    return raw_label, left, top, right, bottom


def build_hwg_sow_from_local(sow_cfg: dict, output_root: Path) -> list[Sample]:
    dataset_root = resolve_path(sow_cfg.get("dataset_root"))
    annotation_root = resolve_path(sow_cfg.get("annotation_root"))
    if dataset_root is None or not dataset_root.exists():
        return []
    annotation_candidates = [
        dataset_root / str(sow_cfg.get("annotation_file", "")),
        annotation_root / str(sow_cfg.get("annotation_file", "")) if annotation_root else None,
    ]
    annotation_file = next((p for p in annotation_candidates if p and p.exists()), None)
    annotation = _parse_sow_annotation(annotation_file) if annotation_file else {}
    crop_root = ensure_dir(output_root / "HWG-SOW")
    samples = []
    split_specs = [
        ("train", sow_cfg.get("train_images"), sow_cfg.get("train_labels")),
        ("test", sow_cfg.get("test_images"), sow_cfg.get("test_labels")),
    ]
    for split, image_rel, label_rel in split_specs:
        image_dir = dataset_root / image_rel if image_rel else None
        label_dir = dataset_root / label_rel if label_rel else None
        if image_dir is None or label_dir is None or not image_dir.exists() or not label_dir.exists():
            continue
        # split_dir_name matches the gt.csv image_name prefix (e.g. "train_sow", "SOW_test")
        split_dir_name = Path(image_rel).parts[0] if image_rel else split
        for image_path in iter_images(image_dir):
            label_file = label_dir / f"{image_path.name}.txt"
            if not label_file.exists():
                label_file = label_dir / f"{image_path.stem}.txt"
            if not label_file.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            for idx, line in enumerate(label_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
                parsed = _parse_yolo_line(line, width, height)
                if parsed is None:
                    continue
                raw_label, left, top, right, bottom = parsed
                # Build the gt.csv image_name key: {split_dir}/{stem}_{clean|strike}_{idx}.jpg
                yolo_class = "strike" if raw_label == 1 else "clean"
                crop_key = f"{split_dir_name}/{image_path.stem}_{yolo_class}_{idx}.jpg"
                gt_entry = annotation.get(crop_key)
                if gt_entry is None:
                    # No gt.csv annotation for this crop — skip, gt.csv is the single source of truth.
                    continue
                gt_label_str, strike_type_str = gt_entry
                label = LABEL_STRIKE if gt_label_str.strip().lower() == "struck-out" else LABEL_CLEAN
                crop = image[top:bottom, left:right]
                if crop.size == 0:
                    continue
                crop_name = Path(crop_key).name
                strike_type = "none" if label == LABEL_CLEAN else normalize_type(strike_type_str)
                label_dir_name = "non-struck-out" if label == LABEL_CLEAN else "struck-out"
                out_path = crop_root / split / label_dir_name / crop_name
                ensure_dir(out_path.parent)
                cv2.imwrite(str(out_path), crop)
                samples.append(
                    Sample(
                        image_path=str(out_path),
                        label=label,
                        dataset="HWG-SOW",
                        split=split,
                        source_id=f"{image_path.name}:{idx}",
                        strike_type=strike_type,
                    )
                )
    return samples


def load_sws(root: Path) -> list[Sample]:
    from .utils import ProgressReporter

    samples = []
    if not root.exists():
        return samples
    for split in ["train", "validation", "test"]:
        split_root = root / split
        if not split_root.exists():
            continue
        type_map = {}
        for csv_path in split_root.rglob("*.csv"):
            try:
                frame = pd.read_csv(csv_path)
            except Exception:
                continue
            for _, row in frame.iterrows():
                values = [str(v) for v in row.values if pd.notna(v)]
                if not values:
                    continue
                type_map[Path(values[0]).name] = normalize_type(values[-1])
        images = iter_images(split_root)
        progress = ProgressReporter(len(images), desc=f"    SWS/{split}")
        for image_path in images:
            # struck_gt contains the ground-truth clean references for restoration,
            # not struck-out samples — skip to avoid duplicates / wrong labels.
            if any(part == "struck_gt" for part in image_path.parts):
                continue
            parts = [part.lower() for part in image_path.parts]
            is_clean = any(part == "clean" for part in parts)
            label = LABEL_CLEAN if is_clean else LABEL_STRIKE
            samples.append(
                Sample(
                    image_path=str(image_path),
                    label=label,
                    dataset="SWS",
                    split=split,
                    source_id=str(image_path.relative_to(root)),
                    strike_type="none" if label == LABEL_CLEAN else type_map.get(image_path.name, "unknown"),
                )
            )
            progress.update(1)
        progress.close()
    return samples


def discover_hwg_datasets(hwg_root: Path, iam_words_root: Path | None = None) -> dict[str, list[Sample]]:
    inner = hwg_root / "HWG-Dataset" if (hwg_root / "HWG-Dataset").exists() else hwg_root

    specs = [
        ("HWG-written", "HWG-written"),
        ("HWG-collected", "HWG-collected"),
        ("HWG-synthetic", "HWG-synthetic"),
    ]

    datasets: dict[str, list[Sample]] = {}

    for name, dir_name in specs:
        root = inner / dir_name
        if not root.exists():
            print(f"    {name}: skipped (directory not found)", flush=True)
            datasets[name] = []
            continue
        external_root = iam_words_root if name == "HWG-collected" else None
        print(f"    loading {name} ...", flush=True)
        datasets[name] = load_gt_csv_dataset(root, name, external_root, workers=8)
        if not datasets[name]:
            print(f"    {name}: no gt.csv, scanning folder ...", flush=True)
            datasets[name] = load_folder_dataset(root, name)
        print(f"    {name}: {len(datasets[name])} samples", flush=True)

    datasets["HWG-SOW"] = []
    return datasets


def save_manifests(output_dir: Path, datasets: dict[str, list[Sample]]) -> None:
    from .utils import ProgressReporter

    manifest_dir = ensure_dir(output_dir / "manifests")
    inventory = []
    for name, samples in datasets.items():
        print(f"    saving {name} manifest ({len(samples)} samples) ...", flush=True)
        progress = ProgressReporter(len(samples), desc=f"    {name}")
        rows = []
        for sample in samples:
            rows.append(sample_to_row(sample))
            progress.update(1)
        progress.close()
        write_csv(manifest_dir / f"{name}.csv", rows)
        counts = {}
        for sample in samples:
            key = sample.label_name
            counts[key] = counts.get(key, 0) + 1
        inventory.append({"dataset": name, "n_samples": len(samples), **counts})
    write_csv(output_dir / "dataset_inventory.csv", inventory)
    write_json(output_dir / "dataset_inventory.json", inventory)


def load_manifest(path: Path) -> list[Sample]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                Sample(
                    image_path=row["image_path"],
                    label=int(row["label"]),
                    dataset=row["dataset"],
                    split=row.get("split", "all"),
                    source_id=row.get("source_id", ""),
                    strike_type=row.get("strike_type", "unknown"),
                )
            )
    return rows


def stable_name(sample: Sample) -> str:
    digest = hashlib.sha1(sample.image_path.encode("utf-8")).hexdigest()[:12]
    return f"{sample.dataset}_{digest}{Path(sample.image_path).suffix.lower() or '.jpg'}"
