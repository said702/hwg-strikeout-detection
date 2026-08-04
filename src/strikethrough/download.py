from __future__ import annotations

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from .utils import ensure_dir


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _stream_download(url: str, out_path: Path, label: str = "", session: requests.Session | None = None) -> Path:
    ensure_dir(out_path.parent)
    client = session or requests.Session()
    with client.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        tqdm_kwargs = {
            "desc": label or f"  {out_path.name}",
            "unit": "B",
            "unit_scale": True,
            "unit_divisor": 1024,
            "total": total if total else None,
        }
        with out_path.open("wb") as handle:
            with tqdm(**tqdm_kwargs) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        pbar.update(len(chunk))
    return out_path


def download_zenodo_record(record_id: int | str, target_dir: Path, file_names: list[str] | None = None) -> list[Path]:
    ensure_dir(target_dir)
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Fetching metadata for Zenodo record {record_id} ...")
    response = requests.get(api_url, timeout=60)
    response.raise_for_status()
    metadata = response.json()

    wanted = set(file_names or [])
    downloaded: list[Path] = []
    for item in metadata.get("files", []):
        name = item.get("key") or item.get("filename")
        if not name:
            continue
        if wanted and name not in wanted:
            continue
        out_path = target_dir / name
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  Skipping {name} (already exists)")
            downloaded.append(out_path)
            continue
        link = item.get("links", {}).get("self")
        if not link:
            continue
        size = item.get("size", 0)
        label = f"  {name} ({_format_bytes(size)})" if size else f"  {name}"
        downloaded.append(_stream_download(link, out_path, label=label))
    return downloaded


def unpack_zip(zip_path: Path, target_dir: Path, overwrite: bool = False) -> None:
    if target_dir.exists() and any(target_dir.iterdir()) and not overwrite:
        return
    ensure_dir(target_dir)
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)


def ensure_hwg_dataset(record_id: int | str, target_dir: Path) -> Path:
    if target_dir.exists() and any(target_dir.iterdir()):
        print("HWG dataset already exists, skipping download.")
        return target_dir
    print(f"Downloading HWG dataset from Zenodo record {record_id} ...")
    download_dir = target_dir.parent / "downloads"
    ensure_dir(download_dir)
    files = download_zenodo_record(record_id, download_dir)
    for path in files:
        if zipfile.is_zipfile(path):
            unpack_zip(path, target_dir)
    return target_dir


def ensure_sws_dataset(record_id: int | str, target_dir: Path) -> Path:
    if target_dir.exists() and any(target_dir.iterdir()):
        print("SWS dataset already exists, skipping download.")
        return target_dir
    print(f"Downloading SWS dataset from Zenodo record {record_id} ...")
    download_dir = target_dir / "_downloads"
    files = download_zenodo_record(record_id, download_dir, ["train.zip", "validation.zip", "test.zip", "README.md", "LICENSE"])
    for path in files:
        if path.suffix.lower() == ".zip":
            split_dir = target_dir / path.stem
            unpack_zip(path, split_dir)
    return target_dir