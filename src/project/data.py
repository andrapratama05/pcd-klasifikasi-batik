"""Dataset extraction, indexing, and auditing."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from . import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _safe_target(out_dir: Path, rel: PurePosixPath) -> Path:
    """Resolve an archive member path and refuse anything escaping out_dir."""
    target = (out_dir / Path(*rel.parts)).resolve()
    if not target.is_relative_to(out_dir.resolve()):
        raise ValueError(f"Unsafe path in archive: {rel}")
    return target


def _extract_images(zf: zipfile.ZipFile, out_dir: Path) -> int:
    """Extract image members to out_dir/<class folder>/<file>, dropping wrapper folders."""
    n = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        rel = PurePosixPath(info.filename)
        if rel.suffix.lower() not in IMAGE_EXTS or len(rel.parts) < 2:
            continue
        # keep only "<class folder>/<file>"; the class folder is the image's parent
        target = _safe_target(out_dir, PurePosixPath(rel.parts[-2], rel.parts[-1]))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(info))
        n += 1
    return n


def extract_dataset(zip_path: Path | None = None, out_dir: Path = config.RAW_DIR) -> Path:
    """Extract the Mendeley ZIP (which contains a nested ZIP) into data/raw/.

    Skips extraction when data/raw/ already contains images. The original ZIP is
    never modified.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(p.suffix.lower() in IMAGE_EXTS for p in out_dir.rglob("*")):
        print(f"Dataset already extracted in {out_dir.relative_to(config.ROOT)}")
        return out_dir

    if zip_path is None:
        zips = sorted(config.DATASET_ZIP_DIR.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"No ZIP file found in {config.DATASET_ZIP_DIR}")
        zip_path = zips[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as outer:
        n += _extract_images(outer, out_dir)
        for info in outer.infolist():
            if info.filename.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(outer.read(info))) as inner:
                    n += _extract_images(inner, out_dir)
    print(f"Extracted {n} images to {out_dir.relative_to(config.ROOT)}")
    return out_dir


def normalize_label(folder_name: str) -> str:
    """Map folder names like 'Kawung', 'kawung', 'Batik Kawung' -> 'kawung'."""
    name = re.sub(r"[^a-z]", "", folder_name.lower())
    name = name.removeprefix("batik")
    return name


def build_index(raw_dir: Path = config.RAW_DIR, classes: list[str] | None = None) -> pd.DataFrame:
    """List every image with its label (taken from the parent folder name)."""
    rows = []
    for p in sorted(Path(raw_dir).rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        rows.append(
            {
                "filepath": p.relative_to(config.ROOT).as_posix(),
                "filename": p.name,
                "folder": p.parent.name,
                "label": normalize_label(p.parent.name),
            }
        )
    df = pd.DataFrame(rows)
    if classes is not None:
        df = df[df["label"].isin(classes)].reset_index(drop=True)
    return df


def audit_images(df: pd.DataFrame) -> pd.DataFrame:
    """Open every image and record size, color mode, and whether it is readable."""
    records = []
    for fp in tqdm(df["filepath"], desc="Audit"):
        rec = {"filepath": fp, "ok": True, "width": None, "height": None, "mode": None, "error": ""}
        try:
            with Image.open(config.ROOT / fp) as im:
                rec["mode"] = im.mode
                rec["width"], rec["height"] = im.size
                im.load()  # forces full decode, catches truncated files
        except Exception as e:  # noqa: BLE001 - we want to log any decode failure
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
        records.append(rec)
    return df.merge(pd.DataFrame(records), on="filepath")
