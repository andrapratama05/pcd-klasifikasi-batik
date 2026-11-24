"""Dataset indexing, audit, and the per-origin train/test split."""

import re
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src import config

FILENAME_RE = re.compile(config.FILENAME_PATTERN)


def parse_filename(name: str) -> dict | None:
    """Parse 'Motif (idx)[_Aug].jpg' into motif, idx, origin_id, aug_type.

    Returns None if the name does not match the pattern.
    """
    m = FILENAME_RE.match(name)
    if m is None:
        return None
    motif, idx, aug = m.group("motif"), int(m.group("idx")), m.group("aug")
    return {
        "motif": motif,
        "idx": idx,
        "origin_id": f"{motif}_{idx}",
        "aug_type": aug if aug is not None else "original",
    }


def build_index(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """One row per .jpg file. Label comes from the folder name.

    Columns: filename (relative to raw_dir), label, motif, idx, origin_id,
    aug_type, matched, label_consistent.
    """
    rows = []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for f in sorted(class_dir.iterdir()):
            if not (f.is_file() and f.suffix.lower() == ".jpg"):
                continue
            parsed = parse_filename(f.name)
            row = {"filename": f"{class_dir.name}/{f.name}", "label": class_dir.name}
            if parsed is None:
                row.update(motif=None, idx=None, origin_id=None, aug_type=None, matched=False)
            else:
                row.update(parsed, matched=True)
            row["label_consistent"] = row["motif"] == class_dir.name
            rows.append(row)
    return pd.DataFrame(rows)


def read_image_info(path: Path) -> dict:
    """Size, mode, and whether the file decodes fully."""
    try:
        with Image.open(path) as im:
            im.load()  # force full decode to catch truncated files
            return {"width": im.width, "height": im.height, "mode": im.mode, "corrupt": False}
    except Exception:
        return {"width": None, "height": None, "mode": None, "corrupt": True}


def add_image_info(df: pd.DataFrame, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    from tqdm.auto import tqdm

    info = [read_image_info(raw_dir / fn) for fn in tqdm(df["filename"], desc="baca citra")]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(info)], axis=1)


def audit(df: pd.DataFrame) -> dict:
    """Summarise the index. Works with or without image-info columns."""
    ok = df[df["matched"]]
    files_per_origin = ok.groupby("origin_id").size()
    origin_has_original = ok.groupby("origin_id")["aug_type"].apply(lambda s: (s == "original").any())
    report = {
        "n_files": len(df),
        "per_class": df["label"].value_counts().sort_index(),
        "originals_per_class": ok[ok["aug_type"] == "original"]["label"].value_counts().sort_index(),
        "per_aug_type": ok["aug_type"].value_counts(),
        "n_originals": int((ok["aug_type"] == "original").sum()),
        "n_origin_ids": ok["origin_id"].nunique(),
        "unmatched": df.loc[~df["matched"], "filename"].tolist(),
        "label_mismatch": ok.loc[~ok["label_consistent"], "filename"].tolist(),
        "bad_origins": files_per_origin[files_per_origin != config.FILES_PER_ORIGIN],
        "origins_without_original": origin_has_original[~origin_has_original].index.tolist(),
        "unknown_aug_types": sorted(set(ok["aug_type"]) - set(config.AUG_TYPES) - {"original"}),
    }
    if "corrupt" in df.columns:
        good = df[~df["corrupt"]]
        report["corrupt"] = df.loc[df["corrupt"], "filename"].tolist()
        report["sizes"] = good.groupby(["width", "height"]).size().sort_values(ascending=False)
        report["modes"] = good["mode"].value_counts()
    return report


def make_split(
    df: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.SEED
) -> pd.DataFrame:
    """Split by origin_id (stratified by label); the test set keeps originals only.

    Augmented copies of test origins are dropped entirely, so no photo leaks
    between train and test. Returns columns: filename, label, origin_id, aug_type, split.
    """
    ok = df[df["matched"]] if "matched" in df.columns else df
    origins = ok[["origin_id", "label"]].drop_duplicates("origin_id")
    if origins["origin_id"].duplicated().any():
        raise ValueError("Satu origin_id punya lebih dari satu label")
    train_ids, test_ids = train_test_split(
        origins["origin_id"],
        test_size=test_size,
        stratify=origins["label"],
        random_state=seed,
    )
    out = ok[["filename", "label", "origin_id", "aug_type"]].copy()
    out["split"] = None
    out.loc[out["origin_id"].isin(set(train_ids)), "split"] = "train"
    is_test = out["origin_id"].isin(set(test_ids))
    out.loc[is_test & (out["aug_type"] == "original"), "split"] = "test"
    return out.dropna(subset=["split"]).reset_index(drop=True)
