"""Classify batik images with the saved best model.

Usage:
    uv run python -m src.predict <image or folder> [<image or folder> ...]

The model and its metadata are written by notebook 04 (models/best_model.joblib
and models/best_model.json). Preprocessing and feature extraction are the exact
same functions used in training (src.preprocess.preprocess, src.features.extract_all),
and features are ordered by the column list stored in the metadata.

Scores come from SVC.decision_function (one-vs-one votes plus a confidence term,
reshaped per class). They rank the classes but are NOT probabilities.
"""

import argparse
import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import sklearn

from src import config
from src.features import extract_all
from src.preprocess import preprocess

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@lru_cache(maxsize=4)
def load_model(model_dir: Path = config.MODELS_DIR):
    """Load (model, metadata). Warns if the scikit-learn version differs from training."""
    model_path, meta_path = model_dir / "best_model.joblib", model_dir / "best_model.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di {model_dir}. Jalankan notebook 04 terlebih dahulu."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"Model disimpan dengan scikit-learn {meta.get('sklearn_version')}, "
            f"sekarang {sklearn.__version__}. Hasil bisa berbeda; jalankan ulang notebook 04."
        )
    return joblib.load(model_path), meta


def features_for(path: Path, meta: dict) -> np.ndarray:
    """Same preprocessing + features as training, ordered as in the metadata."""
    gray = preprocess(Path(path), size=meta["IMG_SIZE"])
    feats = extract_all(gray)
    return np.array([[feats[c] for c in meta["feature_columns"]]])


def predict_image(path, top_k: int = 3, model_dir: Path = config.MODELS_DIR) -> dict:
    """Return {'label': str, 'top': [(label, score), ...]} for one image file."""
    model, meta = load_model(Path(model_dir))
    X = features_for(Path(path), meta)
    label = model.predict(X)[0]
    scores = model.decision_function(X)[0]
    order = np.argsort(scores)[::-1][:top_k]
    top = [(str(model.classes_[i]), float(scores[i])) for i in order]
    return {"label": str(label), "top": top}


def _collect(paths: list[str]) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            if not found:
                print(f"[lewati] tidak ada citra di folder: {p}")
            files += found
        elif p.is_file():
            files.append(p)
        else:
            print(f"[lewati] tidak ditemukan: {p}")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Klasifikasi motif Batik Banyumasan dari file citra.")
    parser.add_argument("paths", nargs="+", help="file citra atau folder berisi citra")
    parser.add_argument("--top", type=int, default=3, help="jumlah kandidat yang ditampilkan (default 3)")
    args = parser.parse_args(argv)

    files = _collect(args.paths)
    if not files:
        print("Tidak ada citra yang bisa diproses.")
        return 1
    _, meta = load_model(config.MODELS_DIR)
    print(f"Model: {meta['scenario']} + SVM {meta['kernel']} {meta['params']} (F1 CV {meta['cv_f1_macro_mean']})")
    for f in files:
        res = predict_image(f, top_k=args.top)
        cands = ", ".join(f"{lbl} ({s:.2f})" for lbl, s in res["top"])
        print(f"{f.name}: {res['label']}  | top-{args.top}: {cands}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
