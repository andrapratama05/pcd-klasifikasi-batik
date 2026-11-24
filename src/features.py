"""GLCM and LBP features (see CLAUDE.md, 'Detail implementasi yang mudah salah').

Column naming:
    glcm_<prop>_d<d>_<a0|a45|a90|a135>   per angle (S1 uses the a0 columns)
    glcm_<prop>_d<d>_mean                 mean over the 4 angles (S2)
    lbp_00 ... lbp_58                     uniform LBP histogram, L2-normalised (S3)
"""

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from src import config

GLCM_PROPS = config.GLCM_PROPS  # contrast, correlation, asm, homogeneity


def quantize(gray: np.ndarray, levels: int = config.GLCM_LEVELS) -> np.ndarray:
    """Map uint8 [0, 255] to `levels` equal-width bins (like MATLAB NumLevels with GrayLimits [0 255])."""
    return (gray.astype(np.uint16) * levels // 256).astype(np.uint8)


def glcm_matrix(gray: np.ndarray) -> np.ndarray:
    """Normalised, non-symmetric GLCM of shape (levels, levels, n_dist, n_angle)."""
    return graycomatrix(
        quantize(gray),
        distances=config.GLCM_DISTANCES,
        angles=config.GLCM_ANGLES,
        levels=config.GLCM_LEVELS,
        symmetric=False,
        normed=True,
    )


def homogeneity_abs(P: np.ndarray) -> np.ndarray:
    """Homogeneity with 1 / (1 + |i - j|) (paper / MATLAB); skimage uses (i - j)^2."""
    n = P.shape[0]
    i, j = np.ogrid[:n, :n]
    w = 1.0 / (1.0 + np.abs(i - j))
    return np.einsum("ij,ijda->da", w, P)


def glcm_props(P: np.ndarray) -> dict[str, np.ndarray]:
    """Each value has shape (n_dist, n_angle)."""
    return {
        "contrast": graycoprops(P, "contrast"),
        "correlation": graycoprops(P, "correlation"),
        "asm": graycoprops(P, "ASM"),  # sum P^2 (paper's 'energy', eq. 7)
        "homogeneity": homogeneity_abs(P),
    }


def glcm_features(gray: np.ndarray) -> dict[str, float]:
    props = glcm_props(glcm_matrix(gray))
    feats = {}
    for name, vals in props.items():
        for di, d in enumerate(config.GLCM_DISTANCES):
            for ai, a in enumerate(config.GLCM_ANGLE_NAMES):
                feats[f"glcm_{name}_d{d}_{a}"] = float(vals[di, ai])
            feats[f"glcm_{name}_d{d}_mean"] = float(vals[di].mean())
    return feats


def lbp_map(gray: np.ndarray) -> np.ndarray:
    return local_binary_pattern(gray, config.LBP_P, config.LBP_R, method=config.LBP_METHOD)


def lbp_features(gray: np.ndarray) -> dict[str, float]:
    hist, _ = np.histogram(lbp_map(gray), bins=config.LBP_BINS, range=(0, config.LBP_BINS))
    hist = hist.astype(np.float64)
    hist /= np.linalg.norm(hist) or 1.0
    return {f"lbp_{k:02d}": float(v) for k, v in enumerate(hist)}


def extract_all(gray: np.ndarray) -> dict[str, float]:
    return {**glcm_features(gray), **lbp_features(gray)}


# --- Scenario column sets --------------------------------------------------
def scenario_columns(columns) -> dict[str, list[str]]:
    cols = list(columns)
    s1 = [c for c in cols if c.startswith("glcm_") and c.endswith("_a0")]
    s2 = [c for c in cols if c.startswith("glcm_") and c.endswith("_mean")]
    s3 = [c for c in cols if c.startswith("lbp_")]
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s2 + s3}


def extract_from_file(path) -> dict[str, float]:
    from src.preprocess import preprocess

    return extract_all(preprocess(path))


def extract_table(meta: pd.DataFrame, raw_dir=config.RAW_DIR, n_jobs: int = -1) -> pd.DataFrame:
    """Extract features for every row of meta (needs a 'filename' column), in parallel."""
    from joblib import Parallel, delayed

    rows = Parallel(n_jobs=n_jobs)(delayed(extract_from_file)(raw_dir / fn) for fn in meta["filename"])
    return pd.concat([meta.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
