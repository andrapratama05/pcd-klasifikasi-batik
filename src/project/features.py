"""GLCM and LBP feature extraction following the reference papers / MATLAB defaults."""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from . import config


# --- GLCM --------------------------------------------------------------------
def quantize(gray: np.ndarray, levels: int = config.GLCM_LEVELS) -> np.ndarray:
    """Linearly map uint8 [0, 255] to [0, levels-1] (MATLAB GrayLimits=[0 255])."""
    return (gray.astype(np.int64) * levels // 256).astype(np.uint8)


def glcm_matrices(
    gray: np.ndarray,
    distances=config.GLCM_DISTANCES,
    angles=config.GLCM_ANGLES,
    levels: int = config.GLCM_LEVELS,
    symmetric: bool = config.GLCM_SYMMETRIC,
) -> np.ndarray:
    """Normalised GLCM with shape (levels, levels, n_distances, n_angles)."""
    q = quantize(gray, levels)
    return graycomatrix(q, distances=distances, angles=angles, levels=levels,
                        symmetric=symmetric, normed=True)


def homogeneity_paper(P: np.ndarray) -> np.ndarray:
    """Homogeneity with 1/(1+|i-j|) (paper eq. 8 / MATLAB), not skimage's 1/(1+(i-j)^2)."""
    levels = P.shape[0]
    i, j = np.ogrid[:levels, :levels]
    w = 1.0 / (1.0 + np.abs(i - j))
    return np.einsum("ij,ijda->da", w, P)


def glcm_props(P: np.ndarray) -> dict[str, np.ndarray]:
    """Return the 4 properties, each with shape (n_distances, n_angles)."""
    return {
        "contrast": graycoprops(P, "contrast"),
        "correlation": graycoprops(P, "correlation"),
        "asm": graycoprops(P, "ASM"),  # paper 'energy' = sum P^2 = skimage ASM
        "homogeneity": homogeneity_paper(P),
    }


def glcm_features(
    gray: np.ndarray,
    mode: str = "4dir",
    distances=config.GLCM_DISTANCES,
    levels: int = config.GLCM_LEVELS,
) -> dict[str, float]:
    """GLCM feature vector.

    mode='0deg' -> S1: angle 0 only, 4 props x len(distances) features.
    mode='4dir' -> S2: mean over 0/45/90/135 per distance, same dimensionality.
    """
    if mode == "0deg":
        angles, suffix = [0.0], "a0"
    elif mode == "4dir":
        angles, suffix = config.GLCM_ANGLES, "mean"
    else:
        raise ValueError("mode must be '0deg' or '4dir'")

    props = glcm_props(glcm_matrices(gray, distances, angles, levels))
    feats = {}
    for name in config.GLCM_PROPS:
        per_dist = props[name].mean(axis=1)
        for d, v in zip(distances, per_dist):
            feats[f"glcm_{name}_d{d}_{suffix}"] = float(v)
    return feats


# --- LBP ---------------------------------------------------------------------
def lbp_map(gray: np.ndarray, P: int = config.LBP_P, R: float = config.LBP_R) -> np.ndarray:
    """Per-pixel LBP code (0..58 for non-rotation-invariant uniform, P=8)."""
    return local_binary_pattern(gray, P, R, method=config.LBP_METHOD)


def lbp_features(gray: np.ndarray, P: int = config.LBP_P, R: float = config.LBP_R) -> dict[str, float]:
    """59-bin uniform LBP histogram, L2-normalised (MATLAB default)."""
    codes = lbp_map(gray, P, R)
    n_bins = P * (P - 1) + 3
    hist, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return {f"lbp_{k:02d}": float(v) for k, v in enumerate(hist)}
