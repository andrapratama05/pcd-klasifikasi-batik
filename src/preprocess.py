"""Per-image preprocessing, identical for all scenarios.

center-crop to square -> resize to IMG_SIZE (anti-aliased) -> grayscale
(0.299R + 0.587G + 0.114B) -> CLAHE (clip_limit=0.01) -> uint8.
No sharpening. Every step uses only the pixels of the image itself, so no
information leaks between images.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist
from skimage.transform import resize

from src import config

GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114])


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64)


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return img[top:top + s, left:left + s]


def resize_square(img: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return resize(img, (size, size), anti_aliasing=True, preserve_range=True)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """MATLAB rgb2gray weights; returns uint8."""
    return np.clip(np.round(rgb @ GRAY_WEIGHTS), 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float = config.CLAHE_CLIP_LIMIT) -> np.ndarray:
    """CLAHE (equivalent to MATLAB adapthisteq defaults), back to uint8."""
    out = equalize_adapthist(gray, clip_limit=clip_limit)  # float in [0, 1]
    return np.clip(np.round(out * 255), 0, 255).astype(np.uint8)


def preprocess(path: Path, size: int = config.IMG_SIZE, return_steps: bool = False):
    """Full pipeline for one image file. With return_steps, also returns the intermediates."""
    rgb = resize_square(center_crop_square(load_rgb(path)), size)
    gray = to_gray(rgb)
    clahe = apply_clahe(gray)
    if return_steps:
        return {"rgb": rgb.astype(np.uint8), "gray": gray, "clahe": clahe}
    return clahe
