"""Preprocessing: load RGB -> (size check) -> grayscale -> CLAHE (uint8).

No sharpening (design decision: sharpening amplifies noise that corrupts LBP codes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage import exposure, img_as_ubyte

from . import config


def load_rgb(path: str | Path) -> np.ndarray:
    """Load any image as uint8 RGB (RGBA/palette/grayscale are converted)."""
    path = Path(path)
    if not path.is_absolute():
        path = config.ROOT / path
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def ensure_size(rgb: np.ndarray, size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Return the image unchanged if it is size x size, else center-crop + resize."""
    h, w = rgb.shape[:2]
    if (h, w) == (size, size):
        return rgb
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    crop = rgb[top : top + s, left : left + s]
    return np.asarray(Image.fromarray(crop).resize((size, size), Image.Resampling.BICUBIC))


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Weighted grayscale 0.299R + 0.587G + 0.114B, rounded to uint8 (like rgb2gray)."""
    w = np.asarray(config.GRAY_WEIGHTS)
    gray = rgb[..., :3].astype(np.float64) @ w
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def apply_clahe(gray: np.ndarray, clip_limit: float = config.CLAHE_CLIP_LIMIT) -> np.ndarray:
    """CLAHE with 8x8 tiles (skimage default kernel = 1/8 of image), back to uint8."""
    out = exposure.equalize_adapthist(gray, clip_limit=clip_limit)
    return img_as_ubyte(out)


def preprocess_stages(path: str | Path) -> dict[str, np.ndarray]:
    """Return every intermediate stage, handy for visualisation."""
    rgb = ensure_size(load_rgb(path))
    gray = to_gray(rgb)
    clahe = apply_clahe(gray)
    return {"rgb": rgb, "gray": gray, "clahe": clahe}


def preprocess(path: str | Path) -> np.ndarray:
    """Full preprocessing pipeline; returns the uint8 CLAHE image used for features."""
    return preprocess_stages(path)["clahe"]
