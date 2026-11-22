"""Central configuration: paths, seed, and research-design parameters.

All paths are relative to the repository root so the project runs on any machine.
"""

import os
from pathlib import Path

import numpy as np

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATASET_ZIP_DIR = ROOT / "dataset"  # original ZIP downloaded from Mendeley (untouched)
DATA_DIR = ROOT / "data"
# Extracted images: <RAW_DIR>/<Class>/<file>.jpg. Override with BATIK_RAW_DIR, e.g. on
# Colab to keep 2k+ small files on fast local disk instead of Google Drive.
RAW_DIR = Path(os.environ.get("BATIK_RAW_DIR", DATA_DIR / "raw"))
FEATURES_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Classes -----------------------------------------------------------------
ALL_CLASSES = ["kawung", "megamendung", "parang", "sekarjagad", "truntum"]
MAIN_CLASSES = ["kawung", "megamendung", "parang"]  # used for the 12 main models

# --- Preprocessing -----------------------------------------------------------
IMAGE_SIZE = 224  # keep native size, no upscaling to 256
GRAY_WEIGHTS = (0.299, 0.587, 0.114)  # same as MATLAB rgb2gray
CLAHE_CLIP_LIMIT = 0.01  # MATLAB adapthisteq default

# --- GLCM --------------------------------------------------------------------
GLCM_LEVELS = 8  # MATLAB graycomatrix default NumLevels
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_ANGLE_NAMES = ["a0", "a45", "a90", "a135"]
GLCM_SYMMETRIC = False  # MATLAB default
GLCM_PROPS = ["contrast", "correlation", "asm", "homogeneity"]

# --- LBP ---------------------------------------------------------------------
LBP_P = 8
LBP_R = 1
LBP_METHOD = "nri_uniform"  # 59 patterns, same as MATLAB extractLBPFeatures(Upright=true)
LBP_BINS = 59

# --- SVM grid ----------------------------------------------------------------
SVM_C = [0.1, 1, 10, 100]
SVM_GAMMA = [0.001, 0.01, 0.1, 1]
SVM_DEGREE = [2, 3]
CV_SPLITS = 5
TEST_SIZE = 0.2


def ensure_dirs() -> None:
    """Create output folders if they do not exist yet."""
    for d in (DATA_DIR, RAW_DIR, FEATURES_DIR, RESULTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
