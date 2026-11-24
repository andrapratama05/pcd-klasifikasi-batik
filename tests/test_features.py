import numpy as np
import pytest

from src import config
from src.features import (
    extract_all,
    glcm_features,
    glcm_props,
    homogeneity_abs,
    lbp_features,
    quantize,
    scenario_columns,
)


def test_constant_image_glcm():
    f = glcm_features(np.full((32, 32), 100, dtype=np.uint8))
    for d in config.GLCM_DISTANCES:
        for a in [*config.GLCM_ANGLE_NAMES, "mean"]:
            assert f[f"glcm_contrast_d{d}_{a}"] == pytest.approx(0)
            assert f[f"glcm_asm_d{d}_{a}"] == pytest.approx(1)
            assert f[f"glcm_homogeneity_d{d}_{a}"] == pytest.approx(1)


def test_vertical_stripes_contrast_depends_on_angle():
    # columns alternate 0 / 255 -> horizontal neighbours always differ, vertical never
    img = np.tile(np.array([0, 255], dtype=np.uint8), (32, 16))
    f = glcm_features(img)
    assert f["glcm_contrast_d1_a0"] > 10
    assert f["glcm_contrast_d1_a90"] == pytest.approx(0)
    assert f["glcm_contrast_d1_a0"] > 10 * (f["glcm_contrast_d1_a90"] + 1e-9)


def test_lbp_histogram_59_bins_unit_norm():
    rng = np.random.default_rng(0)
    f = lbp_features(rng.integers(0, 256, (64, 64), dtype=np.uint8))
    v = np.array(list(f.values()))
    assert len(v) == 59
    assert np.linalg.norm(v) == pytest.approx(1)


def test_homogeneity_manual_matches_hand_computation():
    # 3x3 normalised matrix, one distance and one angle
    from skimage.feature import graycoprops

    P = np.array([[0.20, 0.10, 0.05],
                  [0.10, 0.20, 0.10],
                  [0.05, 0.00, 0.20]])[:, :, None, None]
    # |i-j|=0: 0.6 * 1;  |i-j|=1: 0.3 * 1/2;  |i-j|=2: 0.1 * 1/3
    expected = 0.6 + 0.3 / 2 + 0.1 / 3
    assert homogeneity_abs(P)[0, 0] == pytest.approx(expected)
    # skimage's 'homogeneity' uses 1/(1+(i-j)^2): 0.6 + 0.3/2 + 0.1/5, so it must differ
    assert graycoprops(P, "homogeneity")[0, 0] == pytest.approx(0.6 + 0.3 / 2 + 0.1 / 5)
    assert homogeneity_abs(P)[0, 0] != pytest.approx(graycoprops(P, "homogeneity")[0, 0])


def test_asm_is_sum_of_squares():
    rng = np.random.default_rng(1)
    P = rng.random((8, 8, 1, 1))
    P /= P.sum()
    assert glcm_props(P)["asm"][0, 0] == pytest.approx((P ** 2).sum())


def test_quantize_8_levels():
    q = quantize(np.arange(256, dtype=np.uint8))
    assert q.min() == 0 and q.max() == 7
    assert np.bincount(q).tolist() == [32] * 8


def test_scenario_dimensions():
    cols = extract_all(np.random.default_rng(2).integers(0, 256, (64, 64), dtype=np.uint8)).keys()
    dims = {k: len(v) for k, v in scenario_columns(cols).items()}
    assert dims == {"S1": 12, "S2": 12, "S3": 59, "S4": 71}


def test_feature_pipeline_uses_abs_homogeneity():
    """glcm_features must use 1/(1+|i-j|), not skimage's 1/(1+(i-j)^2)."""
    from src.features import glcm_matrix

    img = np.random.default_rng(3).integers(0, 256, (64, 64), dtype=np.uint8)
    P = glcm_matrix(img)[:, :, 0, 0]
    i, j = np.indices(P.shape)
    expected = (P / (1 + np.abs(i - j))).sum()
    squared = (P / (1 + (i - j) ** 2)).sum()
    f = glcm_features(img)
    assert f["glcm_homogeneity_d1_a0"] == pytest.approx(expected)
    assert abs(expected - squared) > 1e-3  # the two definitions are distinguishable here
