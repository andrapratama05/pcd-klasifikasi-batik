import numpy as np
import pytest

from project.features import (
    glcm_matrices,
    glcm_props,
    homogeneity_paper,
    lbp_features,
)


def test_constant_image_glcm():
    img = np.full((32, 32), 120, dtype=np.uint8)
    props = glcm_props(glcm_matrices(img))
    assert np.allclose(props["contrast"], 0)
    assert np.allclose(props["asm"], 1)
    assert np.allclose(props["homogeneity"], 1)


def test_vertical_stripes_direction():
    img = np.zeros((32, 32), dtype=np.uint8)
    img[:, ::2] = 255  # alternating vertical lines
    props = glcm_props(glcm_matrices(img, distances=[1]))
    contrast_0, contrast_90 = props["contrast"][0, 0], props["contrast"][0, 2]
    assert contrast_0 > 10 * (contrast_90 + 1e-9)
    assert contrast_90 == pytest.approx(0)


def test_lbp_histogram_shape_and_norm():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    feats = lbp_features(img)
    v = np.array(list(feats.values()))
    assert len(v) == 59
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_homogeneity_hand_computed():
    # 3-level normalised GLCM, single distance/angle
    P = np.array([[0.2, 0.1, 0.0],
                  [0.1, 0.3, 0.1],
                  [0.0, 0.1, 0.1]])[:, :, None, None]
    # diagonal: (0.2+0.3+0.1)*1 ; |i-j|=1: (0.1*4)*1/2 ; |i-j|=2: 0
    expected = 0.6 + 0.4 / 2
    assert homogeneity_paper(P)[0, 0] == pytest.approx(expected)
