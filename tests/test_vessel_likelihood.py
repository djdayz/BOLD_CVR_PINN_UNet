import pytest

np = pytest.importorskip("numpy")

from hybrid_cvr.segmentation.vessel_likelihood import compute_vessel_likelihood


def test_vessel_likelihood_map_shape_and_range():
    brain = np.ones((6, 6), dtype="uint8")
    feature = np.zeros((6, 6), dtype=float)
    feature[2:4, 2:4] = 10
    out = compute_vessel_likelihood({"abs_cvr": feature, "r2": feature}, brain)
    score = out["vessel_likelihood"]
    assert score.shape == brain.shape
    assert score.min() >= 0
    assert score.max() <= 1
    assert out["vessel_like_high_confidence_mask"].shape == brain.shape
