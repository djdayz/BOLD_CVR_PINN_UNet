import pytest

np = pytest.importorskip("numpy")

from hybrid_cvr.simulation.mida import (  # noqa: E402
    block_average,
    dominant_region_label,
    fill_lowres_brain_support,
    normalize_fraction_maps,
    patchwise_rows,
    sample_partial_volume_maps,
    smooth_parameter_maps_in_support,
    softly_fill_internal_holes,
    subvoxel_average_rows,
)


def test_block_average_preserves_partial_volume_fraction():
    mask = np.zeros((4, 4, 4), dtype=np.float32)
    mask[:2, :2, :2] = 1.0
    mask[2:, 2:, 2:] = 0.5

    lowres = block_average(mask, (2, 2, 2))

    assert lowres.shape == (2, 2, 2)
    assert lowres[0, 0, 0] == pytest.approx(1.0)
    assert lowres[1, 1, 1] == pytest.approx(0.5)
    assert lowres[0, 1, 0] == pytest.approx(0.0)


def test_soft_fill_assigns_internal_hole_without_vessel_leakage():
    masks = {
        "cortical_gm": np.zeros((5, 5, 5), dtype=bool),
        "subcortical_gm": np.zeros((5, 5, 5), dtype=bool),
        "wm": np.zeros((5, 5, 5), dtype=bool),
        "vcsf": np.zeros((5, 5, 5), dtype=bool),
        "vessel_like": np.zeros((5, 5, 5), dtype=bool),
    }
    masks["cortical_gm"][1:4, 1:4, 1:4] = True
    masks["cortical_gm"][2, 2, 2] = False
    brain_support = np.zeros((5, 5, 5), dtype=bool)
    brain_support[1:4, 1:4, 1:4] = True

    filled = softly_fill_internal_holes(masks, brain_support, sigma_vox=(0.5, 0.5, 0.5))

    tissue_sum = sum(filled[name][2, 2, 2] for name in ("cortical_gm", "subcortical_gm", "wm", "vcsf"))
    assert tissue_sum == pytest.approx(1.0)
    assert filled["vessel_like"][2, 2, 2] == pytest.approx(0.0)


def test_fraction_normalization_and_dominant_labels():
    fractions = {
        "cortical_gm": np.array([[[0.6, 0.0]]], dtype=np.float32),
        "subcortical_gm": np.array([[[0.2, 0.0]]], dtype=np.float32),
        "wm": np.array([[[0.2, 0.0]]], dtype=np.float32),
        "vcsf": np.array([[[0.0, 0.0]]], dtype=np.float32),
        "vessel_like": np.array([[[0.0, 0.0]]], dtype=np.float32),
    }

    normalized = normalize_fraction_maps(fractions, min_support=0.05)
    dominant = dominant_region_label(normalized, sum(normalized.values()) > 0)

    assert sum(m[0, 0, 0] for m in normalized.values()) == pytest.approx(1.0)
    assert sum(m[0, 0, 1] for m in normalized.values()) == pytest.approx(0.0)
    assert dominant[0, 0, 0] == 1
    assert dominant[0, 0, 1] == 0


def test_lowres_brain_support_fills_slice_visible_internal_holes():
    support = np.zeros((7, 7, 3), dtype=bool)
    support[1:6, 1:6, 1] = True
    support[3, 3, 1] = False

    filled = fill_lowres_brain_support(support)

    assert filled[3, 3, 1]
    assert filled.sum() == support.sum() + 1


def test_partial_volume_sampling_is_reproducible_and_weighted():
    fractions = {
        "cortical_gm": np.array([[[0.75]]], dtype=np.float32),
        "subcortical_gm": np.array([[[0.0]]], dtype=np.float32),
        "wm": np.array([[[0.25]]], dtype=np.float32),
        "vcsf": np.array([[[0.0]]], dtype=np.float32),
        "vessel_like": np.array([[[0.0]]], dtype=np.float32),
    }
    samples = {
        "cortical_gm": np.array([[1.0, 10.0, 20.0]], dtype=np.float32),
        "subcortical_gm": np.array([[2.0, 20.0, 30.0]], dtype=np.float32),
        "wm": np.array([[0.2, 30.0, 40.0]], dtype=np.float32),
        "vcsf": np.array([[0.0, 0.0, 10.0]], dtype=np.float32),
        "vessel_like": np.array([[1.5, 5.0, 5.0]], dtype=np.float32),
    }

    maps = sample_partial_volume_maps(fractions, samples, np.random.default_rng(17))

    assert maps["CVR"][0, 0, 0] == pytest.approx(0.8)
    assert maps["delay"][0, 0, 0] == pytest.approx(15.0)
    assert maps["T"][0, 0, 0] == pytest.approx(25.0)


def test_patchwise_sampling_gives_multiple_values_inside_one_tissue():
    rows = np.array(
        [
            [0.1, 0.0, 2.0],
            [0.2, 40.0, 20.0],
            [0.3, 80.0, 100.0],
        ],
        dtype=np.float32,
    )
    keep = np.ones((6, 6, 4), dtype=bool)

    samples = patchwise_rows(rows, keep, np.random.default_rng(17), (3, 3, 2))

    assert samples.shape == (144, 3)
    assert np.unique(samples[:, 1]).size > 1
    assert np.unique(samples[:, 2]).size > 1


def test_subvoxel_average_rows_averages_multiple_draws():
    rows = np.array(
        [
            [0.0, 0.0, 2.0],
            [2.0, 80.0, 100.0],
        ],
        dtype=np.float32,
    )
    fractions = np.ones(200, dtype=np.float32)

    samples = subvoxel_average_rows(rows, fractions, np.random.default_rng(17), n_subvoxels=125)

    assert samples.shape == (200, 3)
    assert samples[:, 0].std() < 0.25
    assert 0.5 < samples[:, 0].mean() < 1.5
    assert np.unique(np.round(samples[:, 0], 2)).size > 1


def test_parameter_smoothing_preserves_sampled_distribution():
    maps = {
        "CVR": np.zeros((5, 5, 1), dtype=np.float32),
        "delay": np.zeros((5, 5, 1), dtype=np.float32),
        "T": np.zeros((5, 5, 1), dtype=np.float32),
    }
    for values in maps.values():
        values[2, 2, 0] = 10.0
    support = np.ones((5, 5, 1), dtype=bool)

    smoothed = smooth_parameter_maps_in_support(maps, support, sigma_vox=0.75, blend=1.0)

    assert np.sort(smoothed["T"][support]).tolist() == pytest.approx(
        np.sort(maps["T"][support]).tolist()
    )
    assert smoothed["T"].max() == pytest.approx(10.0)


def test_stratified_sampling_uses_full_delay_and_T_ranges():
    fractions = {
        "cortical_gm": np.ones((12, 1, 1), dtype=np.float32),
        "subcortical_gm": np.zeros((12, 1, 1), dtype=np.float32),
        "wm": np.zeros((12, 1, 1), dtype=np.float32),
        "vcsf": np.zeros((12, 1, 1), dtype=np.float32),
        "vessel_like": np.zeros((12, 1, 1), dtype=np.float32),
    }
    central_samples = {
        region: np.array([[0.2, 0.0, 20.0]], dtype=np.float32)
        for region in fractions
    }
    full_delay_samples = {
        region: np.array([[0.2, 0.0, 2.0], [0.2, 10.0, 20.0], [0.2, 80.0, 100.0]], dtype=np.float32)
        for region in fractions
    }

    maps = sample_partial_volume_maps(
        fractions,
        central_samples,
        np.random.default_rng(17),
        delay_region_samples=full_delay_samples,
        delay_sampling_mode="tissue_stratified_full",
        T_region_samples=full_delay_samples,
        T_sampling_mode="tissue_stratified_full",
    )

    assert maps["delay"].min() == pytest.approx(0.0)
    assert maps["delay"].max() == pytest.approx(80.0)
    assert maps["T"].min() == pytest.approx(2.0)
    assert maps["T"].max() == pytest.approx(100.0)
