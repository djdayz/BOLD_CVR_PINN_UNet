import pytest

np = pytest.importorskip("numpy")

from hybrid_cvr.preprocessing.bold import percent_signal_change, simple_epi_mask


def test_bold_psc_conversion_and_mask_shape():
    data = np.ones((4, 4, 2, 5), dtype="float32") * 100.0
    data[..., 2:] = 110.0
    psc = percent_signal_change(data, baseline_volumes=2)
    assert psc.shape == data.shape
    assert np.isclose(psc[..., 0].mean(), 0.0)
    assert np.isclose(psc[..., -1].mean(), 10.0)
    mask = simple_epi_mask(data.mean(axis=-1))
    assert mask.shape == data.shape[:3]


def test_bold_psc_zero_baseline_stays_zero():
    data = np.zeros((2, 2, 1, 3), dtype="float32")
    data[0, 0, 0, :] = [0, 10, 20]
    psc = percent_signal_change(data, baseline_volumes=1)
    assert np.all(np.isfinite(psc))
    assert np.all(psc == 0)


def test_preprocess_bold_masks_psc_outside_brain(tmp_path):
    nib = pytest.importorskip("nibabel")
    from hybrid_cvr.preprocessing.bold import BoldPreprocessConfig, preprocess_bold_file

    data = np.zeros((4, 4, 1, 4), dtype="float32")
    data[1:3, 1:3, 0, :] = 100.0
    data[1:3, 1:3, 0, 2:] = 110.0
    data[0, 0, 0, :] = [0.1, 0.1, 100.0, 100.0]
    img = nib.Nifti1Image(data, np.eye(4))
    path = tmp_path / "bold.nii.gz"
    nib.save(img, path)

    out = preprocess_bold_file(path, tmp_path / "out", BoldPreprocessConfig(baseline_volumes=2))
    psc = nib.load(out["bold_psc"]).get_fdata(dtype=np.float32)
    mask = nib.load(out["brain_mask"]).get_fdata(dtype=np.float32)
    assert mask[0, 0, 0] == 0
    assert np.all(psc[0, 0, 0, :] == 0)
    assert "valid_signal_mask" in out
