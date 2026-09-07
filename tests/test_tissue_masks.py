import pytest

np = pytest.importorskip("numpy")

from hybrid_cvr.segmentation.anatomical import fallback_bold_tissue_segmentation
from hybrid_cvr.segmentation.boundary_uncertainty import tissue_boundary_uncertainty
from hybrid_cvr.segmentation.fastsurfer import extract_fastsurfer_masks
from hybrid_cvr.segmentation.tissue_masks import high_confidence_core


def test_tissue_mask_core_and_boundary_shape():
    gm = np.zeros((8, 8), dtype=float)
    wm = np.zeros((8, 8), dtype=float)
    csf = np.zeros((8, 8), dtype=float)
    gm[1:5, 1:5] = 1
    wm[4:7, 4:7] = 1
    csf[0, 0] = 1
    core = high_confidence_core(gm, erosion_iters=1)
    boundary = tissue_boundary_uncertainty({"gm": gm, "wm": wm, "csf": csf})
    assert core.shape == gm.shape
    assert boundary.shape == gm.shape
    assert boundary.max() <= 1.0


def test_fallback_bold_tissue_segmentation_shapes_and_ranges():
    mean_bold = np.ones((8, 8, 8), dtype="float32") * 0.7
    mean_bold[:2] = 0.45
    mean_bold[3:5, 3:5, 3:5] = 0.05
    mean_bold[6:] = 0.95
    brain = np.ones((8, 8, 8), dtype="uint8")
    out = fallback_bold_tissue_segmentation(mean_bold, brain)
    for key in ["gm_prob_or_mask", "wm_prob_or_mask", "csf_prob_or_mask", "tissue_boundary_uncertainty"]:
        assert out[key].shape == brain.shape
        assert out[key].min() >= 0
        assert out[key].max() <= 1
    binary_keys = [
        "gm_mask",
        "wm_mask",
        "csf_mask",
        "cortical_gm_mask",
        "subcortical_gm_mask",
        "gm_core",
        "wm_core",
        "csf_core",
        "cortical_gm_core",
        "subcortical_gm_core",
    ]
    for key in binary_keys:
        assert out[key].shape == brain.shape
        assert set(np.unique(out[key])).issubset({0, 1})
    assert np.all(out["cortical_gm_mask"] + out["subcortical_gm_mask"] == out["gm_mask"])
    assert out["cortical_gm_mask"].sum() > 0
    assert out["subcortical_gm_mask"].sum() > 0
    probs = out["gm_prob_or_mask"] + out["wm_prob_or_mask"] + out["csf_prob_or_mask"]
    assert np.allclose(probs[brain > 0], 1.0)
    assert out["wm_core"].sum() > 0
    assert out["ventricle_mask"].sum() > 0


def test_extract_fastsurfer_masks_from_labels(tmp_path):
    nib = pytest.importorskip("nibabel")

    mri_dir = tmp_path / "sub-01" / "mri"
    mri_dir.mkdir(parents=True)
    labels = np.zeros((4, 4, 4), dtype=np.int16)
    labels[0, 0, 0] = 2
    labels[1, 0, 0] = 10
    labels[2, 0, 0] = 4
    labels[3, 0, 0] = 1001
    img = nib.MGHImage(labels, np.eye(4))
    nib.save(img, str(mri_dir / "aparc.DKTatlas+aseg.deep.mgz"))

    outputs = extract_fastsurfer_masks(mri_dir, tmp_path / "masks")

    assert set(outputs) == {"wm_mask", "subcortical_gm_mask", "vcsf_mask", "cortical_gm_mask"}
    counts = {name: int(nib.load(str(path)).get_fdata().sum()) for name, path in outputs.items()}
    assert counts == {
        "wm_mask": 1,
        "subcortical_gm_mask": 1,
        "vcsf_mask": 1,
        "cortical_gm_mask": 1,
    }
