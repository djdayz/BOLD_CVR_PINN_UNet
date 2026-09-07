from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
torch = pytest.importorskip("torch")

from hybrid_cvr.inference.predict_sim import predict_sim_parameter_maps
from hybrid_cvr.training.train import train_unet_pinn


def _write_img(path: Path, data):
    img = nib.Nifti1Image(np.asarray(data), np.eye(4))
    nib.save(img, str(path))


def _write_parameter_case(case_dir: Path, shape=(12, 12, 3)):
    case_dir.mkdir(parents=True)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[3:9, 3:9, :] = 1
    mask = labels > 0
    cvr = np.zeros(shape, dtype=np.float32)
    delay = np.zeros(shape, dtype=np.float32)
    T = np.zeros(shape, dtype=np.float32)
    cvr[mask] = 0.3
    delay[mask] = 6.0
    T[mask] = 18.0
    _write_img(case_dir / "GT_CVR.nii.gz", cvr)
    _write_img(case_dir / "GT_delay.nii.gz", delay)
    _write_img(case_dir / "GT_T.nii.gz", T)
    _write_img(case_dir / "GT_region_labels.nii.gz", labels)
    _write_img(case_dir / "GT_vessel_likelihood.nii.gz", np.zeros(shape, dtype=np.float32))
    for idx, name in enumerate(["cortical_gm", "subcortical_gm", "wm", "vcsf", "vessel_like"]):
        frac = np.zeros(shape, dtype=np.float32)
        if idx == 0:
            frac[mask] = 1.0
        _write_img(case_dir / f"GT_fraction_{name}.nii.gz", frac)


def _tiny_config(tmp_path):
    return {
        "features": {"use_gt_maps_as_input": False},
        "dataset": {
            "sim_root": str(tmp_path / "sim"),
            "samples_per_epoch": 1,
            "n_timepoints": 12,
            "tr_seconds": 1.55,
            "temporal_mode": "windowed",
            "temporal_window_length": 6,
            "paradigms": ["block"],
            "tcnr_levels": [5.0],
            "etco2_input": "clean",
            "randomize_artifacts_for_training": False,
        },
        "model": {"base_channels": 2, "depth": 1, "dropout": 0.0, "T_mode": "global"},
        "training": {
            "batch_size": 1,
            "num_workers": 0,
            "use_amp": False,
            "gradient_clip_norm": 1.0,
            "use_supervised_parameter_loss": False,
        },
        "losses": {
            "lambda_recon": 1.0,
            "lambda_residual": 0.0,
            "lambda_uncertainty": 0.0,
            "lambda_smooth": 0.0,
            "lambda_prior": 0.0,
        },
    }


def test_predict_sim_writes_predicted_parameter_maps(tmp_path):
    _write_parameter_case(tmp_path / "sim" / "mida_parameters" / "case_000")
    cfg = _tiny_config(tmp_path)
    result = train_unet_pinn(cfg, tmp_path / "sim", tmp_path / "model", max_epochs=1)

    pred = predict_sim_parameter_maps(
        result["best_checkpoint"],
        cfg,
        tmp_path / "sim",
        tmp_path / "pred",
        split="test",
        case_id="case_000",
        paradigm="block",
        tcnr=5.0,
    )

    assert pred["processed_slices"] == 3
    for name in [
        "predicted_CVR",
        "predicted_delay",
        "predicted_T",
        "predicted_uncertainty_sigma",
    ]:
        path = pred[name]
        assert path.exists()
        img = nib.load(str(path))
        assert img.shape == (12, 12, 3)
        assert np.isfinite(img.get_fdata()).all()
    assert pred["metadata"].exists()
    assert pred["slice_summary"].exists()
