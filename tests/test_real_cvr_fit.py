from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")

from hybrid_cvr.cvr.fit_real import RealCVRFitConfig, fit_real_cvr_session
from hybrid_cvr.cvr.ode import solve_ode_response_numpy


def test_fit_real_cvr_session_writes_parameter_maps(tmp_path: Path):
    time = np.arange(24, dtype=np.float32) * 1.5
    etco2 = np.zeros_like(time)
    etco2[6:] = 5.0
    cvr = np.zeros((3, 3, 2), dtype=np.float32)
    delay = np.zeros_like(cvr)
    T = np.ones_like(cvr) * 6.0
    mask = np.zeros_like(cvr, dtype=np.uint8)
    mask[1, 1, 0] = 1
    cvr[mask > 0] = 0.3
    psc = solve_ode_response_numpy(cvr, delay, T, etco2, time).astype(np.float32)
    psc = np.moveaxis(psc, 0, -1)
    affine = np.eye(4)
    psc_path = tmp_path / "bold_psc.nii.gz"
    mean_path = tmp_path / "mean_bold.nii.gz"
    mask_path = tmp_path / "valid_signal_mask.nii.gz"
    etco2_path = tmp_path / "etco2_resampled.tsv"
    nib.save(nib.Nifti1Image(psc, affine), psc_path)
    nib.save(nib.Nifti1Image(np.ones(cvr.shape, dtype=np.float32), affine), mean_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    with etco2_path.open("w", encoding="utf-8") as f:
        f.write("time\tetco2_mmhg\tdelta_etco2_mmhg\tbaseline_mmhg\n")
        for t, u in zip(time, etco2, strict=False):
            f.write(f"{t}\t{u + 35.0}\t{u}\t35.0\n")

    outputs = fit_real_cvr_session(
        psc_path,
        etco2_path,
        mask_path,
        mean_path,
        tmp_path / "real_cvr",
        RealCVRFitConfig(
            delay_min=0,
            delay_max=3,
            delay_step=1.5,
            T_min=4,
            T_max=8,
            T_step=2,
            chunk_voxels=2,
            candidate_batch_size=2,
        ),
    )

    for key in [
        "valid_fit_mask",
        "glm_cvr",
        "glm_delay",
        "glm_r2",
        "glm_rmse",
        "hrf_cvr",
        "hrf_delay",
        "hrf_T",
        "hrf_effective_delay",
        "hrf_r2",
        "hrf_rmse",
        "tCNR",
        "etco2_correlation_peak",
        "fit_quality",
        "hrf_residual_std",
        "ode_residual_rms",
    ]:
        assert Path(outputs[key]).exists()
    hrf_cvr = nib.load(str(outputs["hrf_cvr"])).get_fdata()
    hrf_T = nib.load(str(outputs["hrf_T"])).get_fdata()
    assert hrf_cvr[1, 1, 0] == pytest.approx(0.3, abs=0.08)
    assert hrf_T[1, 1, 0] == pytest.approx(6.0, abs=2.1)
