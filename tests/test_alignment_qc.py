from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
pytest.importorskip("matplotlib")

from hybrid_cvr.cvr.alignment_qc import save_fit_alignment_plot
from hybrid_cvr.cvr.fit_real import RealCVRFitConfig
from hybrid_cvr.cvr.ode import solve_ode_response_numpy


def test_save_fit_alignment_plot(tmp_path: Path):
    time = np.arange(32, dtype=np.float32) * 1.55
    etco2 = np.zeros_like(time)
    etco2[8:20] = 6.0
    cvr = np.ones((2, 2, 1), dtype=np.float32) * 0.25
    delay = np.ones_like(cvr) * 3.1
    T = np.ones_like(cvr) * 6.0
    psc = np.moveaxis(solve_ode_response_numpy(cvr, delay, T, etco2, time), 0, -1).astype(
        np.float32
    )
    mask = np.ones(cvr.shape, dtype=np.uint8)
    affine = np.eye(4)
    psc_path = tmp_path / "bold_psc.nii.gz"
    mask_path = tmp_path / "valid_signal_mask.nii.gz"
    etco2_path = tmp_path / "etco2_resampled.tsv"
    out = tmp_path / "fit_alignment_qc.png"
    nib.save(nib.Nifti1Image(psc, affine), psc_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)
    with etco2_path.open("w", encoding="utf-8") as f:
        f.write("time\tsource_time\tetco2_mmhg\tdelta_etco2_mmhg\tbaseline_mmhg\n")
        for t, u in zip(time, etco2, strict=False):
            f.write(f"{t}\t{t}\t{u + 35.0}\t{u}\t35.0\n")

    outputs = save_fit_alignment_plot(
        psc_path,
        etco2_path,
        mask_path,
        out,
        RealCVRFitConfig(
            delay_min=0,
            delay_max=6.2,
            delay_step=1.55,
            T_min=4,
            T_max=8,
            T_step=2,
            glm_delay_min=-3.1,
            glm_delay_max=9.3,
            glm_delay_step=1.55,
        ),
        "synthetic fit alignment",
    )

    assert out.exists()
    assert out.stat().st_size > 0
    assert Path(outputs["alignment_qc"]) == out
    assert outputs["hrf_T_seconds"] == pytest.approx(6.0, abs=2.1)
