from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")

from hybrid_cvr.simulation.co2_paradigms import block_paradigm, make_paradigm
from hybrid_cvr.simulation.mida_bold import FEATURE_NAMES, MidaBoldSimulationConfig, simulate_mida_bold_dataset


def _write_img(path: Path, data):
    img = nib.Nifti1Image(np.asarray(data), np.eye(4))
    nib.save(img, str(path))


def _write_parameter_case(case_dir: Path, shape=(8, 8, 3)):
    case_dir.mkdir()
    labels = np.zeros(shape, dtype=np.uint8)
    labels[2:6, 2:6, :] = 1
    mask = labels > 0
    cvr = np.zeros(shape, dtype=np.float32)
    delay = np.zeros(shape, dtype=np.float32)
    T = np.zeros(shape, dtype=np.float32)
    cvr[mask] = 0.2
    delay[mask] = 8.0
    T[mask] = 20.0
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


def test_new_co2_paradigms_are_finite_and_smooth_length():
    time = np.arange(40, dtype=np.float32) * 1.55
    for name in [
        "block",
        "ramp",
        "multi_step",
        "pseudo_random_binary",
        "sinusoidal",
        "breath_hold_like",
        "resting_state_like",
    ]:
        u = make_paradigm(name, time, seed=17)
        assert u.shape == time.shape
        assert np.all(np.isfinite(u))


def test_block_paradigm_has_two_smooth_blocks_by_default():
    time = np.arange(0, 520, 1, dtype=np.float32)
    u = block_paradigm(time)
    high = u > 4.0
    starts = np.flatnonzero(high[1:] & ~high[:-1])
    stops = np.flatnonzero(~high[1:] & high[:-1])
    assert starts.size == 2
    assert stops.size == 2


def test_simulate_mida_bold_dataset_writes_npz_etco2_and_summary(tmp_path):
    case_dir = tmp_path / "params"
    _write_parameter_case(case_dir)

    out_dir = tmp_path / "sim"
    result = simulate_mida_bold_dataset(
        MidaBoldSimulationConfig(
            parameter_case_dir=case_dir,
            output_dir=out_dir,
            output_mode="slice_npz",
            tcnr_levels=(0.5, 2.0),
            paradigms=("block", "ramp"),
            slice_indices=(1,),
            n_timepoints=24,
            seed=11,
            motion_spike_probability=0.0,
        )
    )

    assert result["n_cases"] == 4
    assert (out_dir / "simulation_summary.csv").exists()
    first = np.load(out_dir / "sim_000.npz")
    assert first["bold_psc"].shape == (24, 8, 8)
    assert first["features"].shape == (len(FEATURE_NAMES), 8, 8)
    assert first["tissue_maps"].shape == (5, 8, 8)
    assert (out_dir / "etco2" / "sim_000_etco2.tsv").exists()
    assert (out_dir / "qc" / "sim_000_etco2.png").exists()


def test_simulate_mida_bold_dataset_writes_4d_bold_without_sim_gt_exports(tmp_path):
    case_dir = tmp_path / "params"
    _write_parameter_case(case_dir, shape=(6, 6, 2))

    out_dir = tmp_path / "sim4d"
    result = simulate_mida_bold_dataset(
        MidaBoldSimulationConfig(
            parameter_case_dir=case_dir,
            output_dir=out_dir,
            output_mode="volume4d",
            tcnr_levels=(1.0,),
            paradigms=("block",),
            n_timepoints=12,
            seed=11,
            motion_spike_probability=0.0,
            save_bold_psc_nifti=True,
            save_bold_intensity_nifti=True,
        )
    )

    assert result["n_cases"] == 1
    bold = nib.load(str(out_dir / "bold4d" / "sim_000_block_tcnr-1_bold.nii.gz"))
    psc = nib.load(str(out_dir / "bold4d" / "sim_000_block_tcnr-1_bold_psc.nii.gz"))
    assert bold.shape == (6, 6, 2, 12)
    assert psc.shape == (6, 6, 2, 12)
    assert not list((out_dir / "nifti").glob("*gt*.nii.gz"))
    assert (out_dir / "simulation_summary.csv").exists()
    summary = np.genfromtxt(out_dir / "simulation_summary.csv", delimiter=",", names=True, dtype=None, encoding=None)
    assert summary["file_stem"] == "sim_000_block_tcnr-1"
