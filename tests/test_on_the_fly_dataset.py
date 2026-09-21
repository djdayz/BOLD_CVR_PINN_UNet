from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")
torch = pytest.importorskip("torch")

from hybrid_cvr.training.on_the_fly_dataset import (
    OnTheFlyCVRDataset,
    OnTheFlyDatasetConfig,
    assert_feature_names_are_observable,
)
from hybrid_cvr.training.train import train_physiology_model, validate_training_config


def _write_img(path: Path, data):
    img = nib.Nifti1Image(np.asarray(data), np.eye(4))
    nib.save(img, str(path))


def _write_parameter_case(case_dir: Path, shape=(10, 10, 3)):
    case_dir.mkdir(parents=True)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[2:8, 2:8, :] = 1
    mask = labels > 0
    cvr = np.zeros(shape, dtype=np.float32)
    delay = np.zeros(shape, dtype=np.float32)
    T = np.zeros(shape, dtype=np.float32)
    cvr[mask] = 0.3
    delay[mask] = 5.0
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


def test_on_the_fly_dataset_returns_shapes_without_saved_4d_bold(tmp_path):
    case_dir = tmp_path / "sim" / "mida_parameters" / "case_000"
    _write_parameter_case(case_dir)
    dataset = OnTheFlyCVRDataset(
        OnTheFlyDatasetConfig(
            sim_root=tmp_path / "sim",
            samples_per_epoch=2,
            n_timepoints=24,
            tr_seconds=1.55,
            temporal_mode="windowed",
            temporal_window_length=12,
            paradigms=("block", "ramp"),
            tcnr_levels=(2.0,),
            etco2_input="mixed",
        )
    )
    sample = dataset[0]
    assert sample["features"].shape[0] == len(dataset.feature_names)
    assert sample["bold_psc"].shape == (12, 10, 10)
    assert sample["etco2_for_model"].shape == (12,)
    assert sample["transition_weight"].shape == (12,)
    assert sample["tissue_maps"].shape == (5, 10, 10)
    assert torch.isfinite(sample["features"]).all()
    assert not list((tmp_path / "sim").glob("**/*_bold.nii.gz"))
    assert_feature_names_are_observable(dataset.feature_names)
    assert not any("gt" in name.lower() for name in dataset.feature_names)
    assert {"gt_cvr", "gt_delay", "gt_T"}.issubset(sample)


def test_transition_weights_are_larger_near_etco2_changes():
    etco2 = np.zeros(20, dtype=np.float32)
    etco2[8:] = 5.0
    weights = OnTheFlyCVRDataset.compute_transition_weight(etco2, alpha=2.0)
    assert weights[8] > weights[0]
    assert np.isfinite(weights).all()


def test_one_case_train_and_validation_use_different_seed_streams(tmp_path):
    case_dir = tmp_path / "sim" / "mida_parameters" / "case_000"
    _write_parameter_case(case_dir)
    base = {
        "sim_root": tmp_path / "sim",
        "samples_per_epoch": 1,
        "n_timepoints": 24,
        "temporal_window_length": 12,
        "paradigms": ("block", "ramp"),
        "tcnr_levels": (2.0,),
    }
    train_sample = OnTheFlyCVRDataset(OnTheFlyDatasetConfig(**base, split="train"))[0]
    val_sample = OnTheFlyCVRDataset(OnTheFlyDatasetConfig(**base, split="val"))[0]
    assert train_sample["metadata"]["case_id"] == "case_000"
    assert val_sample["metadata"]["case_id"] == "case_000"
    assert train_sample["metadata"]["noise_seed"] != val_sample["metadata"]["noise_seed"]


def test_balanced_grid_covers_each_paradigm_and_tcnr(tmp_path):
    case_dir = tmp_path / "sim" / "mida_parameters" / "case_000"
    _write_parameter_case(case_dir)
    paradigms = ("block", "ramp")
    tcnr_levels = (1.0, 5.0)
    dataset = OnTheFlyCVRDataset(
        OnTheFlyDatasetConfig(
            sim_root=tmp_path / "sim",
            samples_per_epoch=1,
            n_timepoints=16,
            temporal_window_length=8,
            paradigms=paradigms,
            tcnr_levels=tcnr_levels,
            sampling_strategy="balanced_grid",
        )
    )
    seen = {(dataset[i]["metadata"]["paradigm"], dataset[i]["metadata"]["target_tcnr"]) for i in range(len(dataset))}
    assert len(dataset) == len(paradigms) * len(tcnr_levels)
    assert seen == {(paradigm, tcnr) for paradigm in paradigms for tcnr in tcnr_levels}


def test_weighted_tcnr_sampling_reduces_very_low_tcnr_frequency(tmp_path):
    case_dir = tmp_path / "sim" / "mida_parameters" / "case_000"
    _write_parameter_case(case_dir)
    dataset = OnTheFlyCVRDataset(
        OnTheFlyDatasetConfig(
            sim_root=tmp_path / "sim",
            samples_per_epoch=1000,
            n_timepoints=8,
            temporal_window_length=8,
            paradigms=("block",),
            tcnr_levels=(0.1, 0.2, 1.0, 2.0),
            tcnr_probabilities=(0.05, 0.05, 0.45, 0.45),
            sampling_strategy="random",
        )
    )
    values = [dataset._choose_condition(i, np.random.default_rng(i))[1] for i in range(1000)]
    very_low = sum(value in {0.1, 0.2} for value in values)
    assert 50 <= very_low <= 160


def test_training_artifact_sampling_changes_by_epoch_but_validation_is_fixed(tmp_path):
    case_dir = tmp_path / "sim" / "mida_parameters" / "case_000"
    _write_parameter_case(case_dir)
    base = {
        "sim_root": tmp_path / "sim",
        "samples_per_epoch": 1,
        "n_timepoints": 24,
        "temporal_window_length": 12,
        "paradigms": ("block",),
        "tcnr_levels": (2.0,),
        "randomize_artifacts_for_training": True,
        "motion_spike_probability_options": (0.0, 0.03),
    }

    train_dataset = OnTheFlyCVRDataset(OnTheFlyDatasetConfig(**base, split="train"))
    train_dataset.set_epoch(1)
    first = train_dataset[0]["metadata"]
    train_dataset.set_epoch(2)
    second = train_dataset[0]["metadata"]
    assert first["noise_seed"] != second["noise_seed"]
    assert first["etco2_noise_sd_mmhg"] != second["etco2_noise_sd_mmhg"]

    val_dataset = OnTheFlyCVRDataset(OnTheFlyDatasetConfig(**base, split="val"))
    val_dataset.set_epoch(1)
    val_first = val_dataset[0]["metadata"]
    val_dataset.set_epoch(99)
    val_second = val_dataset[0]["metadata"]
    assert val_first["noise_seed"] == val_second["noise_seed"]
    assert val_first["etco2_noise_sd_mmhg"] == val_second["etco2_noise_sd_mmhg"]


def test_training_config_rejects_supervised_or_gt_feature_settings():
    with pytest.raises(ValueError):
        validate_training_config({"training": {"use_supervised_parameter_loss": True}})
    with pytest.raises(ValueError):
        validate_training_config({"features": {"use_gt_maps_as_input": True}})


def test_one_training_epoch_saves_self_supervised_checkpoint(tmp_path):
    case_dir = tmp_path / "sim" / "mida_parameters" / "case_000"
    _write_parameter_case(case_dir, shape=(12, 12, 3))
    cfg = {
        "features": {"use_gt_maps_as_input": False},
        "dataset": {
            "sim_root": str(tmp_path / "sim"),
            "samples_per_epoch": 1,
            "n_timepoints": 16,
            "tr_seconds": 1.55,
            "temporal_mode": "windowed",
            "temporal_window_length": 8,
            "paradigms": ["block"],
            "tcnr_levels": [5.0],
            "etco2_input": "clean",
        },
        "model": {"base_channels": 2, "depth": 1, "dropout": 0.0, "T_mode": "global"},
        "training": {
            "batch_size": 1,
            "num_workers": 0,
            "use_amp": False,
            "gradient_clip_norm": 1.0,
        },
        "losses": {
            "lambda_data": 1.0,
            "lambda_view": 0.0,
        },
    }
    result = train_physiology_model(
        cfg,
        tmp_path / "sim",
        tmp_path / "model",
        max_epochs=1,
        samples_per_epoch=1,
    )
    assert result["best_checkpoint"].exists()
    assert result["history"].exists()
    checkpoint = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["input_channel_names"]
    assert checkpoint["T_mode"] == "global"
    assert "val_total" in checkpoint["metrics"]


def test_staged_training_switches_T_mode_and_writes_history(tmp_path):
    case_dir = tmp_path / "sim" / "mida_parameters" / "case_000"
    _write_parameter_case(case_dir, shape=(12, 12, 3))
    cfg = {
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
        },
        "model": {"base_channels": 2, "depth": 1, "dropout": 0.0, "T_mode": "tissuewise"},
        "training": {
            "batch_size": 1,
            "num_workers": 0,
            "use_amp": False,
            "gradient_clip_norm": 1.0,
        },
        "losses": {
            "lambda_data": 1.0,
            "lambda_view": 0.0,
        },
        "stages": {
            "stage_1_clean": {
                "epochs": 1,
                "T_mode": "global",
                "etco2_input": "clean",
                "paradigms": ["block"],
                "tcnr_levels": [5.0],
            },
            "stage_2_physics": {
                "epochs": 1,
                "T_mode": "tissuewise",
                "etco2_input": "measured",
                "paradigms": ["ramp"],
                "tcnr_levels": [2.0],
            },
        },
    }
    result = train_physiology_model(cfg, tmp_path / "sim", tmp_path / "model", samples_per_epoch=1)
    history = result["history"].read_text(encoding="utf-8")
    assert "stage_1_clean" in history
    assert "stage_2_physics" in history
    checkpoint = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["training_stage"] in {"stage_1_clean", "stage_2_physics"}
