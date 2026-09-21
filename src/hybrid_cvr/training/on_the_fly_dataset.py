from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Any

from hybrid_cvr.simulation.co2_paradigms import make_paradigm
from hybrid_cvr.simulation.mida_bold import (
    FEATURE_NAMES,
    MidaBoldSimulationConfig,
    TISSUE_FRACTION_NAMES,
    add_etco2_measurement_error,
    add_tcnr_noise,
    add_tcnr_noise_volume,
    baseline_intensity_map,
    default_brain_slices,
    estimate_tcnr_map,
    interpolate_uniform_1d_at,
    load_mida_parameter_case,
    local_spatial_uncertainty,
    simulate_bold_psc_volume_from_maps,
    tissue_boundary_uncertainty,
)
from hybrid_cvr.simulation.simulate_bold import simulate_bold_psc


TEMPORAL_RESPONSE_FEATURE_NAMES = (
    "psc_co2_beta_0lag",
    "psc_co2_corr_0lag",
    "psc_co2_beta_peak",
    "psc_co2_corr_peak",
    "psc_co2_lag_peak_norm",
)

ON_THE_FLY_FEATURE_NAMES = (
    "baseline_bold",
    "psc_mean",
    "psc_std",
    "psc_delta",
    "tcnr",
    "mask",
    "psc_co2_beta_0lag",
    "psc_co2_corr_0lag",
    "psc_co2_corr_peak",
    "psc_co2_lag_peak_norm",
)

TEMPORAL_RESPONSE_LAG_SECONDS = (0.0, 10.0, 20.0, 35.0, 50.0, 65.0, 80.0)


@dataclass(frozen=True)
class OnTheFlyDatasetConfig:
    sim_root: Path = Path("data/simulated")
    case_index_path: Path | None = None
    split_json_path: Path | None = None
    split: str = "train"
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    samples_per_epoch: int = 1024
    seed: int = 17
    split_seed_offsets: tuple[int, int, int] = (0, 100_000, 200_000)
    slice_mode: str = "2d"
    slice_axis: str = "axial"
    slice_index_range: tuple[int, int] | list[int] | None = None
    slice_indices: tuple[int, ...] = field(default_factory=tuple)
    patch_size: tuple[int, int, int] | list[int] = (32, 32, 24)
    tissue_balanced_patches: bool = False
    model_mismatch_probability: float = 0.0
    model_mismatch_strength_range: tuple[float, float] = (0.05, 0.20)
    paradigms: tuple[str, ...] = (
        "block",
        "multi_step",
        "pseudo_random_binary",
    )
    tcnr_levels: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
    n_timepoints: int = 480
    tr_seconds: float = 1.55
    etco2_input: str = "measured"
    temporal_mode: str = "windowed"
    temporal_window_length: int = 160
    temporal_windows_per_sample: int = 1
    prefer_transition_windows: bool = True
    transition_alpha: float = 2.0
    simulation_backend: str = "cpu"
    sampling_strategy: str = "balanced_grid"
    tcnr_probabilities: tuple[float, ...] | list[float] | None = None
    randomize_artifacts_for_training: bool = True
    etco2_noise_sd_mmhg_range: tuple[float, float] = (0.0, 0.40)
    etco2_drift_sd_mmhg_range: tuple[float, float] = (0.0, 0.25)
    drift_fraction_of_noise_range: tuple[float, float] = (0.0, 0.25)
    motion_spike_probability_options: tuple[float, ...] = (0.0, 0.005, 0.015, 0.03)
    motion_spike_scale_range: tuple[float, float] = (1.5, 4.0)
    ar1_rho_range: tuple[float, float] = (0.15, 0.55)
    return_consistency_pair: bool = False
    consistency_seed_offset: int = 50_000_000
    consistency_mode: str = "same_paradigm"


class OnTheFlyCVRDataset(__import__("torch").utils.data.Dataset):
    """Generate hidden-parameter BOLD views without exposing parameter labels."""

    feature_names = ON_THE_FLY_FEATURE_NAMES

    def __init__(self, config: OnTheFlyDatasetConfig | dict[str, Any] | None = None, **kwargs: Any):
        torch = __import__("torch")
        self.torch = torch
        self.np = __import__("numpy")
        if config is None:
            config = OnTheFlyDatasetConfig(**kwargs)
        elif isinstance(config, dict):
            config = OnTheFlyDatasetConfig(**{**config, **kwargs})
        elif kwargs:
            config = OnTheFlyDatasetConfig(**{**config.__dict__, **kwargs})
        self.config = config
        if config.etco2_input not in {"clean", "measured", "mixed"}:
            raise ValueError("etco2_input must be clean, measured, or mixed")
        if config.temporal_mode not in {"full", "windowed", "transition_weighted", "mixed"}:
            raise ValueError("temporal_mode must be full, windowed, transition_weighted, or mixed")
        if config.simulation_backend not in {"cpu", "torch_gpu"}:
            raise ValueError("simulation_backend must be cpu or torch_gpu")
        if config.sampling_strategy not in {"balanced_grid", "random"}:
            raise ValueError("sampling_strategy must be balanced_grid or random")
        if config.tcnr_probabilities is not None:
            probabilities = self.np.asarray(config.tcnr_probabilities, dtype=float)
            if probabilities.size != len(config.tcnr_levels):
                raise ValueError("tcnr_probabilities must match tcnr_levels")
            if self.np.any(probabilities < 0) or float(probabilities.sum()) <= 0:
                raise ValueError("tcnr_probabilities must be nonnegative with positive sum")
        self.case_dirs = self._split_case_dirs(self._discover_case_dirs())
        if not self.case_dirs:
            raise FileNotFoundError(f"No {config.split} parameter cases found under {config.sim_root}")
        self.base_seed = int(config.seed) + self._split_seed_offset(config.split)
        self.epoch = 0
        self._case_cache: dict[Path, tuple[dict[str, Any], Any]] = {}
        self.condition_grid = tuple(
            (str(paradigm), float(tcnr))
            for paradigm in config.paradigms
            for tcnr in config.tcnr_levels
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def __len__(self) -> int:
        if self.config.sampling_strategy == "balanced_grid":
            return max(int(self.config.samples_per_epoch), len(self.condition_grid))
        return int(self.config.samples_per_epoch)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seed = int(self.base_seed) + int(idx)
        if self.config.split.lower() == "train":
            seed += 1_000_003 * int(self.epoch)
        rng = self.np.random.default_rng(seed)
        case_dir = self.case_dirs[int(rng.integers(0, len(self.case_dirs)))]
        maps, ref_img = self._load_case(case_dir)
        fractions = self.np.stack(
            [maps[f"fraction_{name}"] for name in TISSUE_FRACTION_NAMES], axis=0
        ).astype("float32")
        support = self.np.asarray(maps["region_labels"] > 0, dtype=bool)
        spatial_mode = str(self.config.slice_mode).lower()
        full_volume = spatial_mode in {"3d", "volume", "full", "full_volume", "3d_patch", "patch"}
        if full_volume:
            slice_axis = -1
            z = -1
            slice_maps = self._volume_maps(maps, fractions)
            if spatial_mode in {"3d_patch", "patch"}:
                slice_maps = self._crop_volume_patch(slice_maps, support, rng)
        else:
            slice_axis = self._slice_axis(ref_img, support.ndim)
            z = self._choose_slice(support, rng, slice_axis)
            slice_maps = self._slice_maps(maps, fractions, z, slice_axis)
        mask = slice_maps["region_labels"] > 0
        if not self.np.any(mask):
            raise ValueError(f"Chosen slice {z} in {case_dir} has no brain support")

        time_grid = self.np.arange(self.config.n_timepoints, dtype="float32") * float(
            self.config.tr_seconds
        )
        paradigm, tcnr = self._choose_condition(idx, rng)
        paradigm_seed = int(rng.integers(0, 2**31 - 1))
        noise_seed = int(rng.integers(0, 2**31 - 1))
        etco2_clean = make_paradigm(paradigm, time_grid, seed=paradigm_seed).astype("float32")
        window = self._choose_temporal_indices(etco2_clean, rng)
        boundary_uncertainty = tissue_boundary_uncertainty(slice_maps["fractions"], mask)
        if self.config.simulation_backend == "torch_gpu":
            return self._make_gpu_simulation_batch(
                case_dir=case_dir,
                slice_maps=slice_maps,
                mask=mask,
                time_grid=time_grid,
                etco2_clean=etco2_clean,
                paradigm=paradigm,
                tcnr=tcnr,
                noise_seed=noise_seed,
                paradigm_seed=paradigm_seed,
                rng=rng,
                boundary_uncertainty=boundary_uncertainty,
                window=window,
                slice_axis=slice_axis,
                z=z,
                full_volume=full_volume,
            )
        clean = self._simulate_clean_bold(slice_maps, mask, time_grid, etco2_clean, full_volume, window)
        view = self._make_observation_view(
            slice_maps,
            mask,
            time_grid,
            etco2_clean,
            clean,
            paradigm,
            tcnr,
            noise_seed,
            rng,
            boundary_uncertainty,
            window,
        )
        out = {
            "features": self._tensor(view["features"]),
            "bold_psc": self._tensor(view["bold_psc"]),
            "etco2_clean": self._tensor(view["etco2_clean"]),
            "etco2_measured": self._tensor(view["etco2_measured"]),
            "etco2_for_model": self._tensor(view["etco2_for_model"]),
            "etco2": self._tensor(view["etco2_for_model"]),
            "time_grid": self._tensor(view["time_grid"]),
            "mask": self._tensor(mask.astype("float32")),
            "local_uncertainty": self._tensor(view["local_uncertainty"]),
            "local_uncertainty_weights": self._tensor(view["local_uncertainty_weights"]),
            "transition_weight": self._tensor(view["transition_weight"]),
            "vessel_likelihood": self._tensor(slice_maps["vessel_likelihood"]),
            "boundary_uncertainty": self._tensor(boundary_uncertainty),
            "tissue_maps": self._tensor(slice_maps["fractions"]),
            "metadata": {
                "case_id": case_dir.name,
                "case_dir": str(case_dir),
                "slice_index": int(z),
                "slice_axis": int(slice_axis),
                "slice_axis_name": str(self.config.slice_axis),
                "spatial_mode": "3d_full_volume" if full_volume else "2d_slice",
                "paradigm": paradigm,
                "target_tcnr": float(tcnr),
                "paradigm_seed": int(paradigm_seed),
                **view["metadata"],
            },
        }
        if self.config.return_consistency_pair:
            pair_seed = seed + int(self.config.consistency_seed_offset)
            pair_rng = self.np.random.default_rng(pair_seed)
            pair_noise_seed = int(pair_rng.integers(0, 2**31 - 1))
            pair = self._make_observation_view(
                slice_maps,
                mask,
                time_grid,
                etco2_clean,
                clean,
                paradigm,
                tcnr,
                pair_noise_seed,
                pair_rng,
                boundary_uncertainty,
                window,
            )
            out.update(
                {
                    "consistency_features": self._tensor(pair["features"]),
                    "consistency_bold_psc": self._tensor(pair["bold_psc"]),
                    "consistency_etco2_for_model": self._tensor(pair["etco2_for_model"]),
                    "consistency_local_uncertainty_weights": self._tensor(
                        pair["local_uncertainty_weights"]
                    ),
                    "consistency_transition_weight": self._tensor(pair["transition_weight"]),
                    "consistency_metadata": pair["metadata"],
                }
            )
        return out

    def _make_gpu_simulation_batch(
        self,
        *,
        case_dir: Path,
        slice_maps: dict[str, Any],
        mask: Any,
        time_grid: Any,
        etco2_clean: Any,
        paradigm: str,
        tcnr: float,
        noise_seed: int,
        paradigm_seed: int,
        rng: Any,
        boundary_uncertainty: Any,
        window: Any,
        slice_axis: int,
        z: int,
        full_volume: bool,
    ) -> dict[str, Any]:
        np = self.np
        artifact_settings = self._sample_artifact_settings(rng)
        sim_config = MidaBoldSimulationConfig(
            n_timepoints=self.config.n_timepoints,
            tr_seconds=self.config.tr_seconds,
            tcnr_levels=(tcnr,),
            paradigms=(paradigm,),
            seed=int(self.config.seed),
            **artifact_settings,
        )
        etco2_measured = add_etco2_measurement_error(
            etco2_clean, time_grid, sim_config, noise_seed
        ).astype("float32")
        etco2_for_model = self._choose_model_etco2(etco2_clean, etco2_measured, rng)
        window = np.asarray(window, dtype=int)
        time_grid_window = time_grid[window] - time_grid[window][0]
        component_stacks = {name: [] for name in ("CVR", "delay", "T")}
        for tissue in TISSUE_FRACTION_NAMES:
            for name in component_stacks:
                component_stacks[name].append(
                    np.asarray(slice_maps[f"component_{name}_{tissue}"], dtype="float32")
                )
        etco2_for_transition = (
            etco2_clean[window] if self.config.etco2_input == "clean" else etco2_for_model[window]
        )
        mismatch_strength = 0.0
        if (
            self.config.split.lower() == "train"
            and float(rng.random()) < float(self.config.model_mismatch_probability)
        ):
            mismatch_strength = self._uniform_range(
                self.config.model_mismatch_strength_range, rng
            )
        out = {
            "simulate_on_gpu": self._tensor(np.asarray(1.0, dtype="float32")),
            "features": self._tensor(np.zeros((len(self.feature_names),) + mask.shape, dtype="float32")),
            "sim_cvr_components": self._tensor(np.stack(component_stacks["CVR"], axis=0)),
            "sim_delay_components": self._tensor(np.stack(component_stacks["delay"], axis=0)),
            "sim_T_components": self._tensor(np.stack(component_stacks["T"], axis=0)),
            "etco2_clean": self._tensor(etco2_clean[window]),
            "etco2_measured": self._tensor(etco2_measured[window]),
            "etco2_for_model": self._tensor(etco2_for_model[window]),
            "etco2": self._tensor(etco2_for_model[window]),
            "time_grid": self._tensor(time_grid_window),
            "valid_time_mask": self._tensor(np.ones(len(window), dtype="float32")),
            "mask": self._tensor(mask.astype("float32")),
            "transition_weight": self._tensor(
                self.compute_transition_weight(etco2_for_transition, alpha=self.config.transition_alpha)
            ),
            "vessel_likelihood": self._tensor(slice_maps["vessel_likelihood"]),
            "boundary_uncertainty": self._tensor(boundary_uncertainty),
            "tissue_maps": self._tensor(slice_maps["fractions"]),
            "target_tcnr": self._tensor(np.asarray(float(tcnr), dtype="float32")),
            "noise_seed": self._tensor(np.asarray(float(noise_seed), dtype="float32")),
            "drift_fraction_of_noise": self._tensor(
                np.asarray(float(artifact_settings["drift_fraction_of_noise"]), dtype="float32")
            ),
            "motion_spike_probability": self._tensor(
                np.asarray(float(artifact_settings["motion_spike_probability"]), dtype="float32")
            ),
            "motion_spike_scale": self._tensor(
                np.asarray(float(artifact_settings["motion_spike_scale"]), dtype="float32")
            ),
            "ar1_rho": self._tensor(np.asarray(float(artifact_settings["ar1_rho"]), dtype="float32")),
            "model_mismatch_strength": self._tensor(
                np.asarray(float(mismatch_strength), dtype="float32")
            ),
            "metadata": {
                "case_id": case_dir.name,
                "case_dir": str(case_dir),
                "slice_index": int(z),
                "slice_axis": int(slice_axis),
                "slice_axis_name": str(self.config.slice_axis),
                "spatial_mode": "3d_full_volume" if full_volume else "2d_slice",
                "paradigm": paradigm,
                "target_tcnr": float(tcnr),
                "paradigm_seed": int(paradigm_seed),
                "noise_seed": int(noise_seed),
                **artifact_settings,
            },
        }
        if self.config.return_consistency_pair:
            pair_rng = np.random.default_rng(noise_seed + int(self.config.consistency_seed_offset))
            alternatives = [name for name in self.config.paradigms if str(name) != str(paradigm)]
            cross = self.config.consistency_mode == "cross_paradigm" and alternatives
            pair_paradigm = str(pair_rng.choice(alternatives)) if cross else str(paradigm)
            pair_paradigm_seed = int(pair_rng.integers(0, 2**31 - 1)) if cross else paradigm_seed
            pair_noise_seed = int(pair_rng.integers(0, 2**31 - 1))
            pair_clean = (make_paradigm(pair_paradigm, time_grid, seed=pair_paradigm_seed).astype("float32")
                          if cross else etco2_clean.copy())
            pair_artifacts = self._sample_artifact_settings(pair_rng)
            pair_config = MidaBoldSimulationConfig(
                n_timepoints=self.config.n_timepoints,
                tr_seconds=self.config.tr_seconds,
                tcnr_levels=(tcnr,),
                paradigms=(pair_paradigm,),
                seed=int(self.config.seed),
                **pair_artifacts,
            )
            pair_measured = add_etco2_measurement_error(
                pair_clean, time_grid, pair_config, pair_noise_seed
            ).astype("float32")
            pair_for_model = self._choose_model_etco2(pair_clean, pair_measured, pair_rng)
            pair_transition_source = (
                pair_clean[window]
                if self.config.etco2_input == "clean"
                else pair_for_model[window]
            )
            out.update(
                {
                    "consistency_etco2_clean": self._tensor(pair_clean[window]),
                    "consistency_etco2_for_model": self._tensor(pair_for_model[window]),
                    "consistency_transition_weight": self._tensor(
                        self.compute_transition_weight(
                            pair_transition_source, alpha=self.config.transition_alpha
                        )
                    ),
                    "consistency_noise_seed": self._tensor(
                        np.asarray(float(pair_noise_seed), dtype="float32")
                    ),
                    "consistency_drift_fraction_of_noise": self._tensor(
                        np.asarray(
                            float(pair_artifacts["drift_fraction_of_noise"]), dtype="float32"
                        )
                    ),
                    "consistency_motion_spike_probability": self._tensor(
                        np.asarray(
                            float(pair_artifacts["motion_spike_probability"]), dtype="float32"
                        )
                    ),
                    "consistency_motion_spike_scale": self._tensor(
                        np.asarray(float(pair_artifacts["motion_spike_scale"]), dtype="float32")
                    ),
                    "consistency_ar1_rho": self._tensor(
                        np.asarray(float(pair_artifacts["ar1_rho"]), dtype="float32")
                    ),
                }
            )
        return out

    def _make_observation_view(
        self,
        slice_maps: dict[str, Any],
        mask: Any,
        time_grid: Any,
        etco2_clean: Any,
        clean: Any,
        paradigm: str,
        tcnr: float,
        noise_seed: int,
        rng: Any,
        boundary_uncertainty: Any,
        window: Any,
    ) -> dict[str, Any]:
        artifact_settings = self._sample_artifact_settings(rng)
        sim_config = MidaBoldSimulationConfig(
            n_timepoints=self.config.n_timepoints,
            tr_seconds=self.config.tr_seconds,
            tcnr_levels=(tcnr,),
            paradigms=(paradigm,),
            seed=int(self.config.seed),
            **artifact_settings,
        )
        etco2_measured = add_etco2_measurement_error(
            etco2_clean, time_grid, sim_config, noise_seed
        ).astype("float32")
        etco2_clean_window = etco2_clean[window]
        etco2_measured_window = etco2_measured[window]
        time_grid_window = time_grid[window] - time_grid[window][0]
        if clean.ndim == 4:
            noisy_last, noise_summary = add_tcnr_noise_volume(
                self.np.moveaxis(clean, 0, -1), mask, tcnr, sim_config, noise_seed
            )
            noisy = self.np.moveaxis(noisy_last, -1, 0).astype("float32")
            local_uncertainty = self._local_spatial_uncertainty_volume(noisy, mask).astype("float32")
            tcnr_map = self._estimate_tcnr_map_volume(clean, noisy, mask)
        else:
            noisy, noise_summary = add_tcnr_noise(clean, mask, tcnr, sim_config, noise_seed)
            local_uncertainty = local_spatial_uncertainty(noisy, mask).astype("float32")
            tcnr_map = estimate_tcnr_map(clean, noisy, mask)
        baseline = baseline_intensity_map(slice_maps["fractions"], mask, sim_config, rng)
        local_weights = self._local_uncertainty_weights(local_uncertainty, mask)
        etco2_for_transition = etco2_measured_window if self.config.etco2_input != "clean" else etco2_clean_window
        transition_weight = self.compute_transition_weight(
            etco2_for_transition,
            alpha=self.config.transition_alpha,
        )
        etco2_for_model = self._choose_model_etco2(etco2_clean, etco2_measured, rng)
        features = self._build_features(
            noisy,
            clean,
            baseline,
            tcnr_map,
            mask,
            slice_maps["fractions"],
            slice_maps["vessel_likelihood"],
            boundary_uncertainty,
            etco2_for_model[window],
        )
        return {
            "features": features,
            "bold_psc": noisy,
            "etco2_clean": etco2_clean_window,
            "etco2_measured": etco2_measured_window,
            "etco2_for_model": etco2_for_model[window],
            "time_grid": time_grid_window,
            "local_uncertainty": local_uncertainty,
            "local_uncertainty_weights": local_weights,
            "transition_weight": transition_weight,
            "metadata": {
                "noise_seed": int(noise_seed),
                **artifact_settings,
                **noise_summary,
            },
        }

    def _choose_condition(self, idx: int, rng: Any) -> tuple[str, float]:
        if self.config.sampling_strategy == "balanced_grid":
            return self.condition_grid[int(idx) % len(self.condition_grid)]
        probabilities = self.config.tcnr_probabilities
        if probabilities is not None:
            probabilities = self.np.asarray(probabilities, dtype=float)
            probabilities = probabilities / probabilities.sum()
        return str(rng.choice(self.config.paradigms)), float(
            rng.choice(self.config.tcnr_levels, p=probabilities)
        )

    def _sample_artifact_settings(self, rng: Any) -> dict[str, float]:
        settings = {
            "etco2_noise_sd_mmhg": 0.25,
            "etco2_drift_sd_mmhg": 0.15,
            "drift_fraction_of_noise": 0.15,
            "motion_spike_probability": 0.015,
            "motion_spike_scale": 3.0,
            "ar1_rho": 0.35,
        }
        if self.config.split.lower() == "train" and self.config.randomize_artifacts_for_training:
            settings = {
                "etco2_noise_sd_mmhg": self._uniform_range(
                    self.config.etco2_noise_sd_mmhg_range, rng
                ),
                "etco2_drift_sd_mmhg": self._uniform_range(
                    self.config.etco2_drift_sd_mmhg_range, rng
                ),
                "drift_fraction_of_noise": self._uniform_range(
                    self.config.drift_fraction_of_noise_range, rng
                ),
                "motion_spike_probability": float(
                    rng.choice(tuple(self.config.motion_spike_probability_options))
                ),
                "motion_spike_scale": self._uniform_range(
                    self.config.motion_spike_scale_range, rng
                ),
                "ar1_rho": self._uniform_range(self.config.ar1_rho_range, rng),
            }
        return settings

    @staticmethod
    def _uniform_range(bounds: tuple[float, float] | list[float], rng: Any) -> float:
        lo, hi = float(bounds[0]), float(bounds[1])
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            return lo
        return float(rng.uniform(lo, hi))

    def _load_case(self, case_dir: Path) -> tuple[dict[str, Any], Any]:
        if case_dir not in self._case_cache:
            self._case_cache[case_dir] = load_mida_parameter_case(case_dir)
        return self._case_cache[case_dir]

    def _discover_case_dirs(self) -> list[Path]:
        if self.config.case_index_path is not None:
            return self._case_dirs_from_index(Path(self.config.case_index_path))
        root = Path(self.config.sim_root)
        candidates = [root / "mida_parameters", root]
        for base in candidates:
            if base.exists():
                case_dirs = sorted(p for p in base.glob("case_*") if p.is_dir())
                if case_dirs:
                    return case_dirs
        return []

    def _split_case_dirs(self, case_dirs: list[Path]) -> list[Path]:
        if self.config.split_json_path is not None:
            selected = self._case_ids_from_split_json(Path(self.config.split_json_path))
            by_id = {p.name: p for p in case_dirs}
            return [by_id[case_id] for case_id in selected if case_id in by_id]
        if len(case_dirs) == 1:
            return case_dirs
        n = len(case_dirs)
        n_train = max(1, int(round(n * self.config.split_fractions[0])))
        n_val = max(1, int(round(n * self.config.split_fractions[1]))) if n >= 3 else 0
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))
        split = self.config.split.lower()
        if split == "train":
            return case_dirs[:n_train]
        if split in {"val", "validation"}:
            return case_dirs[n_train : n_train + n_val]
        if split == "test":
            return case_dirs[n_train + n_val :]
        raise ValueError("split must be train, val, validation, or test")

    def _case_dirs_from_index(self, case_index_path: Path) -> list[Path]:
        if not case_index_path.exists():
            raise FileNotFoundError(f"Missing case index: {case_index_path}")
        split = self.config.split.lower()
        split_aliases = {split}
        if split == "val":
            split_aliases.add("validation")
        if split == "validation":
            split_aliases.add("val")
        rows = []
        with case_index_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split", "").lower() in split_aliases:
                    rows.append(Path(row["case_dir"]))
        return rows

    def _case_ids_from_split_json(self, split_json_path: Path) -> list[str]:
        if not split_json_path.exists():
            raise FileNotFoundError(f"Missing split JSON: {split_json_path}")
        split = self.config.split.lower()
        key = {"train": "train_cases", "val": "val_cases", "validation": "val_cases", "test": "test_cases"}.get(split)
        if key is None:
            raise ValueError("split must be train, val, validation, or test")
        payload = json.loads(split_json_path.read_text(encoding="utf-8"))
        return [str(v) for v in payload.get(key, [])]

    def _split_seed_offset(self, split: str) -> int:
        offsets = self.config.split_seed_offsets
        split = split.lower()
        if split == "train":
            return int(offsets[0])
        if split in {"val", "validation"}:
            return int(offsets[1])
        if split == "test":
            return int(offsets[2])
        return 0

    def _slice_axis(self, ref_img: Any, ndim: int) -> int:
        axis_name = str(self.config.slice_axis).lower()
        if axis_name in {"0", "1", "2"}:
            return int(axis_name)
        if axis_name in {"array_z", "z"}:
            return 2
        anatomical = {
            "axial": {"s", "i"},
            "coronal": {"a", "p"},
            "sagittal": {"l", "r"},
        }
        if axis_name not in anatomical:
            raise ValueError("slice_axis must be axial, coronal, sagittal, array_z, or 0/1/2")
        nib = __import__("nibabel")
        axcodes = tuple(str(code).lower() for code in nib.aff2axcodes(ref_img.affine))
        for axis, code in enumerate(axcodes[:ndim]):
            if code in anatomical[axis_name]:
                return axis
        raise ValueError(f"Could not find anatomical {axis_name} axis from axcodes={axcodes}")

    def _choose_slice(self, support: Any, rng: Any, axis: int) -> int:
        if self.config.slice_indices:
            candidates = [int(v) for v in self.config.slice_indices]
        elif self.config.slice_index_range is not None:
            lo, hi = int(self.config.slice_index_range[0]), int(self.config.slice_index_range[1])
            if hi < lo:
                lo, hi = hi, lo
            candidates = list(range(lo, hi + 1))
        else:
            candidates = list(
                self._default_brain_slices_along_axis(
                    support, axis, n_slices=min(9, max(1, support.shape[axis]))
                )
            )
        valid_indices = self._valid_slice_indices(support, axis)
        candidates = [idx for idx in candidates if idx in valid_indices]
        if not candidates:
            raise ValueError(
                f"No valid brain slices for axis={axis} range={self.config.slice_index_range} "
                f"indices={self.config.slice_indices}"
            )
        return int(rng.choice(candidates))

    def _valid_slice_indices(self, support: Any, axis: int) -> set[int]:
        np = self.np
        counts = np.asarray(support, dtype=bool).sum(
            axis=tuple(i for i in range(support.ndim) if i != axis)
        )
        return {int(v) for v in np.flatnonzero(counts > 0)}

    def _default_brain_slices_along_axis(
        self, support: Any, axis: int, n_slices: int = 1
    ) -> tuple[int, ...]:
        if axis == 2:
            return default_brain_slices(support, n_slices=n_slices)
        np = self.np
        valid = np.asarray(sorted(self._valid_slice_indices(support, axis)), dtype=int)
        if valid.size == 0:
            raise ValueError("Cannot choose simulation slices from an empty brain mask")
        if n_slices <= 1:
            return (int(valid[len(valid) // 2]),)
        positions = np.linspace(0, valid.size - 1, n_slices)
        return tuple(int(valid[int(round(pos))]) for pos in positions)

    def _slice_maps(self, maps: dict[str, Any], fractions: Any, z: int, axis: int) -> dict[str, Any]:
        np = self.np
        out = {
            "GT_CVR": np.take(maps["CVR"], z, axis=axis).astype("float32"),
            "GT_delay": np.take(maps["delay"], z, axis=axis).astype("float32"),
            "GT_T": np.take(maps["T"], z, axis=axis).astype("float32"),
            "region_labels": np.take(maps["region_labels"], z, axis=axis),
            "vessel_likelihood": np.take(maps["vessel_likelihood"], z, axis=axis).astype(
                "float32"
            ),
            "fractions": np.take(fractions, z, axis=axis + 1).astype("float32"),
        }
        for tissue in TISSUE_FRACTION_NAMES:
            for param in ("CVR", "delay", "T"):
                key = f"component_{param}_{tissue}"
                if key in maps:
                    out[key] = np.take(maps[key], z, axis=axis).astype("float32")
        return out

    def _volume_maps(self, maps: dict[str, Any], fractions: Any) -> dict[str, Any]:
        np = self.np
        out = {
            "GT_CVR": np.asarray(maps["CVR"], dtype="float32"),
            "GT_delay": np.asarray(maps["delay"], dtype="float32"),
            "GT_T": np.asarray(maps["T"], dtype="float32"),
            "region_labels": np.asarray(maps["region_labels"]),
            "vessel_likelihood": np.asarray(maps["vessel_likelihood"], dtype="float32"),
            "fractions": np.asarray(fractions, dtype="float32"),
        }
        for tissue in TISSUE_FRACTION_NAMES:
            out[f"fraction_{tissue}"] = np.asarray(maps[f"fraction_{tissue}"], dtype="float32")
            for param in ("CVR", "delay", "T"):
                key = f"component_{param}_{tissue}"
                if key in maps:
                    out[key] = np.asarray(maps[key], dtype="float32")
        return out

    def _crop_volume_patch(self, maps: dict[str, Any], support: Any, rng: Any) -> dict[str, Any]:
        np = self.np
        patch = tuple(int(v) for v in self.config.patch_size)
        if len(patch) != 3 or any(v <= 0 for v in patch):
            raise ValueError("patch_size must contain three positive integers")
        support = np.asarray(support, dtype=bool)
        coords = np.empty((0, 3), dtype=int)
        if self.config.tissue_balanced_patches:
            fractions = np.asarray(maps["fractions"], dtype="float32")
            candidate_tissues = []
            for tissue_idx in range(min(5, fractions.shape[0])):
                threshold = 0.01 if tissue_idx == 4 else 0.25
                tissue_coords = np.argwhere(support & (fractions[tissue_idx] > threshold))
                if len(tissue_coords):
                    candidate_tissues.append(tissue_coords)
            if candidate_tissues:
                coords = candidate_tissues[int(rng.integers(0, len(candidate_tissues)))]
        if coords.size == 0:
            coords = np.argwhere(support)
        if coords.size == 0:
            raise ValueError("Cannot crop a 3D patch from empty brain support")
        center = coords[int(rng.integers(0, len(coords)))]
        starts = tuple(
            max(0, min(int(center[axis]) - patch[axis] // 2, support.shape[axis] - patch[axis]))
            for axis in range(3)
        )
        slices = tuple(slice(start, start + size) for start, size in zip(starts, patch, strict=True))
        out: dict[str, Any] = {}
        for key, value in maps.items():
            arr = np.asarray(value)
            if key == "fractions":
                out[key] = arr[(slice(None),) + slices].copy()
            elif arr.ndim == 3:
                out[key] = arr[slices].copy()
            else:
                out[key] = arr.copy()
        out["patch_origin"] = np.asarray(starts, dtype=np.int32)
        return out

    def _simulate_clean_bold(
        self,
        maps: dict[str, Any],
        mask: Any,
        time_grid: Any,
        etco2_clean: Any,
        full_volume: bool,
        window: Any | None = None,
    ) -> Any:
        window = self.np.arange(len(time_grid), dtype=int) if window is None else self.np.asarray(window, dtype=int)
        if full_volume:
            sim_maps = {
                "CVR": maps["GT_CVR"],
                "delay": maps["GT_delay"],
                "T": maps["GT_T"],
                "region_labels": maps["region_labels"],
                "vessel_likelihood": maps["vessel_likelihood"],
            }
            for tissue in TISSUE_FRACTION_NAMES:
                sim_maps[f"fraction_{tissue}"] = maps[f"fraction_{tissue}"]
                for param in ("CVR", "delay", "T"):
                    key = f"component_{param}_{tissue}"
                    if key in maps:
                        sim_maps[key] = maps[key]
            return self._simulate_signal_pve_volume_window(sim_maps, mask, time_grid, etco2_clean, window)
        if all(f"component_{param}_{tissue}" in maps for tissue in TISSUE_FRACTION_NAMES for param in ("CVR", "delay", "T")):
            out = self.np.zeros((len(window),) + mask.shape, dtype="float32")
            frac_sum = self.np.zeros(mask.shape, dtype="float32")
            for tissue_idx, tissue in enumerate(TISSUE_FRACTION_NAMES):
                frac = maps["fractions"][tissue_idx]
                tissue_mask = mask & (frac > 1e-6)
                if not self.np.any(tissue_mask):
                    continue
                response = simulate_bold_psc(
                    maps[f"component_CVR_{tissue}"],
                    maps[f"component_delay_{tissue}"],
                    maps[f"component_T_{tissue}"],
                    time_grid,
                    etco2_clean,
                )["bold_psc"][window].astype("float32")
                response[:, ~tissue_mask] = 0.0
                out += response * frac[None, ...]
                frac_sum += self.np.where(tissue_mask, frac, 0.0).astype("float32")
            out = self.np.divide(out, frac_sum[None, ...], out=self.np.zeros_like(out), where=frac_sum[None, ...] > 1e-6)
            out[:, ~mask] = 0.0
            return out.astype("float32")
        return simulate_bold_psc(
            maps["GT_CVR"],
            maps["GT_delay"],
            maps["GT_T"],
            time_grid,
            etco2_clean,
        )["bold_psc"][window].astype("float32")

    def _simulate_signal_pve_volume_window(
        self,
        maps: dict[str, Any],
        mask: Any,
        time_grid: Any,
        etco2_clean: Any,
        window: Any,
        chunk_voxels: int = 500000,
    ) -> Any:
        np = self.np
        m = np.asarray(mask, dtype=bool)
        time = np.asarray(time_grid, dtype="float32")
        u = np.asarray(etco2_clean, dtype="float32")
        window = np.asarray(window, dtype=np.int32)
        if window.size == 0:
            raise ValueError("Temporal window is empty")
        out = np.zeros((window.size,) + m.shape, dtype="float32")
        fraction_sum = np.zeros(m.shape, dtype="float32")
        window_lookup = {int(t): i for i, t in enumerate(window.tolist())}
        max_t = int(window.max())
        for tissue in TISSUE_FRACTION_NAMES:
            frac = np.asarray(maps[f"fraction_{tissue}"], dtype="float32")
            tissue_mask = m & (frac > 1e-6)
            if not np.any(tissue_mask):
                continue
            cvr = np.asarray(maps[f"component_CVR_{tissue}"], dtype="float32")
            delay = np.asarray(maps[f"component_delay_{tissue}"], dtype="float32")
            tau = np.maximum(np.asarray(maps[f"component_T_{tissue}"], dtype="float32"), 1e-6)
            flat_valid = np.flatnonzero(tissue_mask.reshape(-1))
            flat_cvr = cvr.reshape(-1)
            flat_delay = delay.reshape(-1)
            flat_tau = tau.reshape(-1)
            flat_frac = frac.reshape(-1)
            flat_out = out.reshape((window.size, -1))
            for start in range(0, flat_valid.size, int(chunk_voxels)):
                idx = flat_valid[start : start + int(chunk_voxels)]
                y_prev = np.zeros(idx.size, dtype="float32")
                cvr_v = flat_cvr[idx]
                delay_v = flat_delay[idx]
                tau_v = flat_tau[idx]
                frac_v = flat_frac[idx]
                if 0 in window_lookup:
                    flat_out[window_lookup[0], idx] += y_prev * frac_v
                for t in range(max_t):
                    dt = np.float32(time[t + 1] - time[t])
                    shifted = interpolate_uniform_1d_at(u, time, float(time[t]), delay_v)
                    y_prev = y_prev + dt * ((cvr_v * shifted - y_prev) / tau_v)
                    out_pos = window_lookup.get(t + 1)
                    if out_pos is not None:
                        flat_out[out_pos, idx] += y_prev * frac_v
            fraction_sum += np.where(tissue_mask, frac, 0.0).astype("float32")
        out = np.divide(
            out,
            fraction_sum[None, ...],
            out=np.zeros_like(out, dtype="float32"),
            where=fraction_sum[None, ...] > 1e-6,
        )
        out[:, ~m] = 0.0
        return out.astype("float32")

    def _estimate_tcnr_map_volume(self, clean: Any, noisy: Any, mask: Any) -> Any:
        np = self.np
        residual = np.asarray(noisy, dtype="float32") - np.asarray(clean, dtype="float32")
        signal_std = np.std(clean, axis=0)
        residual_std = np.std(residual, axis=0)
        tcnr = signal_std / np.maximum(residual_std, 1e-6)
        tcnr = np.nan_to_num(tcnr, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
        tcnr[~np.asarray(mask, dtype=bool)] = 0.0
        return tcnr

    def _local_spatial_uncertainty_volume(self, psc: Any, mask: Any, kernel_size: int = 3) -> Any:
        np = self.np
        ndi = __import__("scipy.ndimage").ndimage
        arr = np.asarray(psc, dtype="float32")
        size = (1, int(kernel_size), int(kernel_size), int(kernel_size))
        mean = ndi.uniform_filter(arr, size=size, mode="nearest")
        mean_sq = ndi.uniform_filter(arr * arr, size=size, mode="nearest")
        local = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype("float32")
        local[:, ~np.asarray(mask, dtype=bool)] = 0.0
        return local

    def _build_features(
        self,
        noisy: Any,
        clean: Any,
        baseline: Any,
        tcnr_map: Any,
        mask: Any,
        fractions: Any,
        vessel_likelihood: Any,
        boundary_uncertainty: Any,
        etco2_clean: Any,
    ) -> Any:
        np = self.np
        baseline_n = max(4, noisy.shape[0] // 5)
        positive = etco2_clean > max(1.0, 0.25 * float(np.max(etco2_clean)))
        if not np.any(positive):
            positive = slice(noisy.shape[0] // 2, None)
        psc_mean_baseline = np.mean(noisy[:baseline_n], axis=0)
        psc_mean_hypercapnia = np.mean(noisy[positive], axis=0)
        psc_mean = np.mean(noisy, axis=0)
        psc_std = np.std(noisy, axis=0)
        psc_delta = psc_mean_hypercapnia - psc_mean_baseline
        feature_maps = {
            "baseline_bold": baseline,
            "psc_mean_baseline": psc_mean_baseline,
            "psc_mean_hypercapnia": psc_mean_hypercapnia,
            "psc_mean": psc_mean,
            "psc_std": psc_std,
            "psc_delta": psc_delta,
            "tcnr": tcnr_map,
            "mask": mask.astype("float32"),
            "vessel_likelihood": vessel_likelihood,
            "boundary_uncertainty": boundary_uncertainty,
        }
        for frac, name in zip(fractions, TISSUE_FRACTION_NAMES, strict=True):
            feature_maps[f"fraction_{name}"] = frac
        feature_maps.update(
            temporal_response_feature_maps_numpy(
                noisy,
                etco2_clean,
                mask,
                tr_seconds=float(self.config.tr_seconds),
                time_axis=0,
                np=np,
            )
        )
        features = np.stack([feature_maps[name] for name in self.feature_names], axis=0).astype(
            "float32"
        )
        features[:, ~mask] = 0.0
        return features

    def _choose_model_etco2(self, clean: Any, measured: Any, rng: Any) -> Any:
        mode = self.config.etco2_input
        if mode == "mixed":
            mode = "measured" if float(rng.random()) < 0.75 else "clean"
        return measured.astype("float32") if mode == "measured" else clean.astype("float32")

    def _choose_temporal_indices(self, etco2: Any, rng: Any) -> Any:
        np = self.np
        n_time = int(len(etco2))
        mode = self.config.temporal_mode
        if mode == "mixed":
            mode = "windowed"
            prefer_transition = float(rng.random()) < 0.7
        else:
            prefer_transition = self.config.prefer_transition_windows
        if mode in {"full", "transition_weighted"}:
            return np.arange(n_time)
        length = min(int(self.config.temporal_window_length), n_time)
        if length >= n_time:
            return np.arange(n_time)
        starts = np.arange(0, n_time - length + 1)
        if prefer_transition:
            transition = self.compute_transition_weight(etco2, alpha=1.0)
            scores = np.asarray([np.mean(transition[s : s + length]) for s in starts], dtype=float)
            scores = scores / max(float(scores.sum()), 1e-12)
            start = int(rng.choice(starts, p=scores))
        else:
            start = int(rng.integers(0, n_time - length + 1))
        return np.arange(start, start + length)

    def _local_uncertainty_weights(self, local_uncertainty: Any, mask: Any) -> Any:
        np = self.np
        weights = 1.0 / (1e-3 + np.asarray(local_uncertainty, dtype="float32"))
        inside = weights[:, np.asarray(mask, dtype=bool)]
        scale = float(np.nanmedian(inside)) if inside.size else 1.0
        weights = weights / max(scale, 1e-6)
        weights = np.clip(weights, 0.1, 1.0).astype("float32")
        weights[:, ~np.asarray(mask, dtype=bool)] = 0.0
        return weights

    @staticmethod
    def compute_transition_weight(etco2: Any, alpha: float = 2.0) -> Any:
        np = __import__("numpy")
        u = np.asarray(etco2, dtype="float32")
        if u.size == 0:
            return u
        derivative = np.abs(np.gradient(u)).astype("float32")
        denom = max(float(np.max(derivative)), 1e-6)
        return (1.0 + float(alpha) * derivative / denom).astype("float32")

    def _tensor(self, value: Any):
        return self.torch.as_tensor(value).float()


def assert_feature_names_are_observable(feature_names: tuple[str, ...] | list[str]) -> None:
    forbidden = {"gt_cvr", "gt_delay", "gt_t", "GT_CVR", "GT_delay", "GT_T"}
    lower = {str(name).lower() for name in feature_names}
    if lower.intersection({name.lower() for name in forbidden}):
        raise ValueError("GT parameter maps are forbidden as model input features")


def temporal_response_feature_maps_numpy(
    signal: Any,
    etco2: Any,
    mask: Any,
    *,
    tr_seconds: float,
    time_axis: int,
    np: Any | None = None,
) -> dict[str, Any]:
    np = __import__("numpy") if np is None else np
    y = np.moveaxis(np.asarray(signal, dtype="float32"), int(time_axis), 0)
    u = np.asarray(etco2, dtype="float32").reshape(-1)
    if y.shape[0] != u.size:
        raise ValueError(f"signal time length {y.shape[0]} does not match ETCO2 length {u.size}")
    m = np.asarray(mask, dtype=bool)
    y_centered = y - np.mean(y, axis=0, keepdims=True)
    y_std = np.std(y, axis=0)
    corr_maps = []
    beta_maps = []
    for lag_seconds in TEMPORAL_RESPONSE_LAG_SECONDS:
        lag_frames = int(round(float(lag_seconds) / max(float(tr_seconds), 1e-6)))
        u_lag = _lag_vector_left_constant(u, lag_frames, np)
        u_centered = u_lag - float(np.mean(u_lag))
        u_var = float(np.mean(u_centered * u_centered))
        u_std = float(np.sqrt(max(u_var, 1e-12)))
        cov = np.mean(y_centered * u_centered.reshape((-1,) + (1,) * m.ndim), axis=0)
        beta = cov / max(u_var, 1e-12)
        corr = cov / np.maximum(y_std * u_std, 1e-6)
        beta_maps.append(np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0).astype("float32"))
        corr_maps.append(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype("float32"))
    beta_stack = np.stack(beta_maps, axis=0)
    corr_stack = np.stack(corr_maps, axis=0)
    peak_idx = np.argmax(corr_stack, axis=0)
    peak_corr = np.take_along_axis(corr_stack, peak_idx[None, ...], axis=0)[0]
    peak_beta = np.take_along_axis(beta_stack, peak_idx[None, ...], axis=0)[0]
    lags = np.asarray(TEMPORAL_RESPONSE_LAG_SECONDS, dtype="float32")
    lag_norm = lags[peak_idx] / max(float(max(TEMPORAL_RESPONSE_LAG_SECONDS)), 1e-6)
    out = {
        "psc_co2_beta_0lag": beta_stack[0],
        "psc_co2_corr_0lag": corr_stack[0],
        "psc_co2_beta_peak": peak_beta,
        "psc_co2_corr_peak": peak_corr,
        "psc_co2_lag_peak_norm": lag_norm.astype("float32"),
    }
    for key, value in out.items():
        value = np.asarray(value, dtype="float32")
        value[~m] = 0.0
        out[key] = value
    return out


def _lag_vector_left_constant(values: Any, lag_frames: int, np: Any) -> Any:
    values = np.asarray(values, dtype="float32")
    lag = max(0, int(lag_frames))
    if lag == 0:
        return values
    if lag >= values.size:
        return np.full_like(values, float(values[0]))
    return np.concatenate(
        [np.full(lag, float(values[0]), dtype="float32"), values[:-lag].astype("float32")]
    )
