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
    baseline_intensity_map,
    default_brain_slices,
    estimate_tcnr_map,
    load_mida_parameter_case,
    local_spatial_uncertainty,
    tissue_boundary_uncertainty,
)
from hybrid_cvr.simulation.simulate_bold import simulate_bold_psc


ON_THE_FLY_FEATURE_NAMES = (
    "baseline_bold",
    "psc_mean_baseline",
    "psc_mean_hypercapnia",
    *FEATURE_NAMES,
)


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
    paradigms: tuple[str, ...] = (
        "block",
        "multi_step",
        "pseudo_random_binary",
        "sinusoidal",
        "breath_hold_like",
        "resting_state_like",
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
    sampling_strategy: str = "balanced_grid"
    randomize_artifacts_for_training: bool = True
    etco2_noise_sd_mmhg_range: tuple[float, float] = (0.0, 0.40)
    etco2_drift_sd_mmhg_range: tuple[float, float] = (0.0, 0.25)
    drift_fraction_of_noise_range: tuple[float, float] = (0.0, 0.25)
    motion_spike_probability_options: tuple[float, ...] = (0.0, 0.005, 0.015, 0.03)
    motion_spike_scale_range: tuple[float, float] = (1.5, 4.0)
    ar1_rho_range: tuple[float, float] = (0.15, 0.55)
    return_consistency_pair: bool = False
    consistency_seed_offset: int = 50_000_000


class OnTheFlyCVRDataset(__import__("torch").utils.data.Dataset):
    """Generate simulated BOLD observations in memory from stored GT parameter cases."""

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
        if config.sampling_strategy not in {"balanced_grid", "random"}:
            raise ValueError("sampling_strategy must be balanced_grid or random")
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
        boundary_uncertainty = tissue_boundary_uncertainty(slice_maps["fractions"], mask)
        clean = simulate_bold_psc(
            slice_maps["GT_CVR"],
            slice_maps["GT_delay"],
            slice_maps["GT_T"],
            time_grid,
            etco2_clean,
        )["bold_psc"].astype("float32")
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
                "paradigm": paradigm,
                "target_tcnr": float(tcnr),
                "paradigm_seed": int(paradigm_seed),
                **view["metadata"],
            },
            "gt_cvr": self._tensor(slice_maps["GT_CVR"]),
            "gt_delay": self._tensor(slice_maps["GT_delay"]),
            "gt_T": self._tensor(slice_maps["GT_T"]),
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
        noisy, noise_summary = add_tcnr_noise(clean, mask, tcnr, sim_config, noise_seed)
        baseline = baseline_intensity_map(slice_maps["fractions"], mask, sim_config, rng)
        local_uncertainty = local_spatial_uncertainty(noisy, mask).astype("float32")
        local_weights = self._local_uncertainty_weights(local_uncertainty, mask)
        etco2_for_transition = etco2_measured if self.config.etco2_input != "clean" else etco2_clean
        transition_weight = self.compute_transition_weight(
            etco2_for_transition,
            alpha=self.config.transition_alpha,
        )
        tcnr_map = estimate_tcnr_map(clean, noisy, mask)
        features = self._build_features(
            noisy,
            clean,
            baseline,
            tcnr_map,
            mask,
            slice_maps["fractions"],
            slice_maps["vessel_likelihood"],
            boundary_uncertainty,
            etco2_clean,
        )
        etco2_for_model = self._choose_model_etco2(etco2_clean, etco2_measured, rng)
        window = self._choose_temporal_indices(etco2_for_model, rng)
        return {
            "features": features,
            "bold_psc": noisy[window],
            "etco2_clean": etco2_clean[window],
            "etco2_measured": etco2_measured[window],
            "etco2_for_model": etco2_for_model[window],
            "time_grid": time_grid[window] - time_grid[window][0],
            "local_uncertainty": local_uncertainty[window],
            "local_uncertainty_weights": local_weights[window],
            "transition_weight": transition_weight[window],
            "metadata": {
                "noise_seed": int(noise_seed),
                **artifact_settings,
                **noise_summary,
            },
        }

    def _choose_condition(self, idx: int, rng: Any) -> tuple[str, float]:
        if self.config.sampling_strategy == "balanced_grid":
            return self.condition_grid[int(idx) % len(self.condition_grid)]
        return str(rng.choice(self.config.paradigms)), float(rng.choice(self.config.tcnr_levels))

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
        return {
            "GT_CVR": np.take(maps["CVR"], z, axis=axis).astype("float32"),
            "GT_delay": np.take(maps["delay"], z, axis=axis).astype("float32"),
            "GT_T": np.take(maps["T"], z, axis=axis).astype("float32"),
            "region_labels": np.take(maps["region_labels"], z, axis=axis),
            "vessel_likelihood": np.take(maps["vessel_likelihood"], z, axis=axis).astype(
                "float32"
            ),
            "fractions": np.take(fractions, z, axis=axis + 1).astype("float32"),
        }

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
