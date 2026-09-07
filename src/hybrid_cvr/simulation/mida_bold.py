from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency
from hybrid_cvr.simulation.co2_paradigms import make_paradigm
from hybrid_cvr.simulation.dataset_writer import write_npz_case
from hybrid_cvr.simulation.simulate_bold import simulate_bold_psc


TISSUE_FRACTION_NAMES = (
    "cortical_gm",
    "subcortical_gm",
    "wm",
    "vcsf",
    "vessel_like",
)

FEATURE_NAMES = (
    "psc_mean",
    "psc_std",
    "psc_delta",
    "tcnr",
    "mask",
    "fraction_cortical_gm",
    "fraction_subcortical_gm",
    "fraction_wm",
    "fraction_vcsf",
    "fraction_vessel_like",
    "vessel_likelihood",
    "boundary_uncertainty",
)


@dataclass(frozen=True)
class MidaBoldSimulationConfig:
    parameter_case_dir: Path = Path("data/simulated/mida_parameters/case_000")
    output_dir: Path = Path("data/simulated")
    output_mode: str = "volume4d"
    tcnr_levels: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
    paradigms: tuple[str, ...] = (
        "block",
        "multi_step",
        "pseudo_random_binary",
        "sinusoidal",
        "breath_hold_like",
        "resting_state_like",
    )
    slice_indices: tuple[int, ...] = field(default_factory=tuple)
    n_timepoints: int = 480
    tr_seconds: float = 1.55
    seed: int = 17
    baseline_etco2_mmhg: float = 40.0
    etco2_noise_sd_mmhg: float = 0.25
    etco2_drift_sd_mmhg: float = 0.15
    save_bold_psc_nifti: bool = True
    save_bold_intensity_nifti: bool = True
    save_slice_npz: bool = False
    save_gt_niftis: bool = False
    volume_chunk_voxels: int = 50000
    max_cases: int | None = None
    case_start_index: int = 0
    append_summary: bool = False
    ar1_rho: float = 0.35
    drift_fraction_of_noise: float = 0.15
    motion_spike_probability: float = 0.015
    motion_spike_scale: float = 3.0
    psc_spatial_smoothing_sigma_vox: float = 0.35
    baseline_intensity: float = 320.0
    baseline_bias_sd: float = 30.0


def simulate_mida_bold_dataset(config: MidaBoldSimulationConfig | None = None) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    pd = require_dependency("pandas", "pip install pandas")
    nib = require_dependency("nibabel", "pip install nibabel")
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")

    config = config or MidaBoldSimulationConfig()
    rng = np.random.default_rng(config.seed)
    case_dir = Path(config.parameter_case_dir).expanduser()
    out_dir = Path(config.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    etco2_dir = out_dir / "etco2"
    qc_dir = out_dir / "qc"
    nifti_dir = out_dir / "nifti"
    etco2_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    nifti_dir.mkdir(parents=True, exist_ok=True)

    maps, ref_img = load_mida_parameter_case(case_dir)
    fraction_stack = np.stack([maps[f"fraction_{name}"] for name in TISSUE_FRACTION_NAMES], axis=0)
    support = np.asarray(maps["region_labels"] > 0, dtype=bool)
    time_grid = np.arange(config.n_timepoints, dtype=np.float32) * float(config.tr_seconds)
    if config.output_mode.lower() in {"volume4d", "4d", "nifti4d"}:
        return simulate_mida_bold_volumes(config, maps, fraction_stack, support, ref_img, time_grid, rng, pd, nib, plt)

    slices = config.slice_indices or default_brain_slices(support, n_slices=1)

    rows: list[dict[str, Any]] = []
    case_idx = 0
    for slice_index in slices:
        z = int(slice_index)
        slice_maps = {
            "GT_CVR": maps["CVR"][:, :, z],
            "GT_delay": maps["delay"][:, :, z],
            "GT_T": maps["T"][:, :, z],
            "region_labels": maps["region_labels"][:, :, z],
            "vessel_likelihood": maps["vessel_likelihood"][:, :, z],
            "fractions": fraction_stack[:, :, :, z],
        }
        mask = slice_maps["region_labels"] > 0
        if not np.any(mask):
            continue
        boundary_uncertainty = tissue_boundary_uncertainty(slice_maps["fractions"], mask)
        baseline = baseline_intensity_map(slice_maps["fractions"], mask, config, rng)

        for paradigm in config.paradigms:
            paradigm_seed = int(rng.integers(0, 2**31 - 1))
            etco2_clean = make_paradigm(paradigm, time_grid, seed=paradigm_seed).astype(np.float32)
            for tcnr in config.tcnr_levels:
                case_id = f"sim_{case_idx:03d}"
                noise_seed = int(rng.integers(0, 2**31 - 1))
                etco2_measured = add_etco2_measurement_error(etco2_clean, time_grid, config, noise_seed)
                clean = simulate_bold_psc(
                    slice_maps["GT_CVR"],
                    slice_maps["GT_delay"],
                    slice_maps["GT_T"],
                    time_grid,
                    etco2_clean,
                )["bold_psc"]
                clean = spatially_smooth_time_series(clean, config.psc_spatial_smoothing_sigma_vox, mask)
                noisy, noise_summary = add_tcnr_noise(clean, mask, float(tcnr), config, noise_seed)
                bold_intensity = baseline[None, :, :] * (1.0 + noisy / 100.0)
                local_uncertainty = local_spatial_uncertainty(noisy, mask)
                local_weights = 1.0 / (1e-3 + local_uncertainty)
                local_weights[:, ~mask] = 0.0
                tcnr_map = estimate_tcnr_map(clean, noisy, mask)
                features = build_feature_stack(
                    noisy,
                    clean,
                    tcnr_map,
                    mask,
                    slice_maps["fractions"],
                    slice_maps["vessel_likelihood"],
                    boundary_uncertainty,
                )

                npz_path = write_npz_case(
                    out_dir / f"{case_id}.npz",
                    features=features,
                    bold_psc=noisy.astype(np.float32),
                    bold_intensity=bold_intensity.astype(np.float32),
                    etco2=etco2_measured.astype(np.float32),
                    etco2_clean=etco2_clean.astype(np.float32),
                    time_grid=time_grid.astype(np.float32),
                    mask=mask.astype(np.float32),
                    local_uncertainty=local_uncertainty.astype(np.float32),
                    local_uncertainty_weights=local_weights.astype(np.float32),
                    tissue_maps=slice_maps["fractions"].astype(np.float32),
                    vessel_likelihood=slice_maps["vessel_likelihood"].astype(np.float32),
                    boundary_uncertainty=boundary_uncertainty.astype(np.float32),
                    region_labels=slice_maps["region_labels"].astype(np.float32),
                    GT_CVR=slice_maps["GT_CVR"].astype(np.float32),
                    GT_delay=slice_maps["GT_delay"].astype(np.float32),
                    GT_T=slice_maps["GT_T"].astype(np.float32),
                )
                etco2_tsv = write_etco2_tsv(
                    etco2_dir / f"{case_id}_etco2.tsv",
                    time_grid,
                    etco2_clean,
                    etco2_measured,
                    config.baseline_etco2_mmhg,
                    pd,
                )
                etco2_png = save_etco2_plot(
                    qc_dir / f"{case_id}_etco2.png",
                    case_id,
                    paradigm,
                    tcnr,
                    time_grid,
                    etco2_clean,
                    etco2_measured,
                    plt,
                )
                gt_paths = {}
                if config.save_gt_niftis:
                    gt_paths = save_case_gt_niftis(
                        nifti_dir,
                        case_id,
                        z,
                        ref_img,
                        slice_maps["GT_CVR"],
                        slice_maps["GT_delay"],
                        slice_maps["GT_T"],
                        slice_maps["region_labels"],
                        nib,
                    )
                rows.append(
                    {
                        "case": case_id,
                        "slice_index": z,
                        "paradigm": paradigm,
                        "target_tcnr": float(tcnr),
                        "achieved_tcnr_median": noise_summary["achieved_tcnr_median"],
                        "noise_sigma_psc": noise_summary["noise_sigma_psc"],
                        "npz": str(npz_path),
                        "etco2_tsv": str(etco2_tsv),
                        "etco2_png": str(etco2_png),
                        **gt_paths,
                    }
                )
                case_idx += 1

    summary = out_dir / "simulation_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    (out_dir / "feature_names.json").write_text(json.dumps(list(FEATURE_NAMES), indent=2), encoding="utf-8")
    return {"output_dir": out_dir, "summary": summary, "n_cases": len(rows)}


def simulate_mida_bold_volumes(
    config: MidaBoldSimulationConfig,
    maps: dict[str, Any],
    fraction_stack: Any,
    support: Any,
    ref_img: Any,
    time_grid: Any,
    rng: Any,
    pd: Any,
    nib: Any,
    plt: Any,
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")

    out_dir = Path(config.output_dir).expanduser()
    bold_dir = out_dir / "bold4d"
    etco2_dir = out_dir / "etco2"
    qc_dir = out_dir / "qc"
    bold_dir.mkdir(parents=True, exist_ok=True)
    etco2_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(support, dtype=bool)
    boundary_uncertainty = tissue_boundary_uncertainty(fraction_stack, mask)
    baseline = baseline_intensity_map(fraction_stack, mask, config, rng)
    rows: list[dict[str, Any]] = []
    case_idx = int(config.case_start_index)
    generated = 0
    stop = False

    for paradigm in config.paradigms:
        paradigm_seed = int(rng.integers(0, 2**31 - 1))
        etco2_clean = make_paradigm(paradigm, time_grid, seed=paradigm_seed).astype(np.float32)
        clean = simulate_bold_psc_volume_chunked(
            maps["CVR"],
            maps["delay"],
            maps["T"],
            etco2_clean,
            time_grid,
            mask,
            chunk_voxels=config.volume_chunk_voxels,
        )
        clean = spatially_smooth_volume_time_series(clean, config.psc_spatial_smoothing_sigma_vox, mask)
        for tcnr in config.tcnr_levels:
            if config.max_cases is not None and generated >= config.max_cases:
                stop = True
                break
            case_id = f"sim_{case_idx:03d}"
            file_stem = simulation_file_stem(case_id, paradigm, tcnr)
            noise_seed = int(rng.integers(0, 2**31 - 1))
            etco2_measured = add_etco2_measurement_error(etco2_clean, time_grid, config, noise_seed)
            noisy, noise_summary = add_tcnr_noise_volume(clean, mask, float(tcnr), config, noise_seed)
            bold_intensity = baseline[..., None] * (1.0 + noisy / 100.0)
            bold_intensity = np.maximum(bold_intensity, 0.0).astype(np.float32)
            bold_intensity[~mask, :] = 0.0

            psc_path = ""
            bold_path = ""
            if config.save_bold_psc_nifti:
                psc_path = str(
                    save_4d_nifti(
                        noisy,
                        ref_img,
                        bold_dir / f"{file_stem}_bold_psc.nii.gz",
                        tr_seconds=config.tr_seconds,
                        dtype=np.float32,
                    )
                )
            if config.save_bold_intensity_nifti:
                bold_path = str(
                    save_4d_nifti(
                        bold_intensity,
                        ref_img,
                        bold_dir / f"{file_stem}_bold.nii.gz",
                        tr_seconds=config.tr_seconds,
                        dtype=np.float32,
                    )
                )
            etco2_tsv = write_etco2_tsv(
                etco2_dir / f"{file_stem}_etco2.tsv",
                time_grid,
                etco2_clean,
                etco2_measured,
                config.baseline_etco2_mmhg,
                pd,
            )
            etco2_png = save_etco2_plot(
                qc_dir / f"{file_stem}_etco2.png",
                case_id,
                paradigm,
                tcnr,
                time_grid,
                etco2_clean,
                etco2_measured,
                plt,
            )
            rows.append(
                {
                    "case": case_id,
                    "file_stem": file_stem,
                    "output_mode": "volume4d",
                    "paradigm": paradigm,
                    "target_tcnr": float(tcnr),
                    "achieved_tcnr_median": noise_summary["achieved_tcnr_median"],
                    "noise_sigma_psc": noise_summary["noise_sigma_psc"],
                    "bold_4d": bold_path,
                    "bold_psc_4d": psc_path,
                    "etco2_tsv": str(etco2_tsv),
                    "etco2_png": str(etco2_png),
                    "source_GT_CVR": str(Path(config.parameter_case_dir) / "GT_CVR.nii.gz"),
                    "source_GT_delay": str(Path(config.parameter_case_dir) / "GT_delay.nii.gz"),
                    "source_GT_T": str(Path(config.parameter_case_dir) / "GT_T.nii.gz"),
                    "boundary_uncertainty_source": "computed_from_GT_fraction_maps",
                    "mask_voxels": int(mask.sum()),
                    "shape_x": int(mask.shape[0]),
                    "shape_y": int(mask.shape[1]),
                    "shape_z": int(mask.shape[2]),
                    "timepoints": int(np.asarray(time_grid).size),
                    "tr_seconds": float(config.tr_seconds),
                }
            )
            case_idx += 1
            generated += 1
        if stop:
            break

    summary = out_dir / "simulation_summary.csv"
    table = pd.DataFrame(rows)
    if config.append_summary and summary.exists():
        table = pd.concat([pd.read_csv(summary), table], ignore_index=True)
    table.to_csv(summary, index=False)
    metadata = {
        "mode": "volume4d",
        "uses_gt_parameter_maps_internally_for_ode": True,
        "exports_gt_parameter_maps_per_simulation": False,
        "ode": "dy/dt = (CVR * delta_ETCO2(t - delay) - y) / T",
        "shape": list(mask.shape),
        "timepoints": int(np.asarray(time_grid).size),
        "tr_seconds": float(config.tr_seconds),
        "tissue_fraction_names": list(TISSUE_FRACTION_NAMES),
        "boundary_uncertainty_available_in_memory": bool(np.any(boundary_uncertainty)),
    }
    (out_dir / "simulation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"output_dir": out_dir, "summary": summary, "n_cases": len(rows)}


def load_mida_parameter_case(case_dir: Path) -> tuple[dict[str, Any], Any]:
    np = require_dependency("numpy", "pip install numpy")
    nib = require_dependency("nibabel", "pip install nibabel")
    paths = {
        "CVR": case_dir / "GT_CVR.nii.gz",
        "delay": case_dir / "GT_delay.nii.gz",
        "T": case_dir / "GT_T.nii.gz",
        "region_labels": case_dir / "GT_region_labels.nii.gz",
        "vessel_likelihood": case_dir / "GT_vessel_likelihood.nii.gz",
    }
    for name in TISSUE_FRACTION_NAMES:
        paths[f"fraction_{name}"] = case_dir / f"GT_fraction_{name}.nii.gz"
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing MIDA parameter files: " + ", ".join(missing))
    ref_img = nib.load(str(paths["CVR"]))
    maps = {name: np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32)) for name, path in paths.items()}
    return maps, ref_img


def simulation_file_stem(case_id: str, paradigm: str, tcnr: float) -> str:
    safe_paradigm = paradigm.lower().replace("-", "_").replace(" ", "_")
    safe_tcnr = f"{float(tcnr):g}".replace(".", "p")
    return f"{case_id}_{safe_paradigm}_tcnr-{safe_tcnr}"


def default_brain_slices(mask: Any, n_slices: int = 1) -> tuple[int, ...]:
    np = require_dependency("numpy", "pip install numpy")
    z_counts = np.asarray(mask, dtype=bool).sum(axis=(0, 1))
    valid = np.flatnonzero(z_counts > 0)
    if valid.size == 0:
        raise ValueError("Cannot choose simulation slices from an empty brain mask")
    if n_slices <= 1:
        return (int(valid[len(valid) // 2]),)
    positions = np.linspace(0, valid.size - 1, n_slices)
    return tuple(int(valid[int(round(pos))]) for pos in positions)


def add_etco2_measurement_error(etco2_clean: Any, time_grid: Any, config: MidaBoldSimulationConfig, seed: int) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    rng = np.random.default_rng(seed)
    clean = np.asarray(etco2_clean, dtype=np.float32)
    time = np.asarray(time_grid, dtype=np.float32)
    noise = rng.normal(0.0, config.etco2_noise_sd_mmhg, size=clean.shape).astype(np.float32)
    drift = np.linspace(-1.0, 1.0, clean.size, dtype=np.float32)
    drift *= float(rng.normal(0.0, config.etco2_drift_sd_mmhg))
    return clean + noise + drift


def spatially_smooth_time_series(psc: Any, sigma_vox: float, mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    arr = np.asarray(psc, dtype=np.float32)
    if sigma_vox <= 0:
        return arr
    smoothed = ndi.gaussian_filter(arr, sigma=(0.0, float(sigma_vox), float(sigma_vox)), mode="nearest")
    out = np.asarray(smoothed, dtype=np.float32)
    out[:, ~np.asarray(mask, dtype=bool)] = 0.0
    return out


def spatially_smooth_volume_time_series(psc: Any, sigma_vox: float, mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    arr = np.asarray(psc, dtype=np.float32)
    if sigma_vox <= 0:
        return arr
    smoothed = ndi.gaussian_filter(
        arr,
        sigma=(float(sigma_vox), float(sigma_vox), float(sigma_vox), 0.0),
        mode="nearest",
    )
    out = np.asarray(smoothed, dtype=np.float32)
    out[~np.asarray(mask, dtype=bool), :] = 0.0
    return out


def simulate_bold_psc_volume_chunked(
    cvr_map: Any,
    delay_map: Any,
    T_map: Any,
    etco2: Any,
    time_grid: Any,
    mask: Any,
    chunk_voxels: int = 50000,
) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    cvr = np.asarray(cvr_map, dtype=np.float32)
    delay = np.asarray(delay_map, dtype=np.float32)
    tau = np.maximum(np.asarray(T_map, dtype=np.float32), 1e-6)
    time = np.asarray(time_grid, dtype=np.float32)
    u = np.asarray(etco2, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    out = np.zeros(cvr.shape + (time.size,), dtype=np.float32)
    flat_out = out.reshape((-1, time.size))
    valid = np.flatnonzero(m.reshape(-1))
    if valid.size == 0:
        return out
    flat_cvr = cvr.reshape(-1)
    flat_delay = delay.reshape(-1)
    flat_tau = tau.reshape(-1)
    for start in range(0, valid.size, int(chunk_voxels)):
        idx = valid[start : start + int(chunk_voxels)]
        y_prev = np.zeros(idx.size, dtype=np.float32)
        cvr_v = flat_cvr[idx]
        delay_v = flat_delay[idx]
        tau_v = flat_tau[idx]
        for t in range(time.size - 1):
            dt = float(time[t + 1] - time[t])
            shifted = interpolate_uniform_1d_at(u, time, float(time[t]), delay_v)
            y_prev = y_prev + np.float32(dt) * ((cvr_v * shifted - y_prev) / tau_v)
            flat_out[idx, t + 1] = y_prev
    return out


def interpolate_uniform_1d_at(signal: Any, time_grid: Any, current_time: float, delay_seconds: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    signal = np.asarray(signal, dtype=np.float32)
    time = np.asarray(time_grid, dtype=np.float32)
    delay = np.asarray(delay_seconds, dtype=np.float32)
    if signal.size == 1:
        return np.full(delay.shape, signal[0], dtype=np.float32)
    dt = float(np.median(np.diff(time)))
    pos = (np.float32(current_time) - delay - np.float32(time[0])) / max(dt, 1e-6)
    lo = np.floor(pos).astype(np.int32)
    w = (pos - lo).astype(np.float32)
    lo_clipped = np.clip(lo, 0, signal.size - 1)
    hi_clipped = np.clip(lo + 1, 0, signal.size - 1)
    out = (1.0 - w) * signal[lo_clipped] + w * signal[hi_clipped]
    out = np.where(lo < 0, signal[0], out)
    out = np.where(lo >= signal.size - 1, signal[-1], out)
    return out.astype(np.float32)


def add_tcnr_noise(
    clean: Any,
    mask: Any,
    target_tcnr: float,
    config: MidaBoldSimulationConfig,
    seed: int,
) -> tuple[Any, dict[str, float]]:
    np = require_dependency("numpy", "pip install numpy")
    rng = np.random.default_rng(seed)
    y = np.asarray(clean, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    temporal_std = np.std(y[:, m], axis=0)
    signal_scale = float(np.median(temporal_std[temporal_std > 0])) if np.any(temporal_std > 0) else 1.0
    noise_sigma = signal_scale / max(float(target_tcnr), 1e-6)

    white = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32)
    ar = np.zeros_like(y, dtype=np.float32)
    innovation_scale = (1.0 - float(config.ar1_rho) ** 2) ** 0.5
    ar[0] = rng.normal(0.0, 1.0, size=y.shape[1:]).astype(np.float32)
    for t in range(1, y.shape[0]):
        innovation = rng.normal(0.0, innovation_scale, size=y.shape[1:]).astype(np.float32)
        ar[t] = float(config.ar1_rho) * ar[t - 1] + innovation
    combined = 0.55 * white + 0.45 * ar
    combined = standardize_inside_mask(combined, m)
    noise = combined * noise_sigma

    drift = np.linspace(-1.0, 1.0, y.shape[0], dtype=np.float32).reshape((-1, 1, 1))
    drift_map = rng.normal(0.0, config.drift_fraction_of_noise * noise_sigma, size=y.shape[1:]).astype(np.float32)
    noise = noise + drift * drift_map[None, :, :]

    spike_count = int(round(config.motion_spike_probability * y.shape[0]))
    if spike_count > 0:
        spike_times = rng.choice(y.shape[0], size=spike_count, replace=False)
        for t in spike_times:
            spike = rng.normal(0.0, config.motion_spike_scale * noise_sigma)
            noise[t, m] += np.float32(spike)

    noisy = y + noise
    noisy[:, ~m] = 0.0
    residual = noisy - y
    achieved = estimate_global_tcnr(y, residual, m)
    return noisy.astype(np.float32), {"noise_sigma_psc": float(noise_sigma), "achieved_tcnr_median": float(achieved)}


def add_tcnr_noise_volume(
    clean: Any,
    mask: Any,
    target_tcnr: float,
    config: MidaBoldSimulationConfig,
    seed: int,
) -> tuple[Any, dict[str, float]]:
    np = require_dependency("numpy", "pip install numpy")
    rng = np.random.default_rng(seed)
    y = np.asarray(clean, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    temporal_std = np.std(y[m, :], axis=1)
    signal_scale = float(np.median(temporal_std[temporal_std > 0])) if np.any(temporal_std > 0) else 1.0
    noise_sigma = signal_scale / max(float(target_tcnr), 1e-6)

    noise = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32) * np.float32(0.55)
    innovation_scale = np.float32((1.0 - float(config.ar1_rho) ** 2) ** 0.5)
    ar_prev = rng.normal(0.0, 1.0, size=y.shape[:3]).astype(np.float32)
    noise[..., 0] += np.float32(0.45) * ar_prev
    for t in range(1, y.shape[3]):
        innovation = rng.normal(0.0, innovation_scale, size=y.shape[:3]).astype(np.float32)
        ar_prev = np.float32(config.ar1_rho) * ar_prev + innovation
        noise[..., t] += np.float32(0.45) * ar_prev
    noise = standardize_inside_mask_time_last(noise, m) * np.float32(noise_sigma)

    drift = np.linspace(-1.0, 1.0, y.shape[3], dtype=np.float32)
    drift_map = rng.normal(0.0, config.drift_fraction_of_noise * noise_sigma, size=y.shape[:3]).astype(np.float32)
    noise += drift_map[..., None] * drift[None, None, None, :]

    spike_count = int(round(config.motion_spike_probability * y.shape[3]))
    if spike_count > 0:
        spike_times = rng.choice(y.shape[3], size=spike_count, replace=False)
        for t in spike_times:
            spike = np.float32(rng.normal(0.0, config.motion_spike_scale * noise_sigma))
            frame = noise[..., int(t)]
            frame[m] += spike

    noisy = y + noise
    noisy[~m, :] = 0.0
    achieved = estimate_global_tcnr_time_last(y, noise, m)
    return noisy.astype(np.float32), {"noise_sigma_psc": float(noise_sigma), "achieved_tcnr_median": float(achieved)}


def standardize_inside_mask(arr: Any, mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    out = np.asarray(arr, dtype=np.float32).copy()
    vals = out[:, np.asarray(mask, dtype=bool)]
    std = float(np.std(vals))
    mean = float(np.mean(vals))
    if std > 0:
        out = (out - mean) / std
    return out


def standardize_inside_mask_time_last(arr: Any, mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    out = np.asarray(arr, dtype=np.float32).copy()
    vals = out[np.asarray(mask, dtype=bool), :]
    std = float(np.std(vals))
    mean = float(np.mean(vals))
    if std > 0:
        out = (out - mean) / std
    return out


def estimate_global_tcnr(clean: Any, residual: Any, mask: Any) -> float:
    np = require_dependency("numpy", "pip install numpy")
    m = np.asarray(mask, dtype=bool)
    signal_std = np.std(np.asarray(clean)[:, m], axis=0)
    residual_std = np.std(np.asarray(residual)[:, m], axis=0)
    tcnr = signal_std / np.maximum(residual_std, 1e-6)
    return float(np.median(tcnr[np.isfinite(tcnr)]))


def estimate_global_tcnr_time_last(clean: Any, residual: Any, mask: Any) -> float:
    np = require_dependency("numpy", "pip install numpy")
    m = np.asarray(mask, dtype=bool)
    signal_std = np.std(np.asarray(clean)[m, :], axis=1)
    residual_std = np.std(np.asarray(residual)[m, :], axis=1)
    tcnr = signal_std / np.maximum(residual_std, 1e-6)
    return float(np.median(tcnr[np.isfinite(tcnr)]))


def estimate_tcnr_map(clean: Any, noisy: Any, mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    residual = np.asarray(noisy) - np.asarray(clean)
    signal_std = np.std(clean, axis=0)
    residual_std = np.std(residual, axis=0)
    tcnr = signal_std / np.maximum(residual_std, 1e-6)
    tcnr = np.nan_to_num(tcnr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    tcnr[~np.asarray(mask, dtype=bool)] = 0.0
    return tcnr


def tissue_boundary_uncertainty(fractions: Any, mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    frac = np.asarray(fractions, dtype=np.float32)
    boundary = 1.0 - np.max(frac, axis=0)
    boundary = np.clip(boundary, 0.0, 1.0).astype(np.float32)
    boundary[~np.asarray(mask, dtype=bool)] = 0.0
    return boundary


def baseline_intensity_map(fractions: Any, mask: Any, config: MidaBoldSimulationConfig, rng: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    frac = np.asarray(fractions, dtype=np.float32)
    tissue_offsets = np.asarray([40.0, 30.0, -40.0, 120.0, 80.0], dtype=np.float32)
    base = np.full(frac.shape[1:], config.baseline_intensity, dtype=np.float32)
    offset_shape = (tissue_offsets.size,) + (1,) * (frac.ndim - 1)
    base += np.sum(frac * tissue_offsets.reshape(offset_shape), axis=0)
    bias = rng.normal(0.0, 1.0, size=base.shape).astype(np.float32)
    bias = ndi.gaussian_filter(bias, sigma=10.0, mode="nearest")
    bias = bias / max(float(np.std(bias[np.asarray(mask, dtype=bool)])), 1e-6)
    base += bias * float(config.baseline_bias_sd)
    base = np.maximum(base, 1.0)
    base[~np.asarray(mask, dtype=bool)] = 0.0
    return base.astype(np.float32)


def local_spatial_uncertainty(psc: Any, mask: Any, kernel_size: int = 3) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    arr = np.asarray(psc, dtype=np.float32)
    size = (1, int(kernel_size), int(kernel_size))
    mean = ndi.uniform_filter(arr, size=size, mode="nearest")
    mean_sq = ndi.uniform_filter(arr * arr, size=size, mode="nearest")
    local = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)
    local[:, ~np.asarray(mask, dtype=bool)] = 0.0
    return local


def build_feature_stack(
    noisy: Any,
    clean: Any,
    tcnr_map: Any,
    mask: Any,
    fractions: Any,
    vessel_likelihood: Any,
    boundary_uncertainty: Any,
) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    observed = np.asarray(noisy, dtype=np.float32)
    clean_arr = np.asarray(clean, dtype=np.float32)
    baseline_n = max(4, observed.shape[0] // 5)
    challenge = slice(observed.shape[0] // 2, None)
    psc_mean = np.mean(observed, axis=0)
    psc_std = np.std(observed, axis=0)
    psc_delta = np.mean(clean_arr[challenge], axis=0) - np.mean(clean_arr[:baseline_n], axis=0)
    feature_maps = [
        psc_mean,
        psc_std,
        psc_delta,
        np.asarray(tcnr_map, dtype=np.float32),
        np.asarray(mask, dtype=np.float32),
        *[np.asarray(f, dtype=np.float32) for f in np.asarray(fractions, dtype=np.float32)],
        np.asarray(vessel_likelihood, dtype=np.float32),
        np.asarray(boundary_uncertainty, dtype=np.float32),
    ]
    out = np.stack(feature_maps, axis=0).astype(np.float32)
    out[:, ~np.asarray(mask, dtype=bool)] = 0.0
    return out


def write_etco2_tsv(
    out_path: Path,
    time_grid: Any,
    etco2_clean: Any,
    etco2_measured: Any,
    baseline: float,
    pd: Any,
) -> Path:
    df = pd.DataFrame(
        {
            "time_seconds": time_grid,
            "delta_etco2_clean_mmhg": etco2_clean,
            "delta_etco2_measured_mmhg": etco2_measured,
            "etco2_clean_mmhg": baseline + etco2_clean,
            "etco2_measured_mmhg": baseline + etco2_measured,
        }
    )
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def save_etco2_plot(
    out_path: Path,
    case_id: str,
    paradigm: str,
    tcnr: float,
    time_grid: Any,
    etco2_clean: Any,
    etco2_measured: Any,
    plt: Any,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    ax.plot(time_grid, etco2_clean, label="Clean delta ETCO2", linewidth=1.6)
    ax.plot(time_grid, etco2_measured, label="Measured delta ETCO2", linewidth=0.9, alpha=0.75)
    ax.set_title(f"{case_id} {paradigm} tCNR {tcnr:g}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Delta ETCO2 (mmHg)")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_case_gt_niftis(
    out_dir: Path,
    case_id: str,
    slice_index: int,
    ref_img: Any,
    cvr: Any,
    delay: Any,
    T: Any,
    labels: Any,
    nib: Any,
) -> dict[str, str]:
    np = require_dependency("numpy", "pip install numpy")
    affine = np.asarray(ref_img.affine, dtype=float).copy()
    affine[:3, 3] = affine[:3, 3] + affine[:3, 2] * float(slice_index)
    header = ref_img.header.copy()
    paths = {}
    for key, arr, dtype in (
        ("gt_cvr", cvr, np.float32),
        ("gt_delay", delay, np.float32),
        ("gt_T", T, np.float32),
        ("region_labels", labels, np.uint8),
    ):
        path = out_dir / f"{case_id}_{key}.nii.gz"
        img = nib.Nifti1Image(np.asarray(arr).astype(dtype)[:, :, None], affine, header)
        img.set_data_dtype(dtype)
        nib.save(img, str(path))
        paths[key] = str(path)
    return paths


def save_4d_nifti(data: Any, ref_img: Any, out_path: Path, tr_seconds: float, dtype: Any) -> Path:
    np = require_dependency("numpy", "pip install numpy")
    nib = require_dependency("nibabel", "pip install nibabel")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ref_img.header.copy()
    arr = np.asarray(data).astype(dtype)
    img = nib.Nifti1Image(arr, ref_img.affine, header)
    zooms = tuple(float(z) for z in ref_img.header.get_zooms()[:3])
    img.header.set_data_shape(arr.shape)
    img.header.set_zooms(zooms + (float(tr_seconds),))
    img.set_data_dtype(dtype)
    nib.save(img, str(out_path))
    return out_path
