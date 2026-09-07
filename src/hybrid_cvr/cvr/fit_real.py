from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

from hybrid_cvr.config import require_dependency
from hybrid_cvr.cvr.ode import delayed_input_numpy, solve_ode_response_numpy


@dataclass(frozen=True)
class RealCVRFitConfig:
    output_dir: Path = Path("data/derivatives/real_cvr")
    delay_min: float = 0.0
    delay_max: float = 80.0
    delay_step: float = 1.55
    T_min: float = 2.0
    T_max: float = 100.0
    T_step: float = 2.0
    include_drift: bool = True
    chunk_voxels: int = 6000
    candidate_batch_size: int = 256
    save_reconstruction: bool = False
    glm_delay_mode: str = "global"
    glm_global_delay_seconds: float | None = None
    glm_delay_min: float | None = None
    glm_delay_max: float | None = None
    glm_delay_step: float | None = None


def fit_real_cvr_session(
    psc_path: str | Path,
    etco2_resampled_path: str | Path,
    brain_mask_path: str | Path,
    mean_bold_path: str | Path,
    out_dir: str | Path,
    config: RealCVRFitConfig | None = None,
) -> dict[str, Path | str | int | float]:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    config = config or RealCVRFitConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    psc_img = nib.load(str(psc_path))
    psc = psc_img.get_fdata(dtype=np.float32)
    if psc.ndim != 4:
        raise ValueError(f"Expected 4D BOLD PSC image, got {psc.shape}")
    mean_img = nib.load(str(mean_bold_path))
    mask = nib.load(str(brain_mask_path)).get_fdata() > 0
    if mask.shape != psc.shape[:3]:
        raise ValueError(f"Mask shape {mask.shape} does not match BOLD shape {psc.shape[:3]}")

    time_grid, etco2 = load_resampled_delta_etco2(etco2_resampled_path)
    if time_grid.size != psc.shape[-1]:
        raise ValueError(
            f"ETCO2 time grid length {time_grid.size} does not match BOLD timepoints {psc.shape[-1]}"
        )

    delays = grid_values(config.delay_min, config.delay_max, config.delay_step)
    glm_delays = grid_values(
        config.delay_min if config.glm_delay_min is None else config.glm_delay_min,
        config.delay_max if config.glm_delay_max is None else config.glm_delay_max,
        config.delay_step if config.glm_delay_step is None else config.glm_delay_step,
    )
    T_values = grid_values(config.T_min, config.T_max, config.T_step)
    glm_regs = delayed_input_numpy(etco2, time_grid, glm_delays).T.astype(np.float32)
    hrf_regs = precompute_flat_ode_regressors(etco2, time_grid, delays, T_values).astype(np.float32)

    spatial_shape = psc.shape[:3]
    n_vox_total = int(mask.sum())
    flat_mask = mask.reshape(-1)
    flat_indices = np.flatnonzero(flat_mask)
    flat_psc = psc.reshape((-1, psc.shape[-1]))
    glm_delay_mode = str(config.glm_delay_mode).lower()
    if glm_delay_mode not in {"global", "voxelwise"}:
        raise ValueError("glm_delay_mode must be 'global' or 'voxelwise'")
    glm_global_index = None
    glm_global_delay = None
    glm_global_r2 = None
    if glm_delay_mode == "global":
        if config.glm_global_delay_seconds is None:
            global_y = np.nanmean(flat_psc[flat_indices], axis=0).astype(np.float32)[:, None]
            global_fit = fit_regressor_grid(
                global_y,
                glm_regs,
                time_grid,
                config.include_drift,
                config.candidate_batch_size,
            )
            glm_global_index = int(global_fit["best_index"][0])
            glm_global_r2 = float(global_fit["r2"][0])
        else:
            glm_global_index = int(np.argmin(np.abs(glm_delays - float(config.glm_global_delay_seconds))))
            glm_global_r2 = float("nan")
        glm_global_delay = float(glm_delays[glm_global_index])
        glm_fit_regs = glm_regs[glm_global_index : glm_global_index + 1]
    else:
        glm_fit_regs = glm_regs

    glm_maps = _empty_maps(spatial_shape, ["cvr", "delay", "r2", "ssr", "rmse", "drift"], np.float32)
    hrf_maps = _empty_maps(
        spatial_shape,
        [
            "cvr",
            "delay",
            "T",
            "effective_delay",
            "r2",
            "ssr",
            "rmse",
            "residual_std",
            "tcnr",
            "ode_residual_rms",
        ],
        np.float32,
    )
    corr_peak = np.full(spatial_shape, np.nan, dtype=np.float32)
    fit_quality = np.full(spatial_shape, np.nan, dtype=np.float32)
    valid_fit_mask = np.zeros(spatial_shape, dtype=np.uint8)
    reconstructed = np.zeros(psc.shape, dtype=np.float32) if config.save_reconstruction else None

    nuisance = nuisance_basis(time_grid.size, config.include_drift).astype(np.float32)
    fit_dof = max(1, time_grid.size - nuisance.shape[1] - 1)
    for start in range(0, flat_indices.size, max(1, config.chunk_voxels)):
        stop = min(start + max(1, config.chunk_voxels), flat_indices.size)
        idx = flat_indices[start:stop]
        y = flat_psc[idx].T.astype(np.float32, copy=False)
        finite = np.isfinite(y).all(axis=0)
        if not np.any(finite):
            continue
        fit_idx = idx[finite]
        fit_y = y[:, finite]

        glm = fit_regressor_grid(
            fit_y, glm_fit_regs, time_grid, config.include_drift, config.candidate_batch_size
        )
        hrf = fit_regressor_grid(fit_y, hrf_regs, time_grid, config.include_drift, config.candidate_batch_size)
        hrf_stats = prediction_qc_for_selected(
            fit_y,
            hrf_regs,
            hrf["best_index"],
            hrf["beta"],
            time_grid,
            etco2,
            delays,
            T_values,
            config.include_drift,
            save_prediction=config.save_reconstruction,
        )
        glm_drift = drift_for_selected(fit_y, glm_fit_regs, glm["best_index"], glm["beta"], nuisance)
        corr = correlation_peak(fit_y, glm_regs, time_grid, config.include_drift, config.candidate_batch_size)

        _scatter(glm_maps["cvr"], fit_idx, glm["beta"])
        if glm_delay_mode == "global":
            _scatter(glm_maps["delay"], fit_idx, np.full(fit_idx.shape, glm_global_delay, dtype=np.float32))
        else:
            _scatter(glm_maps["delay"], fit_idx, glm_delays[glm["best_index"]])
        _scatter(glm_maps["r2"], fit_idx, glm["r2"])
        _scatter(glm_maps["ssr"], fit_idx, glm["ssr"])
        _scatter(glm_maps["rmse"], fit_idx, np.sqrt(np.maximum(glm["ssr"], 0.0) / fit_dof))
        _scatter(glm_maps["drift"], fit_idx, glm_drift)
        _scatter(hrf_maps["cvr"], fit_idx, hrf["beta"])
        hrf_delay_index = hrf["best_index"] // T_values.size
        hrf_T_index = hrf["best_index"] % T_values.size
        _scatter(hrf_maps["delay"], fit_idx, delays[hrf_delay_index])
        _scatter(hrf_maps["T"], fit_idx, T_values[hrf_T_index])
        _scatter(
            hrf_maps["effective_delay"],
            fit_idx,
            delays[hrf_delay_index] + T_values[hrf_T_index] * np.log(2.0),
        )
        _scatter(hrf_maps["r2"], fit_idx, hrf["r2"])
        _scatter(hrf_maps["ssr"], fit_idx, hrf["ssr"])
        _scatter(hrf_maps["rmse"], fit_idx, np.sqrt(np.maximum(hrf["ssr"], 0.0) / fit_dof))
        _scatter(hrf_maps["residual_std"], fit_idx, hrf_stats["residual_std"])
        _scatter(hrf_maps["tcnr"], fit_idx, hrf_stats["tcnr"])
        _scatter(hrf_maps["ode_residual_rms"], fit_idx, hrf_stats["ode_residual_rms"])
        _scatter(corr_peak, fit_idx, corr)
        _scatter(fit_quality, fit_idx, np.clip(hrf["r2"], 0.0, 1.0))
        _scatter(valid_fit_mask, fit_idx, np.ones(fit_idx.shape, dtype=np.uint8))
        if reconstructed is not None and hrf_stats["prediction"] is not None:
            recon_flat = reconstructed.reshape((-1, reconstructed.shape[-1]))
            recon_flat[fit_idx] = hrf_stats["prediction"].T.astype(np.float32)

    outputs: dict[str, Path | str | int | float] = {
        "n_voxels": n_vox_total,
        "n_timepoints": int(time_grid.size),
        "n_delay_candidates": int(delays.size),
        "n_glm_delay_candidates": int(glm_delays.size),
        "n_T_candidates": int(T_values.size),
        "delay_min": float(delays.min()),
        "delay_max": float(delays.max()),
        "glm_delay_min": float(glm_delays.min()),
        "glm_delay_max": float(glm_delays.max()),
        "T_min": float(T_values.min()),
        "T_max": float(T_values.max()),
        "glm_delay_mode": glm_delay_mode,
        "glm_global_delay_seconds": "" if glm_global_delay is None else glm_global_delay,
        "glm_global_r2": "" if glm_global_r2 is None else glm_global_r2,
    }
    for name, data in [
        ("valid_fit_mask", valid_fit_mask),
        ("glm_cvr", glm_maps["cvr"]),
        ("glm_delay", glm_maps["delay"]),
        ("glm_r2", glm_maps["r2"]),
        ("glm_ssr", glm_maps["ssr"]),
        ("glm_rmse", glm_maps["rmse"]),
        ("glm_drift", glm_maps["drift"]),
        ("hrf_cvr", hrf_maps["cvr"]),
        ("hrf_delay", hrf_maps["delay"]),
        ("hrf_T", hrf_maps["T"]),
        ("hrf_effective_delay", hrf_maps["effective_delay"]),
        ("hrf_r2", hrf_maps["r2"]),
        ("hrf_ssr", hrf_maps["ssr"]),
        ("hrf_rmse", hrf_maps["rmse"]),
        ("tCNR", hrf_maps["tcnr"]),
        ("etco2_correlation_peak", corr_peak),
        ("fit_quality", fit_quality),
        ("hrf_residual_std", hrf_maps["residual_std"]),
        ("ode_residual_rms", hrf_maps["ode_residual_rms"]),
    ]:
        path = out_dir / f"{name}.nii.gz"
        save_map(data, mean_img, path)
        outputs[name] = path
    if reconstructed is not None:
        path = out_dir / "hrf_reconstructed_bold_psc.nii.gz"
        save_4d(reconstructed, psc_img, path)
        outputs["hrf_reconstructed_bold_psc"] = path
    else:
        outputs["hrf_reconstructed_bold_psc"] = ""
    write_fit_metadata(out_dir / "fit_metadata.csv", outputs)
    return outputs


def fit_regressor_grid(
    y_time_by_voxel: Any,
    regressors_candidate_by_time: Any,
    time_grid: Any,
    include_drift: bool = True,
    candidate_batch_size: int = 256,
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    y = np.asarray(y_time_by_voxel, dtype=np.float32)
    regs = np.asarray(regressors_candidate_by_time, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError("y_time_by_voxel must have shape T,V")
    if regs.ndim != 2 or regs.shape[1] != y.shape[0]:
        raise ValueError("regressors_candidate_by_time must have shape C,T")

    nuisance = nuisance_basis(y.shape[0], include_drift).astype(np.float32)
    q, _ = np.linalg.qr(nuisance)
    y_res = y - q @ (q.T @ y)
    regs_res = regs - (regs @ q) @ q.T
    y_res_ss = np.sum(y_res * y_res, axis=0)
    y_centered = y - y.mean(axis=0, keepdims=True)
    sst = np.sum(y_centered * y_centered, axis=0)
    reg_ss = np.maximum(np.sum(regs_res * regs_res, axis=1), 1e-12)

    n_vox = y.shape[1]
    best_ssr = np.full(n_vox, np.inf, dtype=np.float32)
    best_beta = np.zeros(n_vox, dtype=np.float32)
    best_index = np.zeros(n_vox, dtype=np.int32)
    for start in range(0, regs.shape[0], max(1, candidate_batch_size)):
        stop = min(start + max(1, candidate_batch_size), regs.shape[0])
        cross = regs_res[start:stop] @ y_res
        ssr = y_res_ss[None, :] - (cross * cross) / reg_ss[start:stop, None]
        local = np.argmin(ssr, axis=0)
        local_ssr = ssr[local, np.arange(n_vox)]
        improved = local_ssr < best_ssr
        if np.any(improved):
            selected_cross = cross[local[improved], np.flatnonzero(improved)]
            selected_ss = reg_ss[start + local[improved]]
            best_ssr[improved] = local_ssr[improved]
            best_beta[improved] = selected_cross / selected_ss
            best_index[improved] = start + local[improved]
    r2 = 1.0 - best_ssr / np.maximum(sst, 1e-8)
    return {
        "beta": np.nan_to_num(best_beta, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "best_index": best_index,
        "ssr": np.nan_to_num(best_ssr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        "r2": np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    }


def prediction_qc_for_selected(
    y: Any,
    regressors_candidate_by_time: Any,
    best_index: Any,
    beta: Any,
    time_grid: Any,
    etco2: Any,
    delays: Any,
    T_values: Any,
    include_drift: bool,
    save_prediction: bool = False,
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    y = np.asarray(y, dtype=np.float32)
    regs = np.asarray(regressors_candidate_by_time, dtype=np.float32)
    best_index = np.asarray(best_index, dtype=np.int32)
    beta = np.asarray(beta, dtype=np.float32)
    time_grid = np.asarray(time_grid, dtype=np.float32)
    delays = np.asarray(delays, dtype=np.float32)
    T_values = np.asarray(T_values, dtype=np.float32)
    nuisance = nuisance_basis(y.shape[0], include_drift).astype(np.float32)
    pinv_nuisance = np.linalg.pinv(nuisance)
    residual_std = np.zeros(y.shape[1], dtype=np.float32)
    tcnr = np.zeros(y.shape[1], dtype=np.float32)
    ode_residual_rms = np.zeros(y.shape[1], dtype=np.float32)
    prediction = np.zeros_like(y, dtype=np.float32) if save_prediction else None
    for cand in np.unique(best_index):
        sel = best_index == cand
        reg = regs[int(cand)]
        dynamic = reg[:, None] * beta[sel][None, :]
        gamma = pinv_nuisance @ (y[:, sel] - dynamic)
        nuisance_fit = nuisance @ gamma
        pred = dynamic + nuisance_fit
        resid = y[:, sel] - pred
        residual_std[sel] = np.std(resid, axis=0).astype(np.float32)
        tcnr[sel] = (np.std(dynamic, axis=0) / np.maximum(residual_std[sel], 1e-6)).astype(np.float32)
        delay_idx = int(cand) // T_values.size
        T_idx = int(cand) % T_values.size
        u_shifted = delayed_input_numpy(etco2, time_grid, np.asarray([delays[delay_idx]], dtype=np.float32))[:, 0]
        dy_dt = np.gradient(y[:, sel] - nuisance_fit, time_grid, axis=0)
        rhs = (beta[sel][None, :] * u_shifted[:, None] - (y[:, sel] - nuisance_fit)) / max(float(T_values[T_idx]), 1e-6)
        ode_residual_rms[sel] = np.sqrt(np.mean((dy_dt - rhs) ** 2, axis=0)).astype(np.float32)
        if prediction is not None:
            prediction[:, sel] = pred.astype(np.float32)
    return {
        "residual_std": residual_std,
        "tcnr": np.nan_to_num(tcnr, nan=0.0, posinf=0.0, neginf=0.0),
        "ode_residual_rms": np.nan_to_num(ode_residual_rms, nan=0.0, posinf=0.0, neginf=0.0),
        "prediction": prediction,
    }


def drift_for_selected(
    y: Any, regressors_candidate_by_time: Any, best_index: Any, beta: Any, nuisance: Any
) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    if nuisance.shape[1] < 2:
        return np.zeros(np.asarray(y).shape[1], dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    regs = np.asarray(regressors_candidate_by_time, dtype=np.float32)
    best_index = np.asarray(best_index, dtype=np.int32)
    beta = np.asarray(beta, dtype=np.float32)
    pinv_nuisance = np.linalg.pinv(nuisance)
    drift = np.zeros(y.shape[1], dtype=np.float32)
    for cand in np.unique(best_index):
        sel = best_index == cand
        reg = regs[int(cand)]
        gamma = pinv_nuisance @ (y[:, sel] - reg[:, None] * beta[sel][None, :])
        drift[sel] = gamma[1].astype(np.float32)
    return drift


def correlation_peak(
    y_time_by_voxel: Any,
    regressors_candidate_by_time: Any,
    time_grid: Any,
    include_drift: bool = True,
    candidate_batch_size: int = 256,
) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    y = np.asarray(y_time_by_voxel, dtype=np.float32)
    regs = np.asarray(regressors_candidate_by_time, dtype=np.float32)
    nuisance = nuisance_basis(y.shape[0], include_drift).astype(np.float32)
    q, _ = np.linalg.qr(nuisance)
    y_res = y - q @ (q.T @ y)
    regs_res = regs - (regs @ q) @ q.T
    y_ss = np.maximum(np.sum(y_res * y_res, axis=0), 1e-12)
    reg_ss = np.maximum(np.sum(regs_res * regs_res, axis=1), 1e-12)
    best = np.zeros(y.shape[1], dtype=np.float32)
    for start in range(0, regs.shape[0], max(1, candidate_batch_size)):
        stop = min(start + max(1, candidate_batch_size), regs.shape[0])
        cross = regs_res[start:stop] @ y_res
        corr = np.abs(cross / np.sqrt(reg_ss[start:stop, None] * y_ss[None, :]))
        best = np.maximum(best, np.max(corr, axis=0).astype(np.float32))
    return np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)


def precompute_flat_ode_regressors(etco2: Any, time_grid: Any, delays: Any, T_values: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    regs = []
    for delay in np.asarray(delays, dtype=float):
        for T in np.asarray(T_values, dtype=float):
            reg = solve_ode_response_numpy(
                np.asarray([1.0]), np.asarray([delay]), np.asarray([T]), etco2, time_grid
            )[:, 0]
            regs.append(reg)
    return np.asarray(regs, dtype=np.float32)


def load_resampled_delta_etco2(path: str | Path) -> tuple[Any, Any]:
    np = require_dependency("numpy", "pip install numpy")
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No ETCO2 rows found in {path}")
    time = np.asarray([float(row["time"]) for row in rows], dtype=np.float32)
    delta = np.asarray([float(row["delta_etco2_mmhg"]) for row in rows], dtype=np.float32)
    return time, delta


def grid_values(min_value: float, max_value: float, step: float) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    if step <= 0:
        raise ValueError("Grid step must be positive")
    values = np.arange(float(min_value), float(max_value) + 1e-6, float(step), dtype=np.float32)
    if not np.isclose(float(values[-1]), float(max_value)) and float(values[-1]) < float(max_value):
        values = np.append(values, np.asarray([float(max_value)], dtype=np.float32))
    return values.astype(np.float32)


def nuisance_basis(n_time: int, include_drift: bool) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    cols = [np.ones(n_time, dtype=np.float32)]
    if include_drift:
        cols.append(np.linspace(-1.0, 1.0, n_time, dtype=np.float32))
    return np.stack(cols, axis=1)


def save_map(data: Any, reference_img: Any, out_path: Path) -> None:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference_img.affine, reference_img.header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(out_path))


def save_4d(data: Any, reference_img: Any, out_path: Path) -> None:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference_img.affine, reference_img.header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(out_path))


def write_fit_metadata(path: Path, outputs: dict[str, Path | str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in outputs.items():
            writer.writerow([key, value])


def _empty_maps(shape: tuple[int, int, int], names: list[str], dtype: Any) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    return {name: np.full(shape, np.nan, dtype=dtype) for name in names}


def _scatter(target: Any, flat_indices: Any, values: Any) -> None:
    target.reshape(-1)[flat_indices] = values
