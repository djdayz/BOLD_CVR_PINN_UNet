from __future__ import annotations

from pathlib import Path
from typing import Any
import csv

from hybrid_cvr.config import require_dependency
from hybrid_cvr.cvr.fit_real import (
    RealCVRFitConfig,
    fit_regressor_grid,
    grid_values,
    load_resampled_delta_etco2,
    nuisance_basis,
    precompute_flat_ode_regressors,
)
from hybrid_cvr.cvr.ode import delayed_input_numpy


def save_fit_alignment_plot(
    psc_path: str | Path,
    etco2_resampled_path: str | Path,
    brain_mask_path: str | Path,
    out_path: str | Path,
    config: RealCVRFitConfig,
    title: str,
) -> dict[str, Path | float | str]:
    np = require_dependency("numpy", "pip install numpy")
    nib = require_dependency("nibabel", "pip install nibabel")
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    psc = nib.load(str(psc_path)).get_fdata(dtype=np.float32)
    if psc.ndim != 4:
        raise ValueError(f"Expected 4D BOLD PSC image, got {psc.shape}")
    mask = nib.load(str(brain_mask_path)).get_fdata() > 0
    if mask.shape != psc.shape[:3]:
        raise ValueError(f"Mask shape {mask.shape} does not match BOLD shape {psc.shape[:3]}")

    time_grid, etco2 = load_resampled_delta_etco2(etco2_resampled_path)
    if time_grid.size != psc.shape[-1]:
        raise ValueError(
            f"ETCO2 time grid length {time_grid.size} does not match BOLD timepoints {psc.shape[-1]}"
        )

    y = np.nanmedian(psc[mask], axis=0).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y_fit = y[:, None]

    glm_delays = grid_values(
        config.delay_min if config.glm_delay_min is None else config.glm_delay_min,
        config.delay_max if config.glm_delay_max is None else config.glm_delay_max,
        config.delay_step if config.glm_delay_step is None else config.glm_delay_step,
    )
    hrf_delays = grid_values(config.delay_min, config.delay_max, config.delay_step)
    T_values = grid_values(config.T_min, config.T_max, config.T_step)

    glm_regs = delayed_input_numpy(etco2, time_grid, glm_delays).T.astype(np.float32)
    glm = fit_regressor_grid(
        y_fit, glm_regs, time_grid, config.include_drift, config.candidate_batch_size
    )
    glm_i = int(glm["best_index"][0])
    glm_reg = glm_regs[glm_i]
    glm_pred = _predict_1d(y, glm_reg, float(glm["beta"][0]), config.include_drift)
    glm_delay = float(glm_delays[glm_i])
    glm_shifted = delayed_input_numpy(etco2, time_grid, np.asarray([glm_delay], dtype=np.float32))[
        :, 0
    ]

    hrf_regs = precompute_flat_ode_regressors(etco2, time_grid, hrf_delays, T_values).astype(
        np.float32
    )
    hrf = fit_regressor_grid(
        y_fit, hrf_regs, time_grid, config.include_drift, config.candidate_batch_size
    )
    hrf_i = int(hrf["best_index"][0])
    hrf_delay_i = hrf_i // T_values.size
    hrf_T_i = hrf_i % T_values.size
    hrf_delay = float(hrf_delays[hrf_delay_i])
    hrf_T = float(T_values[hrf_T_i])
    hrf_reg = hrf_regs[hrf_i]
    hrf_pred = _predict_1d(y, hrf_reg, float(hrf["beta"][0]), config.include_drift)
    hrf_shifted = delayed_input_numpy(etco2, time_grid, np.asarray([hrf_delay], dtype=np.float32))[
        :, 0
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(title)
    _plot_panel(
        axes[0],
        time_grid,
        y,
        glm_pred,
        glm_shifted,
        "Lagged GLM",
        "Shifted delta ETCO2",
        f"delay={glm_delay:.2f}s, CVR={float(glm['beta'][0]):.3f}, "
        f"R2={float(glm['r2'][0]):.3f}, RMSE={_rmse(y, glm_pred):.3f}",
    )
    _plot_panel(
        axes[1],
        time_grid,
        y,
        hrf_pred,
        hrf_reg,
        "Exponential-HRF / ODE",
        "ODE regressor",
        f"delay={hrf_delay:.2f}s, T={hrf_T:.2f}s, CVR={float(hrf['beta'][0]):.3f}, "
        f"R2={float(hrf['r2'][0]):.3f}, RMSE={_rmse(y, hrf_pred):.3f}",
    )
    axes[1].set_xlabel("BOLD time (s)")
    for ax in axes:
        ax.set_xlim(float(time_grid[0]), float(time_grid[-1]))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    outputs: dict[str, Path | float | str] = {
        "alignment_qc": out_path,
        "glm_delay_seconds": glm_delay,
        "glm_cvr": float(glm["beta"][0]),
        "glm_r2": float(glm["r2"][0]),
        "glm_rmse": _rmse(y, glm_pred),
        "hrf_delay_seconds": hrf_delay,
        "hrf_T_seconds": hrf_T,
        "hrf_cvr": float(hrf["beta"][0]),
        "hrf_r2": float(hrf["r2"][0]),
        "hrf_rmse": _rmse(y, hrf_pred),
    }
    write_alignment_metadata(out_path.with_name("fit_alignment_qc.csv"), outputs)
    return outputs


def write_alignment_metadata(path: Path, outputs: dict[str, Path | float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in outputs.items():
            writer.writerow([key, value])


def _predict_1d(y: Any, regressor: Any, beta: float, include_drift: bool) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    y = np.asarray(y, dtype=np.float32)
    regressor = np.asarray(regressor, dtype=np.float32)
    nuisance = nuisance_basis(y.size, include_drift).astype(np.float32)
    gamma = np.linalg.pinv(nuisance) @ (y - beta * regressor)
    return beta * regressor + nuisance @ gamma


def _plot_panel(
    ax: Any,
    time_grid: Any,
    bold: Any,
    prediction: Any,
    etco2_regressor: Any,
    title: str,
    regressor_label: str,
    annotation: str,
) -> None:
    ax.plot(time_grid, bold, color="#111827", linewidth=1.3, label="Median brain BOLD PSC")
    ax.plot(time_grid, prediction, color="#2563EB", linewidth=1.6, label="Fitted BOLD PSC")
    ax.set_title(f"{title}: {annotation}")
    ax.set_ylabel("BOLD PSC (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(
        time_grid,
        etco2_regressor,
        color="#DC2626",
        linewidth=1.0,
        alpha=0.65,
        linestyle="--",
        label=regressor_label,
    )
    ax2.set_ylabel(regressor_label)
    ax2.legend(loc="upper right")


def _rmse(observed: Any, predicted: Any) -> float:
    np = require_dependency("numpy", "pip install numpy")
    observed = np.asarray(observed, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))
