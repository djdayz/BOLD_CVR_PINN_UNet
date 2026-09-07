from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency


def save_trace_qc(time: Any, trace: Any, out: str | Path, title: str = "Trace QC") -> Path:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(time, trace)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _read_float_tsv(path: str | Path) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in TSV: {path}")
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in columns:
                columns[name].append(float(row[name]))
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _smooth_trace(trace: Any, dt: float, smoothing_seconds: float = 2.0) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    if smoothing_seconds <= 0 or len(trace) < 2:
        return np.asarray(trace, dtype=float)
    sigma = max(smoothing_seconds / max(dt, 1e-6), 0.0)
    radius = max(1, int(round(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(trace, dtype=float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _interpolate_peaks_to_time_grid(time_grid: Any, peaks: dict[str, Any], fallback_etco2: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    if len(peaks["time"]) == 0:
        return np.full_like(time_grid, float(np.nanmedian(fallback_etco2)), dtype=float)
    if len(peaks["time"]) == 1:
        return np.full_like(time_grid, float(peaks["etco2_peak_mmhg"][0]), dtype=float)
    return np.interp(
        time_grid,
        peaks["time"],
        peaks["etco2_peak_mmhg"],
        left=peaks["etco2_peak_mmhg"][0],
        right=peaks["etco2_peak_mmhg"][-1],
    )


def save_etco2_qc_plot(
    raw_gas_tsv: str | Path,
    peaks_tsv: str | Path,
    resampled_tsv: str | Path,
    out: str | Path,
    title: str = "ETCO2 QC",
) -> Path:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    np = require_dependency("numpy", "pip install numpy")

    raw = _read_float_tsv(raw_gas_tsv)
    peaks = _read_float_tsv(peaks_tsv)
    resampled = _read_float_tsv(resampled_tsv)

    for required in ("time", "co2_mmhg"):
        if required not in raw:
            raise ValueError(f"Missing raw gas column '{required}' in {raw_gas_tsv}")
    for required in ("time", "etco2_peak_mmhg"):
        if required not in peaks:
            raise ValueError(f"Missing ETCO2 peak column '{required}' in {peaks_tsv}")
    for required in ("time", "etco2_mmhg", "delta_etco2_mmhg", "baseline_mmhg"):
        if required not in resampled:
            raise ValueError(f"Missing resampled ETCO2 column '{required}' in {resampled_tsv}")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    fig.suptitle(title)

    axes[0].plot(
        raw["time"],
        raw["co2_mmhg"],
        color="#4B5563",
        linewidth=0.7,
        alpha=0.8,
        label="Raw CO2",
    )
    if len(peaks["time"]):
        axes[0].scatter(
            peaks["time"],
            peaks["etco2_peak_mmhg"],
            s=12,
            color="#DC2626",
            alpha=0.85,
            label="Detected ETCO2 peaks",
            zorder=3,
        )
    axes[0].set_ylabel("CO2 (mmHg)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    x_min = float(np.nanmin(raw["time"]))
    x_max = float(np.nanmax(raw["time"]))
    axes[0].set_xlim(x_min, x_max)
    resampled_x = resampled["source_time"] if "source_time" in resampled else resampled["time"]
    baseline = float(np.nanmedian(resampled["baseline_mmhg"]))
    resampled_dt = float(np.nanmedian(np.diff(resampled_x))) if len(resampled_x) > 1 else 1.0
    full_time = np.arange(x_min, x_max + resampled_dt, resampled_dt, dtype=float)
    full_etco2 = _interpolate_peaks_to_time_grid(full_time, peaks, resampled["etco2_mmhg"])
    full_etco2 = _smooth_trace(full_etco2, resampled_dt)
    full_delta = full_etco2 - baseline

    axes[1].plot(
        full_time,
        full_etco2,
        color="#2563EB",
        linewidth=1.4,
        label="Interpolated ETCO2",
    )
    axes[1].plot(
        resampled_x,
        resampled["etco2_mmhg"],
        color="#1D4ED8",
        linewidth=0.8,
        alpha=0.75,
        label="BOLD-frame ETCO2",
    )
    axes[1].plot(
        full_time,
        full_delta,
        color="#059669",
        linewidth=1.1,
        label="Delta ETCO2",
    )
    axes[1].axhline(
        baseline,
        color="#111827",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label="Baseline",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("ETCO2 (mmHg)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlim(x_min, x_max)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
