from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

from hybrid_cvr.config import require_dependency

MMHG_PER_PERCENT_ATM = 760.0 / 100.0


@dataclass(frozen=True)
class GasConfig:
    time_column: str | int | None = None
    co2_column: str | int | None = None
    sampling_rate_hz: float = 20.0
    co2_units: str = "percent"
    min_distance_seconds: float = 1.0
    prominence: float | None = None
    smoothing_seconds: float = 2.0
    baseline_first_seconds: float = 45.0
    baseline_reference: str = "first_peak"
    baseline_statistic: str = "mean"


def percent_to_mmhg(co2_percent: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    return np.asarray(co2_percent, dtype=float) * MMHG_PER_PERCENT_ATM


def load_gas_table(path: str | Path, config: GasConfig) -> tuple[Any, Any]:
    np = require_dependency("numpy", "pip install numpy")
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    delimiter = "\t" if "\t" in first else None
    headers = first.split(delimiter) if delimiter else first.split()

    def is_float(text: str) -> bool:
        try:
            float(text)
            return True
        except ValueError:
            return False

    has_header = not all(is_float(token) for token in headers)
    if has_header:
        rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            if delimiter:
                reader = csv.DictReader(f, delimiter=delimiter)
                rows.extend(dict(row) for row in reader)
            else:
                names = f.readline().strip().split()
                for line in f:
                    values = line.strip().split()
                    rows.append(dict(zip(names, values, strict=False)))
        time_col = config.time_column if config.time_column is not None else headers[0]
        co2_col = config.co2_column if config.co2_column is not None else headers[-1]
        time = np.asarray([float(row[str(time_col)]) for row in rows], dtype=float)
        co2 = np.asarray([float(row[str(co2_col)]) for row in rows], dtype=float)
    else:
        arr = np.loadtxt(path)
        arr = np.atleast_2d(arr)
        if arr.shape[1] == 1:
            co2 = arr[:, 0]
            time = np.arange(arr.shape[0], dtype=float) / config.sampling_rate_hz
        else:
            time_idx = int(config.time_column) if config.time_column is not None else 0
            co2_idx = int(config.co2_column) if config.co2_column is not None else arr.shape[1] - 1
            time = arr[:, time_idx]
            co2 = arr[:, co2_idx]
    if config.co2_units.lower() == "percent":
        co2 = percent_to_mmhg(co2)
    elif config.co2_units.lower() != "mmhg":
        raise ValueError("co2_units must be 'percent' or 'mmHg'")
    return time, co2


def detect_etco2_peaks(time: Any, co2_mmhg: Any, config: GasConfig) -> tuple[Any, Any]:
    np = require_dependency("numpy", "pip install numpy")
    co2_mmhg = np.asarray(co2_mmhg, dtype=float)
    time = np.asarray(time, dtype=float)
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0 / config.sampling_rate_hz
    distance = max(1, int(round(config.min_distance_seconds / dt)))
    candidates = np.where((co2_mmhg[1:-1] > co2_mmhg[:-2]) & (co2_mmhg[1:-1] >= co2_mmhg[2:]))[0] + 1
    if config.prominence is not None:
        left = co2_mmhg[np.maximum(candidates - distance, 0)]
        right = co2_mmhg[np.minimum(candidates + distance, co2_mmhg.size - 1)]
        local_base = np.maximum(left, right)
        candidates = candidates[(co2_mmhg[candidates] - local_base) >= float(config.prominence)]
    if candidates.size:
        order = candidates[np.argsort(co2_mmhg[candidates])[::-1]]
        selected: list[int] = []
        blocked = np.zeros(co2_mmhg.shape, dtype=bool)
        for idx in order:
            if not blocked[idx]:
                selected.append(int(idx))
                blocked[max(0, idx - distance) : min(co2_mmhg.size, idx + distance + 1)] = True
        peaks = np.asarray(sorted(selected), dtype=int)
    else:
        peaks = candidates
    values = co2_mmhg[peaks]
    plausible = (values >= 10.0) & (values <= 100.0)
    return time[peaks][plausible], values[plausible]


def interpolate_etco2_to_grid(peak_time: Any, peak_co2: Any, target_time: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    if len(peak_time) == 0:
        return np.zeros_like(target_time, dtype=float)
    if len(peak_time) == 1:
        return np.full_like(target_time, float(peak_co2[0]), dtype=float)
    return np.interp(target_time, peak_time, peak_co2, left=peak_co2[0], right=peak_co2[-1])


def smooth_trace(trace: Any, dt: float, smoothing_seconds: float) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    if smoothing_seconds <= 0:
        return np.asarray(trace, dtype=float)
    sigma = max(smoothing_seconds / max(dt, 1e-6), 0.0)
    radius = max(1, int(round(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(np.asarray(trace, dtype=float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _baseline_value(target_time: Any, etco2: Any, peak_time: Any, config: GasConfig) -> float:
    np = require_dependency("numpy", "pip install numpy")
    target_time = np.asarray(target_time, dtype=float)
    etco2 = np.asarray(etco2, dtype=float)
    if str(config.baseline_reference).lower() == "first_peak" and len(peak_time):
        start = float(peak_time[0])
    else:
        start = float(target_time[0])
    baseline_mask = (target_time >= start) & (target_time <= (start + config.baseline_first_seconds))
    if not np.any(baseline_mask):
        fallback_end = float(target_time[0] + config.baseline_first_seconds)
        baseline_mask = target_time <= fallback_end
    baseline_values = etco2[baseline_mask] if np.any(baseline_mask) else etco2
    statistic = str(config.baseline_statistic).lower()
    if statistic == "median":
        return float(np.median(baseline_values))
    if statistic == "mean":
        return float(np.mean(baseline_values))
    raise ValueError("baseline_statistic must be 'mean' or 'median'")


def process_gas_trace(
    path: str | Path,
    bold_time_grid: Any,
    config: GasConfig,
    source_time_offset_seconds: float = 0.0,
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    time, co2 = load_gas_table(path, config)
    peak_time, peak_co2 = detect_etco2_peaks(time, co2, config)
    target_time = np.asarray(bold_time_grid, dtype=float)
    source_time = target_time + float(source_time_offset_seconds)
    etco2 = interpolate_etco2_to_grid(peak_time, peak_co2, source_time)
    dt = float(np.median(np.diff(target_time))) if len(target_time) > 1 else 1.0
    etco2 = smooth_trace(etco2, dt, config.smoothing_seconds)
    baseline = _baseline_value(source_time, etco2, peak_time, config)
    return {
        "raw_time": time,
        "raw_co2_mmhg": co2,
        "peak_time": peak_time,
        "peak_co2_mmhg": peak_co2,
        "time": target_time,
        "source_time": source_time,
        "source_time_offset_seconds": float(source_time_offset_seconds),
        "etco2_mmhg": etco2,
        "delta_etco2_mmhg": etco2 - baseline,
        "baseline_mmhg": baseline,
    }


def write_gas_outputs(result: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    np = require_dependency("numpy", "pip install numpy")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_gas": out_dir / "raw_gas.tsv",
        "etco2_peaks": out_dir / "etco2_peaks.tsv",
        "etco2_resampled": out_dir / "etco2_resampled.tsv",
    }
    _write_tsv(paths["raw_gas"], ["time", "co2_mmhg"], zip(result["raw_time"], result["raw_co2_mmhg"], strict=False))
    _write_tsv(
        paths["etco2_peaks"],
        ["time", "etco2_peak_mmhg"],
        zip(result["peak_time"], result["peak_co2_mmhg"], strict=False),
    )
    baseline = np.full_like(result["time"], result["baseline_mmhg"], dtype=float)
    _write_tsv(
        paths["etco2_resampled"],
        ["time", "source_time", "etco2_mmhg", "delta_etco2_mmhg", "baseline_mmhg"],
        zip(
            result["time"],
            result["source_time"],
            result["etco2_mmhg"],
            result["delta_etco2_mmhg"],
            baseline,
            strict=False,
        ),
    )
    return paths


def _write_tsv(path: Path, headers: list[str], rows: Any) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)
        writer.writerows(rows)
