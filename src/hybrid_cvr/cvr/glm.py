from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency
from hybrid_cvr.cvr.ode import delayed_input_numpy


def delay_grid(min_seconds: float, max_seconds: float, step_seconds: float) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    return np.arange(min_seconds, max_seconds + 0.5 * step_seconds, step_seconds)


def fit_lagged_glm_voxels(
    psc_time_by_voxel: Any,
    etco2: Any,
    time_grid: Any,
    delays: Any,
    include_drift: bool = True,
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    y = np.asarray(psc_time_by_voxel, dtype=float)
    if y.ndim != 2:
        raise ValueError("psc_time_by_voxel must have shape T,V")
    best_ssr = np.full(y.shape[1], np.inf)
    best_beta = np.zeros(y.shape[1])
    best_delay = np.zeros(y.shape[1])
    best_r2 = np.zeros(y.shape[1])
    y_mean = y.mean(axis=0, keepdims=True)
    sst = np.sum((y - y_mean) ** 2, axis=0)
    drift = np.linspace(-1.0, 1.0, y.shape[0])
    for delay in delays:
        reg = delayed_input_numpy(etco2, time_grid, np.asarray([delay]))[:, 0]
        cols = [np.ones_like(reg), reg]
        if include_drift:
            cols.append(drift)
        X = np.stack(cols, axis=1)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        ssr = np.sum((y - pred) ** 2, axis=0)
        improved = ssr < best_ssr
        best_ssr[improved] = ssr[improved]
        best_beta[improved] = beta[1, improved]
        best_delay[improved] = delay
        best_r2[improved] = 1.0 - ssr[improved] / np.maximum(sst[improved], 1e-8)
    return {"cvr": best_beta, "delay": best_delay, "r2": best_r2, "ssr": best_ssr}
