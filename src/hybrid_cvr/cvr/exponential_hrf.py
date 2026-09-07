from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency
from hybrid_cvr.cvr.ode import solve_ode_response_numpy


def precompute_ode_regressors(etco2: Any, time_grid: Any, delays: Any, T_values: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    regressors = []
    for delay in delays:
        row = []
        for T in T_values:
            reg = solve_ode_response_numpy(
                np.asarray([1.0]), np.asarray([delay]), np.asarray([T]), etco2, time_grid
            )[:, 0]
            row.append(reg)
        regressors.append(row)
    return np.asarray(regressors)


def fit_exponential_hrf_voxels(
    psc_time_by_voxel: Any,
    etco2: Any,
    time_grid: Any,
    delays: Any,
    T_values: Any,
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    y = np.asarray(psc_time_by_voxel, dtype=float)
    best_ssr = np.full(y.shape[1], np.inf)
    best_beta = np.zeros(y.shape[1])
    best_delay = np.zeros(y.shape[1])
    best_T = np.zeros(y.shape[1])
    best_r2 = np.zeros(y.shape[1])
    y_mean = y.mean(axis=0, keepdims=True)
    sst = np.sum((y - y_mean) ** 2, axis=0)
    drift = np.linspace(-1.0, 1.0, y.shape[0])
    regs = precompute_ode_regressors(etco2, time_grid, delays, T_values)
    for i, delay in enumerate(delays):
        for j, T in enumerate(T_values):
            reg = regs[i, j]
            X = np.stack([np.ones_like(reg), reg, drift], axis=1)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            ssr = np.sum((y - pred) ** 2, axis=0)
            improved = ssr < best_ssr
            best_ssr[improved] = ssr[improved]
            best_beta[improved] = beta[1, improved]
            best_delay[improved] = delay
            best_T[improved] = T
            best_r2[improved] = 1.0 - ssr[improved] / np.maximum(sst[improved], 1e-8)
    return {"cvr": best_beta, "delay": best_delay, "T": best_T, "r2": best_r2, "ssr": best_ssr}
