from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def delayed_input_numpy(etco2: Any, time_grid: Any, delay_seconds: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    etco2 = np.asarray(etco2, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    delay = np.asarray(delay_seconds, dtype=float)
    flat_delay = delay.reshape(-1)
    shifted = np.empty((time_grid.size, flat_delay.size), dtype=float)
    for i, d in enumerate(flat_delay):
        shifted[:, i] = np.interp(time_grid - d, time_grid, etco2, left=etco2[0], right=etco2[-1])
    return shifted.reshape((time_grid.size,) + delay.shape)


def solve_ode_response_numpy(
    cvr_map: Any,
    delay_map: Any,
    T_map: Any,
    etco2: Any,
    time_grid: Any,
    y0: Any | None = None,
) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    cvr = np.asarray(cvr_map, dtype=float)
    delay = np.asarray(delay_map, dtype=float)
    tau = np.maximum(np.asarray(T_map, dtype=float), 1e-6)
    time_grid = np.asarray(time_grid, dtype=float)
    shifted = delayed_input_numpy(etco2, time_grid, delay)
    y = np.zeros((time_grid.size,) + cvr.shape, dtype=float)
    if y0 is not None:
        y[0] = y0
    for t in range(time_grid.size - 1):
        dt = float(time_grid[t + 1] - time_grid[t])
        dydt = (cvr * shifted[t] - y[t]) / tau
        y[t + 1] = y[t] + dt * dydt
    return y
