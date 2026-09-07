from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def make_time_grid(n_timepoints: int, tr_seconds: float) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    return np.arange(n_timepoints, dtype=float) * float(tr_seconds)


def finite_difference(y: Any, time_grid: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    return np.gradient(y, time_grid, axis=0)
