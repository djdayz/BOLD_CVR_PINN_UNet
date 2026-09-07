from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def add_gaussian_noise(signal: Any, sigma: float, seed: int | None = None) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    rng = np.random.default_rng(seed)
    return np.asarray(signal) + rng.normal(0.0, sigma, size=np.asarray(signal).shape)


def add_linear_drift(signal: Any, amplitude: float = 0.1) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(signal)
    drift = np.linspace(-amplitude, amplitude, arr.shape[0]).reshape((-1,) + (1,) * (arr.ndim - 1))
    return arr + drift
