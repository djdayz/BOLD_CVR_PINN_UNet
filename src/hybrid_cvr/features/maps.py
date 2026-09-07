from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def tcnr_from_psc(psc_time_first: Any, eps: float = 1e-6) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(psc_time_first)
    return np.mean(arr, axis=0) / np.maximum(np.std(arr, axis=0), eps)


def delta_psc_feature(psc_time_first: Any, baseline_n: int, challenge_slice: slice | None = None) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(psc_time_first)
    challenge_slice = challenge_slice or slice(arr.shape[0] // 2, None)
    return arr[challenge_slice].mean(axis=0) - arr[:baseline_n].mean(axis=0)
