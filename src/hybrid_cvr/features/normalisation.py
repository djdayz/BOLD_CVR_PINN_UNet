from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def robust_zscore(x: Any, mask: Any | None = None, eps: float = 1e-6) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(x, dtype=float)
    vals = arr[np.asarray(mask) > 0] if mask is not None else arr.ravel()
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    return ((arr - med) / (1.4826 * mad + eps)).astype("float32")
