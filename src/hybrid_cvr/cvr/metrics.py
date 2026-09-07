from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def regression_metrics(pred: Any, target: Any, mask: Any | None = None) -> dict[str, float]:
    np = require_dependency("numpy", "pip install numpy")
    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    if mask is not None:
        keep = np.asarray(mask) > 0
        p = p[keep]
        t = t[keep]
    err = p - t
    corr = float(np.corrcoef(p, t)[0, 1]) if p.size > 1 and np.std(p) > 0 and np.std(t) > 0 else 0.0
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "pearson": corr,
    }
