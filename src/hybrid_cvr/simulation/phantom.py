from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def make_phantom_labels(shape: tuple[int, int, int] = (32, 32, 8)) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    yy, xx, zz = np.indices(shape)
    cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    labels = np.zeros(shape, dtype="uint8")
    labels[r < min(shape[:2]) * 0.42] = 1  # GM-like
    labels[r < min(shape[:2]) * 0.28] = 2  # WM-like
    labels[r < min(shape[:2]) * 0.12] = 3  # CSF-like
    labels[(abs(xx - cx) < 1.5) & (r < min(shape[:2]) * 0.36)] = 4  # vessel-like stripe
    return labels


def default_parameter_maps(labels: Any) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    labels = np.asarray(labels)
    cvr = np.zeros(labels.shape, dtype="float32")
    delay = np.zeros(labels.shape, dtype="float32")
    T = np.zeros(labels.shape, dtype="float32")
    params = {
        0: (0.0, 0.0, 20.0),
        1: (0.35, 18.0, 35.0),
        2: (0.18, 24.0, 45.0),
        3: (0.02, 10.0, 15.0),
        4: (0.9, 6.0, 10.0),
    }
    for label, values in params.items():
        keep = labels == label
        cvr[keep], delay[keep], T[keep] = values
    return {"GT_CVR": cvr, "GT_delay": delay, "GT_T": T, "GT_region_labels": labels}
