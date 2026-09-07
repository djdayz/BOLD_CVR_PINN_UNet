from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def stack_features(feature_maps: dict[str, Any], names: list[str]) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    missing = [name for name in names if name not in feature_maps]
    if missing:
        raise KeyError(f"Missing requested feature maps: {missing}")
    return np.stack([feature_maps[name] for name in names], axis=0).astype("float32")
