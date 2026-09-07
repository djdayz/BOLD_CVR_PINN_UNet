from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency
from hybrid_cvr.segmentation.tissue_masks import binary_erosion_numpy


def tissue_boundary_uncertainty(tissue_maps: dict[str, Any], brain_mask: Any | None = None) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    if not tissue_maps:
        raise ValueError("tissue_maps cannot be empty")
    stacked = np.stack([np.asarray(v, dtype=float) for v in tissue_maps.values()], axis=0)
    assignment_confidence = stacked.max(axis=0)
    hard = stacked.argmax(axis=0)
    boundary = np.zeros_like(assignment_confidence, dtype=bool)
    for label in range(stacked.shape[0]):
        region = hard == label
        edge = region ^ binary_erosion_numpy(region)
        boundary |= edge
    uncertainty = np.clip(1.25 * (1.0 - assignment_confidence) + 0.35 * boundary.astype(float), 0.0, 1.0)
    if brain_mask is not None:
        uncertainty *= np.asarray(brain_mask) > 0
    return uncertainty.astype("float32")
