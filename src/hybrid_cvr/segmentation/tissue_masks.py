from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def high_confidence_core(mask_or_prob: Any, threshold: float = 0.8, erosion_iters: int = 1) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    mask = np.asarray(mask_or_prob) > threshold
    if erosion_iters > 0:
        for _ in range(erosion_iters):
            mask = binary_erosion_numpy(mask)
    return mask.astype("uint8")


def make_brain_from_tissues(*tissues: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    if not tissues:
        raise ValueError("At least one tissue map is required")
    brain = np.zeros_like(np.asarray(tissues[0]), dtype=bool)
    for tissue in tissues:
        brain |= np.asarray(tissue) > 0
    return brain.astype("uint8")


def binary_erosion_numpy(mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(mask, dtype=bool)
    padded = np.pad(arr, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(arr, dtype=bool)
    for offsets in np.ndindex(*(3,) * arr.ndim):
        slices = tuple(slice(offset, offset + size) for offset, size in zip(offsets, arr.shape, strict=True))
        eroded &= padded[slices]
    return eroded


def binary_dilation_numpy(mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(mask, dtype=bool)
    padded = np.pad(arr, 1, mode="constant", constant_values=False)
    dilated = np.zeros_like(arr, dtype=bool)
    for offsets in np.ndindex(*(3,) * arr.ndim):
        slices = tuple(slice(offset, offset + size) for offset, size in zip(offsets, arr.shape, strict=True))
        dilated |= padded[slices]
    return dilated
