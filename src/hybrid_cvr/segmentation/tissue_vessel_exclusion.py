from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency


TISSUE_MASK_NAMES = ("cortical_gm", "subcortical_gm", "wm", "vcsf")
TISSUE_LABELS = {
    "cortical_gm": 1,
    "subcortical_gm": 2,
    "wm": 3,
    "vcsf": 4,
    "vessel": 5,
}


def save_vessel_excluded_tissue_masks(
    segmentation_dir: str | Path,
    vessel_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Path | int | str]:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    segmentation_dir = Path(segmentation_dir)
    vessel_dir = Path(vessel_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vessel_path = vessel_dir / "vessel_mask_high_confidence.nii.gz"
    if not vessel_path.exists():
        raise FileNotFoundError(vessel_path)
    vessel_img = nib.load(str(vessel_path))
    vessel = vessel_img.get_fdata() > 0
    likelihood_path = vessel_dir / "vessel_likelihood.nii.gz"
    if not likelihood_path.exists():
        raise FileNotFoundError(likelihood_path)
    likelihood = _load_like(likelihood_path, vessel_img)
    occupied = vessel.copy()
    label_map = np.zeros(vessel.shape, dtype=np.uint8)
    label_map[vessel] = TISSUE_LABELS["vessel"]

    outputs: dict[str, Path | int | str] = {
        "vessel_likelihood": _save_map(likelihood, vessel_img, out_dir / "vessel_likelihood.nii.gz"),
        "vessel_mask_high_confidence": _save_mask(vessel, vessel_img, out_dir / "vessel_mask_high_confidence.nii.gz"),
        "n_vessel_voxels": int(np.sum(vessel)),
    }

    for name in TISSUE_MASK_NAMES:
        mask_path = _find_tissue_mask(segmentation_dir, name)
        if mask_path is None:
            raise FileNotFoundError(f"Missing {name}_mask.nii.gz under {segmentation_dir}")
        data = _load_like(mask_path, vessel_img) > 0
        excluded = data & ~vessel
        exclusive = excluded & ~occupied
        occupied |= exclusive
        label_map[exclusive] = TISSUE_LABELS[name]
        outputs[f"{name}_mask_no_vessel"] = _save_mask(
            excluded, vessel_img, out_dir / f"{name}_mask_no_vessel.nii.gz"
        )
        outputs[f"{name}_mask_exclusive_no_vessel"] = _save_mask(
            exclusive, vessel_img, out_dir / f"{name}_mask_exclusive_no_vessel.nii.gz"
        )
        outputs[f"n_{name}_voxels_before"] = int(np.sum(data))
        outputs[f"n_{name}_voxels_no_vessel"] = int(np.sum(excluded))
        outputs[f"n_{name}_voxels_exclusive_no_vessel"] = int(np.sum(exclusive))
        outputs[f"n_{name}_vessel_overlap"] = int(np.sum(data & vessel))

    outputs["tissue_label_map_no_vessel"] = _save_label_map(
        label_map, vessel_img, out_dir / "tissue_label_map_no_vessel.nii.gz"
    )
    write_tissue_exclusion_metadata(out_dir / "tissue_vessel_exclusion_metadata.csv", outputs)
    return outputs


def write_tissue_exclusion_metadata(path: Path, outputs: dict[str, Path | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in outputs.items():
            writer.writerow([key, value])


def _find_tissue_mask(segmentation_dir: Path, name: str) -> Path | None:
    candidates = [
        segmentation_dir / "bold" / f"{name}_mask.nii.gz",
        segmentation_dir / f"{name}_mask.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_like(path: str | Path, reference_img: Any) -> Any:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    img = nib.load(str(path))
    if img.shape[:3] != reference_img.shape[:3] or not np.allclose(img.affine, reference_img.affine):
        raise ValueError(f"Image grid mismatch for {path}")
    return img.get_fdata(dtype=np.float32)


def _save_mask(data: Any, reference_img: Any, out_path: Path) -> Path:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image((np.asarray(data) > 0).astype(np.uint8), reference_img.affine, reference_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))
    return out_path


def _save_map(data: Any, reference_img: Any, out_path: Path) -> Path:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference_img.affine, reference_img.header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(out_path))
    return out_path


def _save_label_map(data: Any, reference_img: Any, out_path: Path) -> Path:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.uint8), reference_img.affine, reference_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))
    return out_path
