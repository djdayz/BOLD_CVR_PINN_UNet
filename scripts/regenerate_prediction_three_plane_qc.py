#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def _write_qc(
    path: Path,
    maps: dict[str, np.ndarray],
    pred_cvr: np.ndarray,
    pred_delay: np.ndarray,
    pred_T: np.ndarray,
    support: np.ndarray,
    affine: np.ndarray,
) -> None:
    def canonical(volume: np.ndarray) -> np.ndarray:
        image = nib.Nifti1Image(np.asarray(volume), np.asarray(affine))
        return np.asanyarray(nib.as_closest_canonical(image).dataobj)

    mask = canonical(np.asarray(support, dtype=np.uint8)).astype(bool)
    coordinates = np.argwhere(mask)
    centers = tuple(int(np.median(coordinates[:, axis])) for axis in range(3))
    rows = (
        ("CVR", canonical(maps["CVR"]), canonical(pred_cvr)),
        ("Delay", canonical(maps["delay"]), canonical(pred_delay)),
        ("T", canonical(maps["T"]), canonical(pred_T)),
    )
    planes = (
        ("Sagittal", lambda volume: volume[centers[0], :, :]),
        ("Coronal", lambda volume: volume[:, centers[1], :]),
        ("Axial", lambda volume: volume[:, :, centers[2]]),
    )
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.2), constrained_layout=True)
    for row_index, (parameter, gt, prediction) in enumerate(rows):
        values = np.concatenate([gt[mask], prediction[mask]])
        finite = values[np.isfinite(values)]
        vmin, vmax = (0.0, 1.0) if finite.size == 0 else tuple(np.percentile(finite, [1, 99]))
        if parameter == "CVR":
            vmin, vmax = 0.0, 1.0
        else:
            vmin = max(0.0, float(vmin))
        vmax = max(float(vmax), float(vmin) + 1e-6)
        last_image = None
        for plane_index, (plane_name, extract) in enumerate(planes):
            for version_index, (version, volume) in enumerate((("GT", gt), ("Predicted", prediction))):
                column = 2 * plane_index + version_index
                axis = axes[row_index, column]
                image = np.flipud(np.rot90(extract(volume), k=1))
                last_image = axis.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
                axis.set_title(f"{parameter} | {plane_name} {version}", fontsize=10)
                axis.axis("off")
        fig.colorbar(last_image, ax=axes[row_index, :], fraction=0.012, pad=0.006)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    metadata_paths = sorted(root.glob("{validation,test}/case_*/*/tcnr_*/prediction_metadata.json"))
    if not metadata_paths:
        metadata_paths = sorted(root.glob("validation/case_*/*/tcnr_*/prediction_metadata.json"))
        metadata_paths += sorted(root.glob("test/case_*/*/tcnr_*/prediction_metadata.json"))
    for index, metadata_path in enumerate(metadata_paths, start=1):
        condition = metadata_path.parent
        gt_cvr_img = nib.load(str(condition / "GT_CVR.nii.gz"))
        maps = {
            "CVR": gt_cvr_img.get_fdata(dtype=np.float32),
            "delay": nib.load(str(condition / "GT_delay.nii.gz")).get_fdata(dtype=np.float32),
            "T": nib.load(str(condition / "GT_T.nii.gz")).get_fdata(dtype=np.float32),
        }
        pred_cvr = nib.load(str(condition / "predicted_CVR.nii.gz")).get_fdata(dtype=np.float32)
        pred_delay = nib.load(str(condition / "predicted_delay.nii.gz")).get_fdata(dtype=np.float32)
        pred_T = nib.load(str(condition / "predicted_T.nii.gz")).get_fdata(dtype=np.float32)
        support = maps["CVR"] > 0
        _write_qc(
            condition / "gt_vs_predicted_three_plane_qc.png",
            maps,
            pred_cvr,
            pred_delay,
            pred_T,
            support,
            gt_cvr_img.affine,
        )
        if index % 50 == 0:
            print(f"regenerated={index}/{len(metadata_paths)}", flush=True)
    print(f"regenerated={len(metadata_paths)}", flush=True)


if __name__ == "__main__":
    main()
