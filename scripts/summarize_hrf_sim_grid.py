#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


PARAMETERS = (
    ("cvr", "GT_CVR.nii.gz", "predicted_CVR.nii.gz", "hrf_CVR.nii.gz", (0.0, 1.0), "CVR"),
    ("delay", "GT_delay.nii.gz", "predicted_delay.nii.gz", "hrf_delay.nii.gz", (0.0, 80.0), "Delay (s)"),
    ("T", "GT_T.nii.gz", "predicted_T.nii.gz", "hrf_T.nii.gz", (0.0, 100.0), "T (s)"),
)


def load(path: Path) -> np.ndarray:
    return nib.as_closest_canonical(nib.load(str(path))).get_fdata(dtype=np.float32)


def metrics(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    t = truth[mask].astype(np.float64)
    p = prediction[mask].astype(np.float64)
    error = p - t
    return {
        "mae": float(np.mean(np.abs(error))),
        "relative_mae_percent": float(100 * np.mean(np.abs(error)) / max(abs(np.mean(t)), 1e-8)),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "pearson_r": float(np.corrcoef(t, p)[0, 1]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    metric_names = [f"{p}_{m}" for p, *_ in PARAMETERS for m in ("mae", "relative_mae_percent", "bias", "rmse", "pearson_r")]
    metric_names += ["hrf_parameter_fit_seconds", "simulation_generation_seconds"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows: groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for group_key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        out = {key: value for key, value in zip(keys, group_key)}; out["n_cases"] = len(group)
        for metric in metric_names:
            values = np.asarray([float(r[metric]) for r in group])
            out[f"{metric}_mean"] = float(np.nanmean(values))
            out[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        result.append(out)
    return result


def qc_png(path: Path, self_dir: Path, hrf_dir: Path) -> None:
    gt_cvr = load(self_dir / "GT_CVR.nii.gz")
    support = gt_cvr > 0
    z = int(np.argmax(np.sum(support, axis=(0, 1))))
    fig, axes = plt.subplots(3, 3, figsize=(8.2, 7.2), constrained_layout=True)
    for row, (name, gt_name, pred_name, hrf_name, limits, label) in enumerate(PARAMETERS):
        arrays = (load(self_dir / gt_name), load(self_dir / pred_name), load(hrf_dir / hrf_name))
        for col, (array, title) in enumerate(zip(arrays, ("Ground truth", "Self-supervised", "HRF/ODE"))):
            image = np.rot90(array[:, :, z])
            shown = axes[row, col].imshow(image, cmap="viridis", vmin=limits[0], vmax=limits[1], origin="lower")
            axes[row, col].axis("off")
            if row == 0: axes[row, col].set_title(title, fontsize=11, fontweight="bold")
        axes[row, 0].text(-0.08, 0.5, label, rotation=90, transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=11, fontweight="bold")
        fig.colorbar(shown, ax=axes[row, :], fraction=0.025, pad=0.01)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hrf_root", type=Path)
    parser.add_argument("self_root", type=Path)
    args = parser.parse_args()
    rows = []; first: set[tuple[str, float]] = set(); qc_dir = args.hrf_root / "comparison_qc"; qc_dir.mkdir(exist_ok=True)
    metadata_paths = sorted(args.hrf_root.glob("case_*/*/tcnr_*/hrf_fit_metadata.json"))
    if len(metadata_paths) != 315: raise RuntimeError(f"Expected 315 HRF fits, found {len(metadata_paths)}")
    for metadata_path in metadata_paths:
        hrf_dir = metadata_path.parent; meta = json.loads(metadata_path.read_text())
        case_id, paradigm, tcnr = meta["case_id"], meta["paradigm"], float(meta["target_tcnr"])
        self_dir = args.self_root / "test" / case_id / paradigm / f"tcnr_{tcnr}"
        gt_cvr = load(self_dir / "GT_CVR.nii.gz"); support = np.isfinite(gt_cvr) & (gt_cvr > 0)
        row: dict[str, object] = {"case_id": case_id, "paradigm": paradigm, "tcnr": tcnr,
            "simulation_generation_seconds": meta["simulation_generation_seconds"],
            "hrf_parameter_fit_seconds": meta["hrf_parameter_fit_seconds"]}
        for name, gt_name, _, hrf_name, _, _ in PARAMETERS:
            truth, prediction = load(self_dir / gt_name), load(hrf_dir / hrf_name)
            valid = support & np.isfinite(truth) & np.isfinite(prediction)
            for metric, value in metrics(truth, prediction, valid).items(): row[f"{name}_{metric}"] = value
        rows.append(row)
        key = (paradigm, tcnr)
        if key not in first:
            first.add(key); qc_png(qc_dir / f"{paradigm}_tcnr_{tcnr}_GT_vs_self_supervised_vs_HRF.png", self_dir, hrf_dir)
    write_csv(args.hrf_root / "test_case_metrics.csv", rows)
    write_csv(args.hrf_root / "test_condition_metrics.csv", aggregate(rows, ("paradigm", "tcnr")))
    write_csv(args.hrf_root / "test_overall_metrics.csv", aggregate(rows, tuple()))
    print(f"Summarized {len(rows)} HRF fits and wrote {len(first)} comparison figures")


if __name__ == "__main__": main()
