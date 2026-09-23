#!/usr/bin/env python3
"""Summarize one validation and one test case across the complete condition grid."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

PARAMETERS = (
    ("cvr", "GT_CVR.nii.gz", "predicted_CVR.nii.gz", "hrf_CVR.nii.gz", (0, 1), "CVR"),
    ("delay", "GT_delay.nii.gz", "predicted_delay.nii.gz", "hrf_delay.nii.gz", (0, 80), "Delay (s)"),
    ("T", "GT_T.nii.gz", "predicted_T.nii.gz", "hrf_T.nii.gz", (0, 100), "T (s)"),
)


def load(path: Path) -> np.ndarray:
    return nib.as_closest_canonical(nib.load(str(path))).get_fdata(dtype=np.float32)


def metric_values(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    truth = truth[mask].astype(np.float64); prediction = prediction[mask].astype(np.float64)
    error = prediction - truth
    return {
        "mae": float(np.mean(np.abs(error))),
        "relative_mae_percent": float(100 * np.mean(np.abs(error)) / max(abs(np.mean(truth)), 1e-8)),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "pearson_r": float(np.corrcoef(truth, prediction)[0, 1]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)


def qc(path: Path, neural: Path, hrf: Path) -> None:
    support = load(neural / "GT_CVR.nii.gz") > 0
    z = int(np.argmax(support.sum(axis=(0, 1))))
    fig, axes = plt.subplots(3, 3, figsize=(8.3, 7.2), constrained_layout=True)
    for row, (_, gt_name, pred_name, hrf_name, limits, label) in enumerate(PARAMETERS):
        arrays = (load(neural / gt_name), load(neural / pred_name), load(hrf / hrf_name))
        for col, (array, title) in enumerate(zip(arrays, ("Ground truth", "Self-supervised", "Exponential HRF"), strict=True)):
            shown_mask = np.flipud(np.rot90(support[:, :, z]))
            shown = np.ma.masked_where(~shown_mask, np.flipud(np.rot90(array[:, :, z])))
            image = axes[row, col].imshow(shown, cmap="viridis", vmin=limits[0], vmax=limits[1])
            axes[row, col].axis("off")
            if row == 0: axes[row, col].set_title(title, fontsize=10, fontweight="bold")
        axes[row, 0].text(-0.07, 0.5, label, rotation=90, transform=axes[row, 0].transAxes,
                          ha="right", va="center", fontsize=10, fontweight="bold")
        fig.colorbar(image, ax=axes[row, :], fraction=0.025, pad=0.01)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-root", type=Path, required=True)
    parser.add_argument("--hrf-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison_qc").mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    for metadata_path in sorted(args.neural_root.glob("*/*/*/tcnr_*/prediction_metadata.json")):
        neural = metadata_path.parent; metadata = json.loads(metadata_path.read_text())
        split = metadata_path.relative_to(args.neural_root).parts[0]
        case_id = Path(metadata["case_dir"]).name; paradigm = metadata["paradigm"]
        tcnr = float(metadata["target_tcnr"]); hrf = args.hrf_root / split / case_id / paradigm / f"tcnr_{tcnr}"
        if not (hrf / "hrf_fit_metadata.json").exists():
            raise FileNotFoundError(hrf / "hrf_fit_metadata.json")
        support = load(neural / "GT_CVR.nii.gz") > 0
        for method, source_index in (("self_supervised", 2), ("exponential_hrf", 3)):
            row: dict[str, object] = {"split": split, "case_id": case_id, "paradigm": paradigm,
                                     "tcnr": tcnr, "method": method}
            for name, gt_name, pred_name, hrf_name, _, _ in PARAMETERS:
                truth = load(neural / gt_name); prediction = load(neural / (pred_name if source_index == 2 else gt_name))
                if source_index == 3: prediction = load(hrf / hrf_name)
                valid = support & np.isfinite(truth) & np.isfinite(prediction)
                for metric, value in metric_values(truth, prediction, valid).items(): row[f"{name}_{metric}"] = value
            rows.append(row)
        qc(args.out / "comparison_qc" / f"{split}_{paradigm}_tcnr_{tcnr}.png", neural, hrf)
    if len(rows) != 84:
        raise RuntimeError(f"Expected 84 method rows (42 conditions x 2), found {len(rows)}")
    write_csv(args.out / "case_condition_metrics.csv", rows)
    aggregate = []
    for keys in (("split", "method"), ("split", "paradigm", "tcnr", "method")):
        groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in rows: groups.setdefault(tuple(row[key] for key in keys), []).append(row)
        for values, group in groups.items():
            summary = {key: value for key, value in zip(keys, values, strict=True)}; summary["n"] = len(group)
            for column in rows[0]:
                if column in {"split", "case_id", "paradigm", "tcnr", "method"}: continue
                data = np.asarray([float(item[column]) for item in group]); summary[column] = float(np.nanmean(data))
            aggregate.append(summary)
    write_csv(args.out / "aggregate_metrics.csv", aggregate)
    print(f"Wrote {len(rows)} metric rows and 42 comparison figures to {args.out}")


if __name__ == "__main__":
    main()
