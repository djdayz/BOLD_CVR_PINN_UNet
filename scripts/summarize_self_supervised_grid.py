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
    ("cvr", "GT_CVR.nii.gz", "predicted_CVR.nii.gz"),
    ("delay", "GT_delay.nii.gz", "predicted_delay.nii.gz"),
    ("T", "GT_T.nii.gz", "predicted_T.nii.gz"),
)
PARADIGMS = ("block", "multi_step", "pseudo_random_binary")
TCNRS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _case_metrics(condition_dir: Path, split: str) -> dict[str, object]:
    metadata = json.loads((condition_dir / "prediction_metadata.json").read_text())
    gt_cvr = nib.load(str(condition_dir / "GT_CVR.nii.gz")).get_fdata(dtype=np.float32)
    support = np.isfinite(gt_cvr) & (gt_cvr > 0)
    timing = metadata.get("timing_seconds", {})
    row: dict[str, object] = {
        "split": split,
        "case_id": Path(metadata["case_dir"]).name,
        "paradigm": metadata["paradigm"],
        "tcnr": float(metadata["target_tcnr"]),
        "checkpoint": metadata["checkpoint"],
        "simulation_generation_seconds": timing.get("simulation_generation", np.nan),
        "model_inference_seconds": timing.get("model_inference_only", np.nan),
    }
    for name, gt_name, prediction_name in PARAMETERS:
        truth = nib.load(str(condition_dir / gt_name)).get_fdata(dtype=np.float32)
        prediction = nib.load(str(condition_dir / prediction_name)).get_fdata(dtype=np.float32)
        mask = support & np.isfinite(truth) & np.isfinite(prediction)
        truth_values = truth[mask].astype(np.float64)
        predicted_values = prediction[mask].astype(np.float64)
        error = predicted_values - truth_values
        gt_mean = float(np.mean(truth_values))
        row[f"gt_{name}_mean"] = gt_mean
        row[f"pred_{name}_mean"] = float(np.mean(predicted_values))
        row[f"{name}_mae"] = float(np.mean(np.abs(error)))
        row[f"{name}_relative_mae_percent"] = float(
            100.0 * np.mean(np.abs(error)) / max(abs(gt_mean), 1e-8)
        )
        row[f"{name}_bias"] = float(np.mean(error))
        row[f"{name}_rmse"] = float(np.sqrt(np.mean(error * error)))
        row[f"{name}_pearson_r"] = float(np.corrcoef(truth_values, predicted_values)[0, 1])
    return row


def _aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    metric_names = [
        f"{parameter}_{metric}"
        for parameter, _, _ in PARAMETERS
        for metric in ("mae", "relative_mae_percent", "bias", "rmse", "pearson_r")
    ] + ["simulation_generation_seconds", "model_inference_seconds"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    result = []
    for group_key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        summary = {key: value for key, value in zip(keys, group_key)}
        summary["n_cases"] = len(group)
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        result.append(summary)
    return result


def _plot_etco2(root: Path, split: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for axis, paradigm in zip(axes, PARADIGMS):
        traces = sorted((root / split).glob(f"case_*/{paradigm}/tcnr_2.0/etco2_trace.csv"))
        if not traces:
            raise FileNotFoundError(f"No ETCO2 trace found for {split}/{paradigm}")
        trace = np.genfromtxt(traces[0], delimiter=",", names=True)
        axis.plot(trace["time_seconds"], trace["etco2_clean_mmhg"], label="Clean", linewidth=2.0)
        axis.plot(
            trace["time_seconds"],
            trace["etco2_model_input_mmhg"],
            label="Model input",
            linewidth=1.2,
            alpha=0.85,
        )
        axis.set_title(paradigm.replace("_", " ").title())
        axis.set_ylabel("ETCO2 (mmHg)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(root / f"{split}_etco2_paradigms_used.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    rows = []
    for split in ("validation", "test"):
        metadata_paths = sorted((root / split).glob("case_*/*/tcnr_*/prediction_metadata.json"))
        if len(metadata_paths) != 315:
            raise RuntimeError(f"Expected 315 {split} exports, found {len(metadata_paths)}")
        rows.extend(_case_metrics(path.parent, split) for path in metadata_paths)
        _plot_etco2(root, split)
    _write_csv(root / "validation_test_case_metrics.csv", rows)
    _write_csv(root / "validation_test_condition_metrics.csv", _aggregate(rows, ("split", "paradigm", "tcnr")))
    _write_csv(root / "validation_test_overall_metrics.csv", _aggregate(rows, ("split",)))
    print(f"Summarized {len(rows)} exports under {root}")


if __name__ == "__main__":
    main()
