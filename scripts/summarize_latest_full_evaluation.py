#!/usr/bin/env python3
"""Compare latest-model and exponential-HRF maps across all held-out cases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


PARAMETERS = (
    ("cvr", "GT_CVR.nii.gz", "predicted_CVR.nii.gz", "hrf_CVR.nii.gz"),
    ("delay", "GT_delay.nii.gz", "predicted_delay.nii.gz", "hrf_delay.nii.gz"),
    ("T", "GT_T.nii.gz", "predicted_T.nii.gz", "hrf_T.nii.gz"),
)
METRICS = ("mae", "relative_mae_percent", "bias", "rmse", "pearson_r")


def load(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def map_metrics(truth: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(truth) & np.isfinite(prediction)
    reference = truth[valid].astype(np.float64)
    estimate = prediction[valid].astype(np.float64)
    if reference.size < 2:
        raise ValueError("Fewer than two valid evaluation voxels")
    error = estimate - reference
    gt_mean = float(np.mean(reference))
    pred_mean = float(np.mean(estimate))
    reference_std = float(np.std(reference))
    estimate_std = float(np.std(estimate))
    correlation = (
        float(np.corrcoef(reference, estimate)[0, 1])
        if reference_std > 0 and estimate_std > 0
        else float("nan")
    )
    mae = float(np.mean(np.abs(error)))
    return {
        "gt_mean": gt_mean,
        "pred_mean": pred_mean,
        "mae": mae,
        "relative_mae_percent": 100.0 * mae / max(abs(gt_mean), 1e-8),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "pearson_r": correlation,
    }


def aggregate(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    metric_names = [f"{name}_{metric}" for name, *_ in PARAMETERS for metric in METRICS]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summaries: list[dict[str, object]] = []
    for group_key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        summary: dict[str, object] = dict(zip(keys, group_key))
        summary["n"] = len(group)
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            summary[metric] = float(np.nanmean(values))
            summary[f"{metric}_std"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            )
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-root", type=Path, required=True)
    parser.add_argument("--hrf-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=630)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    metadata_paths = sorted(args.neural_root.glob("*/*/*/tcnr_*/prediction_metadata.json"))
    if not args.allow_incomplete and len(metadata_paths) != args.expected:
        raise RuntimeError(f"Expected {args.expected} neural outputs, found {len(metadata_paths)}")

    for metadata_path in metadata_paths:
        neural_dir = metadata_path.parent
        relative = neural_dir.relative_to(args.neural_root)
        split, case_id, paradigm, tcnr_dir = relative.parts
        hrf_dir = args.hrf_root / relative
        hrf_metadata = hrf_dir / "hrf_fit_metadata.json"
        if not hrf_metadata.exists():
            if args.allow_incomplete:
                continue
            raise FileNotFoundError(hrf_metadata)
        neural_metadata = json.loads(metadata_path.read_text())
        hrf_meta = json.loads(hrf_metadata.read_text())
        gt_cvr = load(neural_dir / "GT_CVR.nii.gz")
        support = np.isfinite(gt_cvr) & (gt_cvr > 0)
        for method, source_dir, source_column in (
            ("self_supervised", neural_dir, 2),
            ("exponential_hrf", hrf_dir, 3),
        ):
            row: dict[str, object] = {
                "split": split,
                "case_id": case_id,
                "paradigm": paradigm,
                "tcnr": float(tcnr_dir.removeprefix("tcnr_")),
                "method": method,
                "simulation_generation_seconds": float(
                    (neural_metadata.get("timing_seconds") or {}).get(
                        "simulation_generation", np.nan
                    )
                ),
                "parameter_fit_seconds": float(
                    (neural_metadata.get("timing_seconds") or {}).get(
                        "model_inference_only", np.nan
                    )
                    if method == "self_supervised"
                    else hrf_meta.get("hrf_parameter_fit_seconds", np.nan)
                ),
            }
            for name, truth_name, neural_name, hrf_name in PARAMETERS:
                values = map_metrics(
                    load(neural_dir / truth_name),
                    load(source_dir / (neural_name if source_column == 2 else hrf_name)),
                    support,
                )
                for metric, value in values.items():
                    row[f"{name}_{metric}"] = value
            rows.append(row)

    if not args.allow_incomplete and len(rows) != 2 * args.expected:
        raise RuntimeError(f"Expected {2 * args.expected} method rows, found {len(rows)}")
    write_csv(args.out / "all_case_condition_metrics.csv", rows)
    write_csv(args.out / "condition_metrics.csv", aggregate(rows, ("split", "method", "paradigm", "tcnr")))
    write_csv(args.out / "case_metrics.csv", aggregate(rows, ("split", "method", "case_id")))
    write_csv(args.out / "overall_metrics.csv", aggregate(rows, ("split", "method")))
    print(f"Summarized {len(rows) // 2} matched conditions ({len(rows)} method rows)")


if __name__ == "__main__":
    main()
