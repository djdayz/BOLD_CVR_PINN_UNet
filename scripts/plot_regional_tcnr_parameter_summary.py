from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


REGIONS = {
    "CGM": {"label": 1, "name": "CGM"},
    "SGM": {"label": 2, "name": "SGM"},
    "WM": {"label": 3, "name": "WM"},
}

PARAMETERS = {
    "CVR": {
        "pred": "predicted_CVR.nii.gz",
        "gt": "GT_CVR.nii.gz",
        "label": "CVR magnitude",
        "unit": "%BOLD/mmHg",
    },
    "delay": {
        "pred": "predicted_delay.nii.gz",
        "gt": "GT_delay.nii.gz",
        "label": "CVR delay",
        "unit": "s",
    },
    "T": {
        "pred": "predicted_T.nii.gz",
        "gt": "GT_T.nii.gz",
        "label": "T",
        "unit": "s",
    },
}


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot regional predicted CVR/delay/T means versus tCNR with a GT-referenced "
            "relative-error secondary y-axis."
        )
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        required=True,
        help="Root containing case_*/paradigm/tcnr_*/prediction_metadata.json exports.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("data/simulated/mida_parameters"),
        help="Root containing case_*/GT_CVR.nii.gz, GT_delay.nii.gz, GT_T.nii.gz, and GT_region_labels.nii.gz.",
    )
    parser.add_argument("--split-name", default="test", help="Name used in output filenames/titles.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--real-cvr-root",
        type=Path,
        default=Path("data/derivatives/real_cvr"),
        help="Root containing real sub-*/ses-*/tCNR.nii.gz maps.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("dominant_label", "nonzero_prediction"),
        default="dominant_label",
        help="Region mask source. dominant_label uses GT_region_labels values 1/2/3.",
    )
    return parser.parse_args()


def real_tcnr_interval(real_cvr_root: Path) -> tuple[float, float, float]:
    session_medians: list[float] = []
    for path in sorted(real_cvr_root.glob("sub-*/ses-*/tCNR.nii.gz")):
        values = load_nifti(path)
        mask_path = path.parent / "valid_fit_mask.nii.gz"
        mask = load_nifti(mask_path) > 0 if mask_path.exists() else values > 0
        valid = values[mask & np.isfinite(values) & (values > 0)]
        if valid.size:
            session_medians.append(float(np.median(valid)))
    if not session_medians:
        raise FileNotFoundError(f"No valid real tCNR maps found under {real_cvr_root}")
    low, center, high = np.percentile(session_medians, [10, 50, 90])
    return float(low), float(center), float(high)


def case_ground_truth(case_id: str, gt_root: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    case_dir = gt_root / case_id
    labels = load_nifti(case_dir / "GT_region_labels.nii.gz").astype(np.int16)
    gt_maps = {name: load_nifti(case_dir / spec["gt"]) for name, spec in PARAMETERS.items()}
    return labels, gt_maps


def collect_records(
    prediction_root: Path, gt_root: Path, mask_mode: str
) -> tuple[dict[tuple[str, str, float], list[dict[str, float]]], dict[tuple[str, str], list[float]]]:
    records: dict[tuple[str, str, float], list[dict[str, float]]] = defaultdict(list)
    gt_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    gt_cache: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}

    metadata_paths = sorted(prediction_root.glob("case_*/*/tcnr_*/prediction_metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No prediction metadata files found under {prediction_root}")

    for metadata_path in metadata_paths:
        export_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        case_id = Path(metadata["case_dir"]).name
        tcnr = float(metadata["target_tcnr"])

        if case_id not in gt_cache:
            gt_cache[case_id] = case_ground_truth(case_id, gt_root)
        labels, gt_maps = gt_cache[case_id]

        pred_maps = {
            name: load_nifti(export_dir / spec["pred"]) for name, spec in PARAMETERS.items()
        }
        nonzero_eval = np.isfinite(pred_maps["T"]) & (pred_maps["T"] > 0)

        for region, region_spec in REGIONS.items():
            if mask_mode == "dominant_label":
                region_mask = labels == int(region_spec["label"])
            else:
                region_mask = nonzero_eval
            region_mask = region_mask & nonzero_eval

            if not np.any(region_mask):
                continue

            for parameter_name in PARAMETERS:
                pred = pred_maps[parameter_name]
                gt = gt_maps[parameter_name]
                valid = region_mask & np.isfinite(pred) & np.isfinite(gt)
                if not np.any(valid):
                    continue

                pred_region = pred[valid]
                gt_region = gt[valid]
                pred_mean = float(np.mean(pred_region))
                gt_mean = float(np.mean(gt_region))
                rel_error = (pred_mean - gt_mean) / max(abs(gt_mean), 1e-6) * 100.0
                records[(region, parameter_name, tcnr)].append(
                    {
                        "pred_mean": pred_mean,
                        "relative_error_pct": float(rel_error),
                        "gt_mean": gt_mean,
                        "gt_median": float(np.median(gt_region)),
                    }
                )
                gt_values[(region, parameter_name)].extend(float(v) for v in gt_region)

    return records, gt_values


def summarize(
    records: dict[tuple[str, str, float], list[dict[str, float]]],
    gt_values: dict[tuple[str, str], list[float]],
    csv_path: Path,
) -> dict[tuple[str, str], dict[str, np.ndarray | float]]:
    summary: dict[tuple[str, str], dict[str, np.ndarray | float]] = {}
    rows: list[dict[str, str | float | int]] = []

    for region in REGIONS:
        for parameter in PARAMETERS:
            tcnrs = sorted(k[2] for k in records if k[0] == region and k[1] == parameter)
            pred_mean = []
            pred_sd = []
            rel_mean = []
            rel_sd = []
            n = []
            for tcnr in tcnrs:
                vals = records[(region, parameter, tcnr)]
                pred_vals = np.array([v["pred_mean"] for v in vals], dtype=np.float64)
                rel_vals = np.array([v["relative_error_pct"] for v in vals], dtype=np.float64)
                pred_mean.append(float(np.mean(pred_vals)))
                pred_sd.append(float(np.std(pred_vals, ddof=1)) if pred_vals.size > 1 else 0.0)
                rel_mean.append(float(np.mean(rel_vals)))
                rel_sd.append(float(np.std(rel_vals, ddof=1)) if rel_vals.size > 1 else 0.0)
                n.append(int(pred_vals.size))

            gt_array = np.array(gt_values[(region, parameter)], dtype=np.float64)
            gt_mean = float(np.mean(gt_array))
            gt_median = float(np.median(gt_array))
            summary[(region, parameter)] = {
                "tcnr": np.array(tcnrs, dtype=np.float64),
                "pred_mean": np.array(pred_mean, dtype=np.float64),
                "pred_sd": np.array(pred_sd, dtype=np.float64),
                "rel_mean": np.array(rel_mean, dtype=np.float64),
                "rel_sd": np.array(rel_sd, dtype=np.float64),
                "n": np.array(n, dtype=np.int32),
                "gt_mean": gt_mean,
                "gt_median": gt_median,
            }
            for i, tcnr in enumerate(tcnrs):
                rows.append(
                    {
                        "region": region,
                        "parameter": parameter,
                        "tcnr": tcnr,
                        "n": n[i],
                        "predicted_mean": pred_mean[i],
                        "predicted_sd": pred_sd[i],
                        "relative_error_pct_mean": rel_mean[i],
                        "relative_error_pct_sd": rel_sd[i],
                        "GT_mean": gt_mean,
                        "GT_median": gt_median,
                    }
                )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "region",
                "parameter",
                "tcnr",
                "n",
                "predicted_mean",
                "predicted_sd",
                "relative_error_pct_mean",
                "relative_error_pct_sd",
                "GT_mean",
                "GT_median",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary


def plot_grid(
    summary: dict[tuple[str, str], dict[str, np.ndarray | float]],
    split_name: str,
    out_png: Path,
    out_pdf: Path,
    real_tcnr: tuple[float, float, float],
) -> None:
    def secondary_axis_functions(gt_mean: float, scale: float):
        def to_relative_error(y: float | np.ndarray) -> float | np.ndarray:
            return (np.asarray(y) - gt_mean) / scale * 100.0

        def from_relative_error(e: float | np.ndarray) -> float | np.ndarray:
            return gt_mean + np.asarray(e) / 100.0 * scale

        return to_relative_error, from_relative_error

    fig, axes = plt.subplots(3, 3, figsize=(15.0, 11.0), sharex=True, constrained_layout=True)
    real_low, _, real_high = real_tcnr

    for row, (parameter, parameter_spec) in enumerate(PARAMETERS.items()):
        for col, (region, region_spec) in enumerate(REGIONS.items()):
            ax = axes[row, col]
            item = summary[(region, parameter)]
            tcnr = np.asarray(item["tcnr"], dtype=np.float64)
            pred_mean = np.asarray(item["pred_mean"], dtype=np.float64)
            pred_sd = np.asarray(item["pred_sd"], dtype=np.float64)
            gt_mean = float(item["gt_mean"])
            gt_median = float(item["gt_median"])
            scale = max(abs(gt_mean), 1e-6)

            ax.axvspan(
                real_low,
                real_high,
                color="#8ecae6",
                alpha=0.28,
                linewidth=0,
                label="Real BOLD tCNR range" if row == 0 and col == 0 else None,
                zorder=0,
            )

            ax.errorbar(
                tcnr,
                pred_mean,
                yerr=pred_sd,
                marker="o",
                markersize=9,
                linewidth=1.8,
                elinewidth=2.0,
                capsize=5,
                capthick=2.0,
                color="#1f77b4",
                label="Predicted mean +/- SD",
                zorder=3,
            )
            ax.axhline(gt_mean, color="black", linestyle="-", linewidth=1.5, label="GT mean")
            ax.axhline(gt_median, color="black", linestyle="--", linewidth=1.5, label="GT median")
            ax.set_xscale("log")
            ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
            ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
            ax.grid(True, which="both", alpha=0.25, zorder=0)
            if row == 0:
                ax.set_title(region_spec["name"])

            if col == 0:
                ax.set_ylabel(f"{parameter_spec['label']} ({parameter_spec['unit']})")
            if row == 2:
                ax.set_xlabel("tCNR")

            secax = ax.secondary_yaxis(
                "right", functions=secondary_axis_functions(gt_mean, scale)
            )
            secax.set_ylabel("Relative error (%)")

            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)

    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records, gt_values = collect_records(args.prediction_root, args.gt_root, args.mask_mode)
    csv_path = args.out_dir / f"{args.split_name}_regional_3x3_parameter_vs_tcnr_summary.csv"
    summary = summarize(records, gt_values, csv_path)
    out_png = args.out_dir / f"{args.split_name}_regional_3x3_parameter_vs_tcnr.png"
    out_pdf = args.out_dir / f"{args.split_name}_regional_3x3_parameter_vs_tcnr.pdf"
    real_tcnr = real_tcnr_interval(args.real_cvr_root)
    plot_grid(summary, args.split_name, out_png, out_pdf, real_tcnr)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    print(f"Saved {csv_path}")
    print(
        "Real BOLD tCNR band (10th-90th percentile of session medians): "
        f"{real_tcnr[0]:.3f}-{real_tcnr[2]:.3f}; median={real_tcnr[1]:.3f}"
    )


if __name__ == "__main__":
    main()
