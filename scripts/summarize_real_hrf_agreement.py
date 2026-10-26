#!/usr/bin/env python3
"""Measure latest-model agreement with exponential-HRF maps on real sessions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


PARAMETERS = (
    ("cvr", "predicted_CVR.nii.gz", "hrf_cvr.nii.gz"),
    ("delay", "predicted_delay.nii.gz", "hrf_delay.nii.gz"),
    ("T", "predicted_T.nii.gz", "hrf_T.nii.gz"),
)


def load(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--real-cvr-root", type=Path, default=Path("data/derivatives/real_cvr"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for prediction_path in sorted(args.inference_root.glob("sub-*/ses-*/predicted_CVR.nii.gz")):
        session_dir = prediction_path.parent
        subject, session = session_dir.parts[-2:]
        hrf_dir = args.real_cvr_root / subject / session
        if not all((hrf_dir / hrf_name).exists() for _, _, hrf_name in PARAMETERS):
            continue
        row: dict[str, object] = {"subject": subject, "session": session}
        for name, prediction_name, hrf_name in PARAMETERS:
            prediction = load(session_dir / prediction_name)
            hrf = load(hrf_dir / hrf_name)
            if prediction.shape != hrf.shape:
                raise ValueError(f"Shape mismatch for {subject}/{session}/{name}: {prediction.shape} != {hrf.shape}")
            mask = np.isfinite(prediction) & np.isfinite(hrf) & (prediction != 0) & (hrf != 0)
            x = prediction[mask].astype(np.float64)
            y = hrf[mask].astype(np.float64)
            if x.size < 2:
                raise ValueError(f"Insufficient common voxels for {subject}/{session}/{name}")
            error = x - y
            row[f"{name}_n_voxels"] = int(x.size)
            row[f"{name}_neural_mean"] = float(np.mean(x))
            row[f"{name}_hrf_mean"] = float(np.mean(y))
            row[f"{name}_mae_vs_hrf"] = float(np.mean(np.abs(error)))
            row[f"{name}_bias_vs_hrf"] = float(np.mean(error))
            row[f"{name}_rmse_vs_hrf"] = float(np.sqrt(np.mean(error * error)))
            row[f"{name}_pearson_r_vs_hrf"] = float(np.corrcoef(x, y)[0, 1])
        rows.append(row)

    if not rows:
        raise RuntimeError("No matched real inference and exponential-HRF sessions found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote agreement metrics for {len(rows)} real sessions to {args.out}")


if __name__ == "__main__":
    main()
