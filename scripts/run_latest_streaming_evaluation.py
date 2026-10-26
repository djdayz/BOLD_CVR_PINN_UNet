#!/usr/bin/env python3
"""Evaluate all held-out conditions while keeping only metrics on limited storage."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from summarize_latest_full_evaluation import PARAMETERS, aggregate, load, map_metrics, write_csv


def read_manifest(path: Path) -> list[tuple[str, str, str, float]]:
    rows: list[tuple[str, str, str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            rows.append((row["split"], row["case_id"], row["paradigm"], float(row["tcnr"])))
    return rows


def read_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_write(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_csv(temporary, rows)
    temporary.replace(path)


def condition_rows(neural_dir: Path, hrf_dir: Path) -> list[dict[str, object]]:
    neural_meta = json.loads((neural_dir / "prediction_metadata.json").read_text())
    hrf_meta = json.loads((hrf_dir / "hrf_fit_metadata.json").read_text())
    split = str(neural_meta["split"])
    split = "validation" if split == "val" else split
    case_id = Path(neural_meta["case_dir"]).name
    paradigm = str(neural_meta["paradigm"])
    tcnr = float(neural_meta["target_tcnr"])
    gt_cvr = load(neural_dir / "GT_CVR.nii.gz")
    support = np.isfinite(gt_cvr) & (gt_cvr > 0)
    result: list[dict[str, object]] = []
    for method, source_dir, use_hrf in (
        ("self_supervised", neural_dir, False),
        ("exponential_hrf", hrf_dir, True),
    ):
        row: dict[str, object] = {
            "split": split,
            "case_id": case_id,
            "paradigm": paradigm,
            "tcnr": tcnr,
            "method": method,
            "simulation_generation_seconds": float(
                (neural_meta.get("timing_seconds") or {}).get("simulation_generation", np.nan)
            ),
            "parameter_fit_seconds": float(
                hrf_meta.get("hrf_parameter_fit_seconds", np.nan)
                if use_hrf
                else (neural_meta.get("timing_seconds") or {}).get("model_inference_only", np.nan)
            ),
        }
        for name, truth_name, neural_name, hrf_name in PARAMETERS:
            values = map_metrics(
                load(neural_dir / truth_name),
                load(source_dir / (hrf_name if use_hrf else neural_name)),
                support,
            )
            for metric, value in values.items():
                row[f"{name}_{metric}"] = value
        result.append(row)
    return result


def run(command: list[str], log) -> None:
    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    detail_path = args.out / "all_case_condition_metrics.csv"
    rows = read_existing(detail_path)
    complete = {
        (str(row["split"]), str(row["case_id"]), str(row["paradigm"]), float(row["tcnr"]))
        for row in rows
        if str(row["method"]) == "self_supervised"
    }
    conditions = read_manifest(args.manifest)
    log_path = args.out / "condition_commands.log"

    with log_path.open("a", encoding="utf-8") as log:
        for index, (split, case_id, paradigm, tcnr) in enumerate(conditions, start=1):
            key = (split, case_id, paradigm, tcnr)
            if key in complete:
                continue
            suffix = Path(split) / case_id / paradigm / f"tcnr_{tcnr:g}"
            neural_dir = args.work_root / "neural" / suffix
            hrf_dir = args.work_root / "hrf" / suffix
            if not (neural_dir / "prediction_metadata.json").exists():
                run(
                    [
                        sys.executable, "-m", "hybrid_cvr.cli", "predict-sim",
                        "--checkpoint", str(args.checkpoint), "--config", str(args.config),
                        "--sim-root", str(args.sim_root), "--out", str(neural_dir),
                        "--split", split, "--case-id", case_id,
                        "--paradigm", paradigm, "--tcnr", str(tcnr),
                    ],
                    log,
                )
            if not (hrf_dir / "hrf_fit_metadata.json").exists():
                run(
                    [
                        sys.executable, "scripts/evaluate_hrf_sim_condition.py",
                        "--checkpoint", str(args.checkpoint), "--config", str(args.config),
                        "--sim-root", str(args.sim_root), "--out", str(hrf_dir),
                        "--split", split, "--case-id", case_id,
                        "--paradigm", paradigm, "--tcnr", str(tcnr),
                    ],
                    log,
                )
            rows.extend(condition_rows(neural_dir, hrf_dir))
            atomic_write(detail_path, rows)
            shutil.rmtree(neural_dir)
            shutil.rmtree(hrf_dir)
            print(f"completed={len(rows) // 2}/{len(conditions)} condition={index}/{len(conditions)}", flush=True)

    write_csv(args.out / "condition_metrics.csv", aggregate(rows, ("split", "method", "paradigm", "tcnr")))
    write_csv(args.out / "case_metrics.csv", aggregate(rows, ("split", "method", "case_id")))
    write_csv(args.out / "overall_metrics.csv", aggregate(rows, ("split", "method")))
    print(f"evaluation_complete={len(rows) // 2}", flush=True)


if __name__ == "__main__":
    main()
