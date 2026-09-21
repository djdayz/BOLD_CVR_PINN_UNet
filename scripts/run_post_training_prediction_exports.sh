#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-data/models/unet_pinn}"
TRAINING_PID="${2:-}"
CONFIG="${3:-configs/train_unet_pinn.yaml}"
SIM_ROOT="${4:-data/simulated}"
SPLIT_JSON="${5:-data/simulated/splits/train_val_test_split.json}"
MAX_PARALLEL="${6:-4}"

LOG_DIR="${MODEL_DIR}/logs"
EXPORT_ROOT="${MODEL_DIR}/predicted_maps"
VAL_EXPORT_ROOT="${MODEL_DIR}/predicted_maps_validation"
WATCH_LOG="${LOG_DIR}/post_training_prediction_exports.log"

mkdir -p "${LOG_DIR}" "${EXPORT_ROOT}" "${VAL_EXPORT_ROOT}"

{
  echo "post_training_export_started=$(date -Is)"
  echo "model_dir=${MODEL_DIR}"
  echo "training_pid=${TRAINING_PID}"
  echo "config=${CONFIG}"
  echo "sim_root=${SIM_ROOT}"
  echo "split_json=${SPLIT_JSON}"

  if [[ -n "${TRAINING_PID}" ]] && kill -0 "${TRAINING_PID}" 2>/dev/null; then
    echo "waiting_for_training_pid=${TRAINING_PID}"
    while kill -0 "${TRAINING_PID}" 2>/dev/null; do
      sleep 60
    done
  else
    echo "training_pid_not_running_at_watcher_start"
  fi

  echo "training_pid_finished=$(date -Is)"

  if [[ -f "${LOG_DIR}/training_stdout.log" ]]; then
    if grep -q "Training complete:" "${LOG_DIR}/training_stdout.log"; then
      echo "training_status=complete"
    else
      echo "training_status=no_completion_line_found_exporting_best_available_checkpoint"
    fi
  fi

  PYTHONPATH=src python -m hybrid_cvr.cli summarize-training --model-dir "${MODEL_DIR}"

  CHECKPOINT=""
  for candidate in \
    "${MODEL_DIR}/stage_4_low_tcnr_best.pt" \
    "${MODEL_DIR}/stage_3_realistic_mixed_best.pt" \
    "${MODEL_DIR}/stage_2_moderate_noise_best.pt" \
    "${MODEL_DIR}/stage_4_best.pt" \
    "${MODEL_DIR}/stage_3_best.pt" \
    "${MODEL_DIR}/stage_2_best.pt" \
    "${MODEL_DIR}/last.pt" \
    "${MODEL_DIR}/best.pt"; do
    if [[ -f "${candidate}" ]]; then
      CHECKPOINT="${candidate}"
      break
    fi
  done
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "missing_checkpoint=no_stage_or_last_checkpoint_found"
    exit 1
  fi
  echo "export_checkpoint=${CHECKPOINT}"

  PARADIGMS=(
    block
    multi_step
    pseudo_random_binary
  )
  TCNRS=(0.1 0.2 0.5 1.0 2.0 5.0 10.0)

  export_split() {
    local split_name="$1"
    local split_key="$2"
    local split_arg="$3"
    local root="$4"
    mapfile -t CASE_IDS < <(
      python - "${SPLIT_JSON}" "${split_key}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
for case_id in payload.get(key, []):
    print(case_id)
PY
    )

    if [[ "${#CASE_IDS[@]}" -eq 0 ]]; then
      echo "no_${split_name}_cases_found"
      exit 1
    fi

    for case_id in "${CASE_IDS[@]}"; do
      for paradigm in "${PARADIGMS[@]}"; do
        for tcnr in "${TCNRS[@]}"; do
          out="${root}/${case_id}/${paradigm}/tcnr_${tcnr}"
          echo "exporting split=${split_name} case=${case_id} paradigm=${paradigm} tcnr=${tcnr} out=${out}"
          PYTHONPATH=src python -m hybrid_cvr.cli predict-sim \
            --checkpoint "${CHECKPOINT}" \
            --config "${CONFIG}" \
            --sim-root "${SIM_ROOT}" \
            --out "${out}" \
            --split "${split_arg}" \
            --case-id "${case_id}" \
            --paradigm "${paradigm}" \
            --tcnr "${tcnr}" &
          while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
            wait -n
          done
        done
      done
    done
    wait
  }

  rm -rf "${EXPORT_ROOT}" "${VAL_EXPORT_ROOT}"
  mkdir -p "${EXPORT_ROOT}" "${VAL_EXPORT_ROOT}"
  export_split validation val_cases val "${VAL_EXPORT_ROOT}"
  export_split test test_cases test "${EXPORT_ROOT}"

  PYTHONPATH=src python - "${MODEL_DIR}" "${VAL_EXPORT_ROOT}" "${EXPORT_ROOT}" <<'PY'
from pathlib import Path
import csv
import json

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

model_dir = Path(__import__("sys").argv[1])
sets = [
    ("validation", Path(__import__("sys").argv[2])),
    ("test", Path(__import__("sys").argv[3])),
]
rows = []
for split_name, root in sets:
    for meta_path in sorted(root.glob("case_*/*/tcnr_*/prediction_metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out_dir = meta_path.parent
        pred = nib.load(str(out_dir / "predicted_CVR.nii.gz")).get_fdata()
        gt_support = nib.load(str(out_dir / "GT_CVR.nii.gz")).get_fdata()
        valid = np.isfinite(pred) & np.isfinite(gt_support) & (gt_support > 0)
        timing = meta.get("timing_seconds", {})
        row = {
            "split": split_name,
            "case_id": Path(meta["case_dir"]).name,
            "paradigm": meta["paradigm"],
            "tcnr": meta["target_tcnr"],
            "checkpoint": meta["checkpoint"],
            "processed_slices": meta["processed_slices"],
            "simulation_generation_seconds": timing.get("simulation_generation", float("nan")),
            "model_inference_seconds": timing.get("model_inference_only", float("nan")),
        }
        for name, gt_file, pred_file in [
            ("cvr", "GT_CVR.nii.gz", "predicted_CVR.nii.gz"),
            ("delay", "GT_delay.nii.gz", "predicted_delay.nii.gz"),
            ("T", "GT_T.nii.gz", "predicted_T.nii.gz"),
        ]:
            gt = nib.load(str(out_dir / gt_file)).get_fdata()
            estimate = nib.load(str(out_dir / pred_file)).get_fdata()
            mask = valid & np.isfinite(gt) & np.isfinite(estimate)
            truth = gt[mask]
            predicted = estimate[mask]
            error = predicted - truth
            gt_mean = float(np.mean(truth))
            row[f"gt_{name}_mean"] = gt_mean
            row[f"pred_{name}_mean"] = float(np.mean(predicted))
            row[f"{name}_mae"] = float(np.mean(np.abs(error)))
            row[f"{name}_relative_mae_percent"] = float(100.0 * np.mean(np.abs(error)) / max(abs(gt_mean), 1e-8))
            row[f"{name}_bias"] = float(np.mean(error))
            row[f"{name}_rmse"] = float(np.sqrt(np.mean(error * error)))
            row[f"{name}_pearson_r"] = float(np.corrcoef(truth, predicted)[0, 1])
        sigma = nib.load(str(out_dir / "predicted_uncertainty_sigma.nii.gz")).get_fdata()
        row["sigma_mean"] = float(np.mean(sigma[valid & np.isfinite(sigma)]))
        rows.append(row)

summary = model_dir / "validation_test_prediction_summary.csv"
with summary.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote_summary={summary} rows={len(rows)}")

metric_names = [
    f"{parameter}_{metric}"
    for parameter in ("cvr", "delay", "T")
    for metric in ("mae", "relative_mae_percent", "bias", "rmse", "pearson_r")
]
grouped_rows = []
for split_name in ("validation", "test"):
    split_rows = [row for row in rows if row["split"] == split_name]
    for paradigm in ("block", "multi_step", "pseudo_random_binary"):
        for tcnr in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
            group = [
                row for row in split_rows
                if row["paradigm"] == paradigm and float(row["tcnr"]) == tcnr
            ]
            if not group:
                continue
            item = {"split": split_name, "paradigm": paradigm, "tcnr": tcnr, "n_cases": len(group)}
            for metric in metric_names + ["simulation_generation_seconds", "model_inference_seconds"]:
                values = np.asarray([row[metric] for row in group], dtype=float)
                item[f"{metric}_mean"] = float(np.nanmean(values))
                item[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            grouped_rows.append(item)

condition_summary = model_dir / "validation_test_condition_metrics.csv"
with condition_summary.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(grouped_rows[0].keys()))
    writer.writeheader()
    writer.writerows(grouped_rows)
print(f"wrote_condition_metrics={condition_summary} rows={len(grouped_rows)}")

for split_name, root in sets:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for axis, paradigm in zip(axes, ("block", "multi_step", "pseudo_random_binary")):
        traces = sorted(root.glob(f"case_*/{paradigm}/tcnr_2.0/etco2_trace.csv"))
        if not traces:
            continue
        trace = np.genfromtxt(traces[0], delimiter=",", names=True)
        axis.plot(trace["time_seconds"], trace["etco2_clean_mmhg"], label="Clean", linewidth=2.0)
        axis.plot(trace["time_seconds"], trace["etco2_model_input_mmhg"], label="Model input", linewidth=1.2, alpha=0.85)
        axis.set_title(paradigm.replace("_", " ").title())
        axis.set_ylabel("ETCO2 (mmHg)")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(model_dir / f"{split_name}_etco2_paradigms_used.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
PY

  echo "post_training_export_finished=$(date -Is)"
  echo "validation_outputs=${VAL_EXPORT_ROOT}"
  echo "test_outputs=${EXPORT_ROOT}"
} >> "${WATCH_LOG}" 2>&1
