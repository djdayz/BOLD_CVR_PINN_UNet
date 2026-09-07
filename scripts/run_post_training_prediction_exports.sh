#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-data/models/unet_pinn}"
TRAINING_PID="${2:-}"
CONFIG="${3:-configs/train_unet_pinn.yaml}"
SIM_ROOT="${4:-data/simulated}"
SPLIT_JSON="${5:-data/simulated/splits/train_val_test_split.json}"

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
    sinusoidal
    breath_hold_like
    resting_state_like
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
for case_id in payload.get(key, [])[:3]:
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
            --tcnr "${tcnr}"
        done
      done
    done
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
        valid = np.isfinite(pred) & (pred != 0)
        row = {
            "split": split_name,
            "case_id": Path(meta["case_dir"]).name,
            "paradigm": meta["paradigm"],
            "tcnr": meta["target_tcnr"],
            "checkpoint": meta["checkpoint"],
            "processed_slices": meta["processed_slices"],
        }
        for name, fname in [
            ("cvr_abs_error", "CVR_abs_error_from_GT.nii.gz"),
            ("delay_abs_error", "delay_abs_error_from_GT.nii.gz"),
            ("T_abs_error", "T_abs_error_from_GT.nii.gz"),
            ("sigma", "predicted_uncertainty_sigma.nii.gz"),
            ("pred_cvr", "predicted_CVR.nii.gz"),
            ("pred_delay", "predicted_delay.nii.gz"),
            ("pred_T", "predicted_T.nii.gz"),
        ]:
            data = nib.load(str(out_dir / fname)).get_fdata()
            vals = data[valid & np.isfinite(data)]
            row[f"{name}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{name}_median"] = float(np.median(vals)) if vals.size else float("nan")
        rows.append(row)

summary = model_dir / "validation_test_prediction_summary.csv"
with summary.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote_summary={summary} rows={len(rows)}")
PY

  echo "post_training_export_finished=$(date -Is)"
  echo "validation_outputs=${VAL_EXPORT_ROOT}"
  echo "test_outputs=${EXPORT_ROOT}"
} >> "${WATCH_LOG}" 2>&1
