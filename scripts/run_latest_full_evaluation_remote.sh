#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-data/models/self_supervised_cnn1d_cvr_residual_v1/stage_2_mixed_all_conditions_best.pt}"
OUT_ROOT="${2:-data/models/self_supervised_cnn1d_cvr_residual_v1/full_evaluation_all_held_out}"
CONFIG="configs/train_cnn1d_unet3d_cvr_residual.yaml"
SIM_ROOT="data/simulated"
MANIFEST="${OUT_ROOT}/all_conditions.tsv"
RUN_MANIFEST="${OUT_ROOT}/all_conditions_no_header.tsv"

mkdir -p "${OUT_ROOT}/neural" "${OUT_ROOT}/hrf" "${OUT_ROOT}/summary" "${OUT_ROOT}/logs"
.venv/bin/python scripts/build_latest_evaluation_manifest.py --out "${MANIFEST}"
tail -n +2 "${MANIFEST}" > "${RUN_MANIFEST}"

echo "evaluation_started=$(date -Is)"
echo "checkpoint=${CHECKPOINT}"

env PATH="$(pwd)/.venv/bin:${PATH}" PYTHONPATH=src \
  bash scripts/run_missing_prediction_exports.sh \
  "${RUN_MANIFEST}" "${CHECKPOINT}" "${CONFIG}" "${SIM_ROOT}" \
  "${OUT_ROOT}/neural" 1 "${OUT_ROOT}/logs/neural.log"

env PATH="$(pwd)/.venv/bin:${PATH}" PYTHONPATH=src \
  bash scripts/run_hrf_full_manifest.sh \
  "${RUN_MANIFEST}" "${CHECKPOINT}" "${CONFIG}" "${SIM_ROOT}" \
  "${OUT_ROOT}/hrf" 1

.venv/bin/python scripts/summarize_latest_full_evaluation.py \
  --neural-root "${OUT_ROOT}/neural" \
  --hrf-root "${OUT_ROOT}/hrf" \
  --out "${OUT_ROOT}/summary" \
  --expected 630

echo "evaluation_finished=$(date -Is)"
