#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 6 ]]; then
  echo "Usage: $0 HOST PORT KEY LOCAL_REPO REMOTE_REPO CHECKPOINT [CHUNK_SIZE]" >&2
  exit 2
fi

HOST="$1"
PORT="$2"
KEY="$3"
LOCAL_REPO="$4"
REMOTE_REPO="$5"
CHECKPOINT="$6"
CHUNK_SIZE="${7:-30}"

CONFIG="configs/train_cnn1d_unet3d_cvr_residual.yaml"
SIM_ROOT="data/simulated"
LOCAL_ROOT="${LOCAL_REPO}/data/evaluation/self_supervised_cnn1d_cvr_residual_v1_all_held_out"
REMOTE_ROOT="data/models/self_supervised_cnn1d_cvr_residual_v1/full_evaluation_work"
MANIFEST="${LOCAL_ROOT}/all_conditions.tsv"
CHUNK_DIR="${LOCAL_ROOT}/manifests"

mkdir -p "${CHUNK_DIR}" "${LOCAL_ROOT}/neural" "${LOCAL_ROOT}/hrf" "${LOCAL_ROOT}/summary"
cd "${LOCAL_REPO}"
.venv/bin/python scripts/build_latest_evaluation_manifest.py --out "${MANIFEST}"

tail -n +2 "${MANIFEST}" | split -l "${CHUNK_SIZE}" - "${CHUNK_DIR}/chunk_"

for chunk in "${CHUNK_DIR}"/chunk_*; do
  chunk_name="$(basename "${chunk}")"
  remote_manifest="${REMOTE_ROOT}/${chunk_name}.tsv"
  echo "[$(date -Is)] Starting ${chunk_name}"
  ssh -p "${PORT}" -i "${KEY}" "${HOST}" \
    "mkdir -p '${REMOTE_REPO}/${REMOTE_ROOT}'"
  scp -P "${PORT}" -i "${KEY}" "${chunk}" "${HOST}:${REMOTE_REPO}/${remote_manifest}"

  ssh -p "${PORT}" -i "${KEY}" "${HOST}" "cd '${REMOTE_REPO}' && \
    PYTHONPATH=src bash scripts/run_missing_prediction_exports.sh \
      '${remote_manifest}' '${CHECKPOINT}' '${CONFIG}' '${SIM_ROOT}' \
      '${REMOTE_ROOT}/neural' 1 '${REMOTE_ROOT}/${chunk_name}_neural.log' && \
    PYTHONPATH=src bash scripts/run_hrf_full_manifest.sh \
      '${remote_manifest}' '${CHECKPOINT}' '${CONFIG}' '${SIM_ROOT}' \
      '${REMOTE_ROOT}/hrf' 1"

  rsync -az --partial -e "ssh -p ${PORT} -i ${KEY}" \
    "${HOST}:${REMOTE_REPO}/${REMOTE_ROOT}/neural/" "${LOCAL_ROOT}/neural/"
  rsync -az --partial -e "ssh -p ${PORT} -i ${KEY}" \
    "${HOST}:${REMOTE_REPO}/${REMOTE_ROOT}/hrf/" "${LOCAL_ROOT}/hrf/"

  ssh -p "${PORT}" -i "${KEY}" "${HOST}" \
    "find '${REMOTE_REPO}/${REMOTE_ROOT}/neural' -mindepth 1 -delete; \
     find '${REMOTE_REPO}/${REMOTE_ROOT}/hrf' -mindepth 1 -delete"
  echo "[$(date -Is)] Finished ${chunk_name}"
done

.venv/bin/python scripts/summarize_latest_full_evaluation.py \
  --neural-root "${LOCAL_ROOT}/neural" \
  --hrf-root "${LOCAL_ROOT}/hrf" \
  --out "${LOCAL_ROOT}/summary" \
  --expected 630

echo "Complete evaluation: ${LOCAL_ROOT}/summary"
