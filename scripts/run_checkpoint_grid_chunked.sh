#!/usr/bin/env bash
set -euo pipefail

HOST="$1"
PORT="$2"
KEY="$3"
LOCAL_REPO="$4"
REMOTE_REPO="$5"
CHECKPOINT_NAME="$6"
LOCAL_GRID_NAME="$7"

LOCAL_ROOT="${LOCAL_REPO}/data/evaluation/${LOCAL_GRID_NAME}"
REMOTE_MODEL="data/models/self_supervised_temporal_3d"
REMOTE_OUT="${REMOTE_MODEL}/checkpoint_chunk_exports"

for chunk in 1 2 3; do
  manifest="${LOCAL_ROOT}/manifests/chunk_${chunk}.tsv"
  scp -P "${PORT}" -i "${KEY}" "${manifest}" "${HOST}:${REMOTE_REPO}/${REMOTE_MODEL}/checkpoint_chunk.tsv"
  ssh -p "${PORT}" -i "${KEY}" "${HOST}" "cd '${REMOTE_REPO}' && \
    .venv/bin/python -c \"from pathlib import Path; import shutil; p=Path('${REMOTE_OUT}'); shutil.rmtree(p) if p.exists() else None\" && \
    env PATH='${REMOTE_REPO}/.venv/bin':\"\$PATH\" PYTHONPATH=src bash scripts/run_missing_prediction_exports.sh \
      '${REMOTE_MODEL}/checkpoint_chunk.tsv' \
      '${REMOTE_MODEL}/${CHECKPOINT_NAME}' \
      configs/train_temporal_hybrid_3d_v2_refine.yaml \
      data/simulated \
      '${REMOTE_OUT}' \
      4 \
      '${REMOTE_MODEL}/logs/${LOCAL_GRID_NAME}_chunk_${chunk}.log'"
  for split in validation test; do
    mkdir -p "${LOCAL_ROOT}/${split}"
    rsync -az --partial -e "ssh -p ${PORT} -i ${KEY}" \
      "${HOST}:${REMOTE_REPO}/${REMOTE_OUT}/${split}/" \
      "${LOCAL_ROOT}/${split}/" 2>/dev/null || true
  done
  count=$(find "${LOCAL_ROOT}/validation" "${LOCAL_ROOT}/test" -name prediction_metadata.json | wc -l | tr -d ' ')
  expected=$((chunk * 210))
  if [[ "${count}" -ne "${expected}" ]]; then
    echo "Expected ${expected} exports after chunk ${chunk}, found ${count}" >&2
    exit 1
  fi
  ssh -p "${PORT}" -i "${KEY}" "${HOST}" "cd '${REMOTE_REPO}' && \
    .venv/bin/python -c \"from pathlib import Path; import shutil; p=Path('${REMOTE_OUT}'); shutil.rmtree(p) if p.exists() else None\""
  echo "${LOCAL_GRID_NAME}_chunk_${chunk}_complete=${count}/630"
done
