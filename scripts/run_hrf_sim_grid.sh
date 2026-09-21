#!/usr/bin/env bash
set -euo pipefail

MANIFEST="$1"
CHECKPOINT="$2"
CONFIG="$3"
SIM_ROOT="$4"
OUT_ROOT="$5"
WORKERS="${6:-4}"

mkdir -p "${OUT_ROOT}"
tail -n +2 "${MANIFEST}" | xargs -P "${WORKERS}" -n 3 bash -c '
  case_id="$1"; paradigm="$2"; tcnr="$3"
  out="'"${OUT_ROOT}"'/${case_id}/${paradigm}/tcnr_${tcnr}"
  if [[ -f "${out}/hrf_fit_metadata.json" ]]; then exit 0; fi
  PYTHONPATH=src .venv/bin/python scripts/evaluate_hrf_sim_condition.py \
    --checkpoint "'"${CHECKPOINT}"'" \
    --config "'"${CONFIG}"'" \
    --sim-root "'"${SIM_ROOT}"'" \
    --case-id "${case_id}" --paradigm "${paradigm}" --tcnr "${tcnr}" --out "${out}"
' _

echo "hrf_grid_complete=$(find "${OUT_ROOT}" -name hrf_fit_metadata.json | wc -l | tr -d " ")"
