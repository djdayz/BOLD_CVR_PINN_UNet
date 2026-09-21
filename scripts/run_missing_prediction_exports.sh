#!/usr/bin/env bash
set -euo pipefail

MANIFEST="$1"
CHECKPOINT="$2"
CONFIG="$3"
SIM_ROOT="$4"
OUTPUT_ROOT="$5"
MAX_PARALLEL="${6:-4}"
LOG_PATH="${7:-${OUTPUT_ROOT}/missing_prediction_exports.log}"

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${LOG_PATH}")"

run_export() {
  local split="$1"
  local case_id="$2"
  local paradigm="$3"
  local tcnr="$4"
  local split_dir="${split}"
  [[ "${split}" == "val" ]] && split_dir="validation"
  local out="${OUTPUT_ROOT}/${split_dir}/${case_id}/${paradigm}/tcnr_${tcnr}"
  PYTHONPATH=src python -m hybrid_cvr.cli predict-sim \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --sim-root "${SIM_ROOT}" \
    --out "${out}" \
    --split "${split}" \
    --case-id "${case_id}" \
    --paradigm "${paradigm}" \
    --tcnr "${tcnr}"
}

export -f run_export
export CHECKPOINT CONFIG SIM_ROOT OUTPUT_ROOT

{
  echo "missing_export_started=$(date -Is)"
  while read -r split case_id paradigm tcnr; do
    [[ -z "${split}" || "${split}" == \#* ]] && continue
    run_export "${split}" "${case_id}" "${paradigm}" "${tcnr}" &
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
      wait -n
    done
  done < "${MANIFEST}"
  wait
  echo "missing_export_finished=$(date -Is)"
} >> "${LOG_PATH}" 2>&1
