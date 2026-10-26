#!/usr/bin/env bash
set -euo pipefail

MANIFEST="$1"
CHECKPOINT="$2"
CONFIG="$3"
SIM_ROOT="$4"
OUT_ROOT="$5"
WORKERS="${6:-1}"

mkdir -p "${OUT_ROOT}"

run_hrf_condition() {
  local split="$1"
  local case_id="$2"
  local paradigm="$3"
  local tcnr="$4"
  local out="${OUT_ROOT}/${split}/${case_id}/${paradigm}/tcnr_${tcnr}"
  [[ -f "${out}/hrf_fit_metadata.json" ]] && return 0
  PYTHONPATH=src python scripts/evaluate_hrf_sim_condition.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --sim-root "${SIM_ROOT}" \
    --split "${split}" \
    --case-id "${case_id}" \
    --paradigm "${paradigm}" \
    --tcnr "${tcnr}" \
    --out "${out}"
}

export -f run_hrf_condition
export CHECKPOINT CONFIG SIM_ROOT OUT_ROOT

while read -r split case_id paradigm tcnr; do
  [[ -z "${split}" || "${split}" == "split" || "${split}" == \#* ]] && continue
  run_hrf_condition "${split}" "${case_id}" "${paradigm}" "${tcnr}" &
  while (( $(jobs -rp | wc -l) >= WORKERS )); do
    wait -n
  done
done < "${MANIFEST}"
wait

echo "hrf_conditions_complete=$(find "${OUT_ROOT}" -name hrf_fit_metadata.json | wc -l | tr -d ' ')"
