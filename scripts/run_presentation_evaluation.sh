#!/usr/bin/env bash
set -euo pipefail

cd /workspace/DJ_BOLD_CVR_PINN_UNet
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_ROOT=data/models/self_supervised_cnn1d_cvr_residual_v1
CHECKPOINT="$MODEL_ROOT/stage_2_mixed_all_conditions_best.pt"
CONFIG=configs/train_cnn1d_unet3d_cvr_residual.yaml
OUT="$MODEL_ROOT/presentation_evaluation"
mkdir -p "$OUT"

run_split() {
  local split="$1"
  local case_id="$2"
  for paradigm in block multi_step pseudo_random_binary; do
    for tcnr in 0.1 0.2 0.5 1.0 2.0 5.0 10.0; do
      local neural="$OUT/neural/$split/$case_id/$paradigm/tcnr_$tcnr"
      local hrf="$OUT/hrf/$split/$case_id/$paradigm/tcnr_$tcnr"
      if [[ ! -f "$neural/prediction_metadata.json" ]]; then
        echo "[$(date -Is)] Neural: $split $case_id $paradigm tCNR=$tcnr"
        .venv/bin/python -m hybrid_cvr.cli predict-sim \
          --checkpoint "$CHECKPOINT" --config "$CONFIG" --sim-root data/simulated \
          --out "$neural" --split "$split" --case-id "$case_id" \
          --paradigm "$paradigm" --tcnr "$tcnr"
      fi
      if [[ ! -f "$hrf/hrf_fit_metadata.json" ]]; then
        echo "[$(date -Is)] HRF: $split $case_id $paradigm tCNR=$tcnr"
        .venv/bin/python scripts/evaluate_hrf_sim_condition.py \
          --checkpoint "$CHECKPOINT" --config "$CONFIG" --sim-root data/simulated --split "$split" \
          --case-id "$case_id" --paradigm "$paradigm" --tcnr "$tcnr" --out "$hrf"
      fi
    done
  done
}

run_split validation case_070
run_split test case_085

.venv/bin/python scripts/summarize_compact_inference_hrf.py \
  --neural-root "$OUT/neural" --hrf-root "$OUT/hrf" --out "$OUT/summary"

echo "[$(date -Is)] Synthetic presentation evaluation complete."
