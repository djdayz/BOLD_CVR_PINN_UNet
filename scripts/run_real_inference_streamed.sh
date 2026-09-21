#!/usr/bin/env bash
set -euo pipefail

HOST="$1"
PORT="$2"
KEY="$3"
LOCAL_REPO="$4"
REMOTE_REPO="$5"
CHECKPOINT="${6:-data/models/self_supervised_temporal_3d/stage_3_realistic_mixed_best.pt}"
LOCAL_OUT="${7:-${LOCAL_REPO}/data/evaluation/real_inference_stage3_best}"
REMOTE_OUT="data/models/self_supervised_temporal_3d/real_inference_stage3_best"
CONFIG="configs/train_temporal_hybrid_3d_v2_refine.yaml"

mkdir -p "${LOCAL_OUT}"
for session_dir in $(find "${LOCAL_REPO}/data/processed" -mindepth 2 -maxdepth 2 -type d | sort); do
  subject="$(basename "$(dirname "${session_dir}")")"
  session="$(basename "${session_dir}")"
  local_session_out="${LOCAL_OUT}/${subject}/${session}"
  if [[ -f "${local_session_out}/prediction_metadata.json" ]]; then
    echo "skip_complete=${subject}/${session}"
    continue
  fi

  echo "uploading=${subject}/${session}"
  ssh -p "${PORT}" -i "${KEY}" "${HOST}" \
    "mkdir -p '${REMOTE_REPO}/data/processed/${subject}/${session}'"
  rsync -az --no-owner --no-group -e "ssh -p ${PORT} -i ${KEY}" \
    "${session_dir}/bold_psc.nii.gz" \
    "${session_dir}/baseline_bold.nii.gz" \
    "${session_dir}/mean_bold.nii.gz" \
    "${session_dir}/valid_signal_mask.nii.gz" \
    "${session_dir}/etco2_resampled.tsv" \
    "${HOST}:${REMOTE_REPO}/data/processed/${subject}/${session}/"

  echo "inferencing=${subject}/${session}"
  ssh -p "${PORT}" -i "${KEY}" "${HOST}" \
    "cd '${REMOTE_REPO}' && PYTHONPATH=src .venv/bin/python -m hybrid_cvr.cli infer-real \
      --checkpoint '${CHECKPOINT}' --config '${CONFIG}' \
      --subject '${subject}' --session '${session}' \
      --processed-root data/processed \
      --segmentation-root data/derivatives/segmentation \
      --vessel-root data/derivatives/vessels \
      --out '${REMOTE_OUT}/${subject}/${session}'"

  mkdir -p "${local_session_out}"
  rsync -az --no-owner --no-group -e "ssh -p ${PORT} -i ${KEY}" \
    "${HOST}:${REMOTE_REPO}/${REMOTE_OUT}/${subject}/${session}/" \
    "${local_session_out}/"
  for required in predicted_CVR.nii.gz predicted_delay.nii.gz predicted_T.nii.gz \
      predicted_uncertainty_sigma.nii.gz reconstructed_BOLD_PSC_mean.nii.gz \
      residual_rms.nii.gz prediction_metadata.json real_inference_qc_report.png; do
    test -s "${local_session_out}/${required}"
  done

  ssh -p "${PORT}" -i "${KEY}" "${HOST}" \
    "rm -rf '${REMOTE_REPO}/data/processed/${subject}/${session}'"
  echo "complete=${subject}/${session}"
done

echo "real_inference_sessions=$(find "${LOCAL_OUT}" -name prediction_metadata.json | wc -l | tr -d ' ')"
