# Self-Supervised CVR Physiology Model

Research code for unsupervised/self-supervised BOLD-MRI cerebrovascular
reactivity (CVR) mapping with a self-supervised 1D-CNN + 3D-U-Net physiology model and a differentiable
first-order ODE forward model.

The project estimates voxelwise:

- `CVR`: BOLD percent-signal-change response per mmHg ETCO2
- `delay`: ETCO2-to-BOLD temporal lag
- `T`: vascular response time constant in the exponential/ODE HRF
- uncertainty/QC maps
- reconstructed BOLD PSC and residual maps

This is research code only. It is not a clinically validated medical device.

## Scientific Model

The physiological forward model is the first-order exponential HRF ODE:

```text
dy_v(t) / dt = (CVR_v * u(t - tau_v) - y_v(t)) / T_v
```

where:

- `y_v(t)` is reconstructed BOLD PSC for voxel `v`
- `u(t)` is baseline-subtracted ETCO2
- `CVR_v` is the response magnitude
- `tau_v` is the delay
- `T_v` is the response time constant

The non-causal 1D CNN and 3D U-Net predict voxelwise delay and `T`. Given those
timing maps, CVR is profiled differentiably from raw BOLD PSC and delta-ETCO2
amplitudes after removing intercept and linear drift. The differentiable ODE
solver then reconstructs BOLD PSC. Training is strictly self-supervised and uses
robust Student-t reconstruction likelihood, paired-view consistency, nuisance
regularization, and weak label-free physiological constraints.
There is no supervised GT parameter loss, tissue-distribution target, or spatial
smoothness loss.

Ground-truth simulated maps are used to generate synthetic BOLD and to evaluate
held-out synthetic predictions. They are not model inputs and are not used for
checkpoint selection.

## Repository Layout

```text
configs/
  preprocessing.yaml
  segmentation.yaml
  real_cvr_fit.yaml
  simulation.yaml
  train_cnn1d_unet3d_physiology.yaml

src/hybrid_cvr/
  preprocessing/      BOLD motion correction, PSC conversion, gas processing
  segmentation/       FastSurfer/FSL tissue registration and vessel likelihood
  cvr/                GLM and exponential-HRF/ODE real-data CVR fitting
  distributions/      Tissue-specific pooled parameter distributions
  simulation/         MIDA tissue maps and synthetic 4D BOLD generation
  models/             physiology-model definitions
  physiology/        ODE, delay interpolation, losses, uncertainty
  training/           On-the-fly mixed simulation training
  inference/          Synthetic and real-subject prediction export
  visualisation/      QC plots and map summaries

scripts/
  run_checkpoint_grid_chunked.sh
  summarize_self_supervised_grid.py
  evaluate_hrf_sim_condition.py
  summarize_hrf_sim_grid.py
```

Large data, trained checkpoints, synthetic images, and inference outputs are
kept under `data/` locally and should not be committed to GitHub.

## Installation

Python 3.11 is preferred. Python 3.10 is also supported for neuroimaging
dependency compatibility.

```bash
cd /Users/mac/DJ_BOLD_CVR_PINN_UNet
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

External neuroimaging tools used by the full pipeline include FSL, FastSurfer,
and FreeSurfer-compatible label files. GPU training is expected to run faster on
a VM such as RunPod.

## Data Layout

Raw data are expected in a BIDS-like structure:

```text
data/raw/
  sub-01/
    ses-01/
      anat/sub-01_ses-01_T1w.nii.gz
      cvr/sub-01_ses-01_bold.nii.gz
      cvr/sub-01_ses-01_gas_traces.txt
```

The scanner accepts any `sub-*` / `ses-*` pair with a T1w image, BOLD image, and
gas trace. The project has also used the external dataset at:

```text
/Users/mac/Downloads/CVR_repeatability_at_3T_data
```

## Pipeline Overview

1. Scan the BIDS-like raw data and write a manifest.
2. Motion-correct BOLD with MCFLIRT before downstream processing.
3. Convert motion-corrected BOLD to percent signal change.
4. Process ETCO2 traces, apply scan-start/global timing offsets, and resample to
   the BOLD TR.
5. Fit conventional real-data CVR baselines with lagged GLM and ODE/exponential
   HRF fitting.
6. Run FastSurfer tissue segmentation, register T1w/FastSurfer masks to BOLD
   space, and create WM, cortical GM, subcortical GM, and ventricular CSF masks.
7. Create high-confidence BOLD-derived vessel-likelihood masks and exclude those
   voxels from tissue masks when pooling tissue distributions.
8. Pool tissue-specific distributions of CVR, delay, and `T` from real-data maps.
9. Segment MIDA at 0.5 mm, downsample to 2.5 mm tissue-fraction maps, and sample
   pooled distributions to create partial-volume-aware GT parameter-map cases.
10. Generate synthetic 4D BOLD from GT `CVR`, `delay`, `T`, ETCO2 paradigms, and
    tCNR-dependent noise using the ODE forward model.
11. Train the self-supervised 1D-CNN + 3D-U-Net physiology model on-the-fly with mixed cases, slices, ETCO2
    paradigms, tCNR levels, seeds, drift, motion spikes, and ETCO2 noise.
12. Export predicted maps for held-out synthetic validation/test conditions and
    run inference on real MCFLIRT-processed BOLD images.

## Main Commands

Scan and preprocess real data:

```bash
hybrid-cvr scan-data \
  --bids-root data/raw \
  --out data/processed/manifest.csv

hybrid-cvr preprocess-real \
  --config configs/preprocessing.yaml \
  --manifest data/processed/manifest.csv
```

Run segmentation and vessel/tissue mask preparation:

```bash
hybrid-cvr run-fastsurfer \
  --config configs/segmentation.yaml \
  --manifest data/processed/manifest.csv

hybrid-cvr segment-real \
  --config configs/segmentation.yaml \
  --manifest data/processed/manifest.csv

hybrid-cvr segment-vessels \
  --config configs/segmentation.yaml \
  --manifest data/processed/manifest.csv

hybrid-cvr exclude-vessels-from-tissues \
  --config configs/segmentation.yaml \
  --manifest data/processed/manifest.csv
```

Fit real-subject parameter maps and QC plots:

```bash
hybrid-cvr fit-real-cvr \
  --config configs/real_cvr_fit.yaml \
  --manifest data/processed/manifest.csv

hybrid-cvr plot-etco2-qc \
  --config configs/preprocessing.yaml \
  --manifest data/processed/manifest.csv

hybrid-cvr plot-fit-alignment-qc \
  --config configs/real_cvr_fit.yaml \
  --manifest data/processed/manifest.csv
```

Pool tissue distributions and prepare MIDA GT maps:

```bash
hybrid-cvr pool-distributions \
  --config configs/simulation.yaml \
  --real-fit-dir data/derivatives/real_cvr \
  --seg-dir data/derivatives/segmentation \
  --out data/distributions/tissue_parameter_samples.parquet

hybrid-cvr segment-mida \
  --config configs/simulation.yaml

hybrid-cvr generate-mida-parameter-maps \
  --config configs/simulation.yaml

hybrid-cvr prepare-case-index \
  --config configs/train_cnn1d_unet3d_physiology.yaml
```

Generate synthetic BOLD QC examples:

```bash
hybrid-cvr simulate \
  --config configs/simulation.yaml \
  --out data/simulated/bold4d
```

Train the current temporal 3D self-supervised model:

```bash
hybrid-cvr train \
  --config configs/train_cnn1d_unet3d_physiology.yaml \
  --sim-root data/simulated \
  --out data/models/self_supervised_temporal_3d
```

Export one synthetic prediction or run real inference:

```bash
hybrid-cvr predict-sim \
  --checkpoint data/models/self_supervised_temporal_3d/stage_3_realistic_mixed_best.pt \
  --config configs/train_cnn1d_unet3d_physiology.yaml \
  --sim-root data/simulated \
  --split test --case-id case_086 --paradigm block --tcnr 10 \
  --out data/evaluation/example_prediction

hybrid-cvr infer-real \
  --checkpoint data/models/self_supervised_temporal_3d/stage_3_realistic_mixed_best.pt \
  --config configs/train_cnn1d_unet3d_physiology.yaml \
  --subject sub-01 \
  --session ses-01 \
  --processed-root data/processed \
  --segmentation-root data/derivatives/segmentation \
  --vessel-root data/derivatives/vessels \
  --out data/derivatives/real_inference/sub-01/ses-01
```

## Real-Data CVR Mapping

The real-data baseline maps are not ground truth. They are used for comparison,
QC, and tissue-distribution pooling.

The GLM path builds lagged ETCO2 regressors after applying the global
scan-start/acquisition offset. It then searches voxelwise delay candidates and
fits a linear BOLD PSC model:

```text
BOLD_PSC_v(t) = beta0_v + beta1_v * DeltaETCO2(t - tau_v) + error_v(t)
CVR_GLM_v = beta1_v
```

The ODE/exponential-HRF path searches response parameters that align ETCO2-driven
predicted BOLD PSC to the observed BOLD PSC:

```text
dy_v(t) / dt = (CVR_v * DeltaETCO2(t - tau_v) - y_v(t)) / T_v
```

It writes maps such as CVR magnitude, delay, `T`, fit residual/SSR, correlation
peak, and alignment QC plots.

## Tissue Segmentation

FastSurfer products are used for real-subject tissue masks. The implemented
label strategy separates:

- white matter
- cortical grey matter
- subcortical grey matter
- ventricular CSF
- high-confidence vessel-like voxels from BOLD-derived vessel likelihood

Masks are generated in anatomical space and registered to BOLD space using
T1w-to-BOLD registration, nearest-neighbour interpolation, thresholding, and
binary cleanup. The tissue masks used for distribution pooling exclude
high-confidence vessel voxels.

For MIDA, labels are extracted at 0.5 mm and downsampled to 2.5 mm by block
averaging into soft tissue-fraction maps. This avoids nearest-neighbour
downsampling artifacts and better mimics partial volume.

MIDA labels currently used:

```text
WM:          9, 12
CGM:         2, 10
SGM:         4, 5, 7, 8, 16, 17, 20, 21, 99, 116
VCSF:        6
vessel-like: 24, 25
```

## Simulation Strategy

Synthetic BOLD is generated from GT `CVR`, `delay`, and `T` maps using the ODE
forward model. Each synthetic condition combines:

- one MIDA parameter-map case
- one ETCO2 paradigm
- one target tCNR level
- random seed
- optional ETCO2 measurement noise
- drift
- AR(1)-like temporal noise
- sparse motion spikes

The currently used ETCO2 paradigms are:

- `block`
- `multi_step`
- `pseudo_random_binary`

The current tCNR grid is:

```text
0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
```

Full 4D synthetic NIfTIs are saved only for QC examples. Training uses
on-the-fly generation to avoid storing every simulated 4D image.

## Training Strategy

The main strategy is on-the-fly mixed training. The model should not be trained
sequentially on one tCNR/paradigm and then moved to the next, because that risks
catastrophic forgetting and a checkpoint specialized to recent conditions.

Each batch randomly samples:

- parameter-map case ID from the training split
- overlapping 3D brain patch with the complete 480-point time series
- ETCO2 paradigm
- tCNR level
- noise seed
- motion-spike setting
- drift setting
- ETCO2 measurement-noise setting

Validation uses held-out validation parameter-map cases with fixed
seeds/configs, so validation losses are comparable across epochs. Checkpoint
selection uses label-free validation losses only, not GT parameter metrics. GT
parameter maps generate synthetic BOLD but are never passed to the network or
used in the training loss.

Testing uses held-out test parameter-map cases. GT maps are allowed only here for
final metrics and plots.

The current temporal-hybrid config is:

```text
configs/train_cnn1d_unet3d_physiology.yaml
```

Key settings:

```text
patch size: 32 x 32 x 24 at 2.5 mm
temporal input: complete 480-point BOLD PSC + ETCO2 + time
temporal encoder: shared strided 1D CNN, 24-channel voxel embedding
spatial model: 3D U-Net with parameter-specific residual decoders
parameterization: direct voxelwise CVR, delay, and T
split fractions: 70% train, 15% validation, 15% test
epochs: 1000
stage 1: 200 epochs, clean/high-tCNR identification
stage 2: 300 epochs, moderate noise and paired-view consistency
stage 3: 400 epochs, robust mixed conditions and model mismatch
stage 4: 100 epochs, very-low-tCNR stress testing
learning rate scheduler: none
checkpoint metric: label-free validation composite
```

## Current Local Outputs

Current fixed-grid synthetic evaluations and conventional HRF benchmarks are:

```text
data/evaluation/self_supervised_temporal_3d_stage2_grid/
data/evaluation/self_supervised_temporal_3d_stage3_grid/
data/evaluation/hrf_conventional_test_grid/
```

Aggregate result tables are:

```text
data/evaluation/stage2_vs_stage3_overall_metrics.csv
data/evaluation/hrf_conventional_test_grid/test_overall_metrics.csv
data/evaluation/hrf_conventional_test_grid/test_condition_metrics.csv
```

Real-subject inference outputs are written under
`data/derivatives/real_inference/<subject>/<session>/`.

Each real inference folder contains:

```text
predicted_CVR.nii.gz
predicted_delay.nii.gz
predicted_T.nii.gz
predicted_uncertainty_sigma.nii.gz
reconstructed_BOLD_PSC_mean.nii.gz
residual_rms.nii.gz
real_inference_qc_report.png
prediction_metadata.json
slice_summary.csv
```

## Viewing Outputs

Open a real-subject QC report:

```bash
open data/derivatives/real_inference/sub-01/ses-01/real_inference_qc_report.png
```

View real inference maps in FSLeyes:

```bash
fsleyes \
  data/processed/sub-01/ses-01/mean_bold.nii.gz \
  data/derivatives/real_inference/sub-01/ses-01/predicted_CVR.nii.gz \
  data/derivatives/real_inference/sub-01/ses-01/predicted_delay.nii.gz \
  data/derivatives/real_inference/sub-01/ses-01/predicted_T.nii.gz \
  data/derivatives/real_inference/sub-01/ses-01/predicted_uncertainty_sigma.nii.gz
```

Open synthetic prediction QC:

```bash
open data/evaluation/self_supervised_temporal_3d_stage3_grid/test/case_086/pseudo_random_binary/tcnr_10.0/predicted_maps_qc.png
open data/evaluation/self_supervised_temporal_3d_stage3_grid/test/case_086/pseudo_random_binary/tcnr_10.0/prediction_error_uncertainty_qc.png
```

`parameter_error_uncertainty_from_GT.nii.gz` exists only for synthetic
validation/test outputs. It is a normalized GT-error QC map:

```text
(
  abs(pred_CVR - GT_CVR) / 2.2
  + abs(pred_delay - GT_delay) / 80
  + abs(pred_T - GT_T) / 100
) / 3
```

For real subjects, use `predicted_uncertainty_sigma.nii.gz` and
`residual_rms.nii.gz` as QC/uncertainty-style outputs, because no GT maps exist.

## Git Hygiene

Do commit:

- source code under `src/`
- configs under `configs/`
- scripts under `scripts/`
- tests under `tests/`
- README/project documentation

Do not commit:

- raw MRI data
- processed subject data
- MIDA data
- 4D synthetic BOLD NIfTIs
- model checkpoints
- VM output folders
- private subject spreadsheets

Before pushing, check:

```bash
git status --short
git diff --stat
```

## Limitations

Real-data CVR maps used for pooling are not true ground truth. ETCO2 is a proxy
for arterial CO2. BOLD-CVR is affected by motion, physiological noise, baseline
signal, vascular artifacts, partial volume, and scanner/session effects. Delay
and `T` can be partially non-identifiable, especially for simple or low-SNR ETCO2
paradigms. The current model processes overlapping 3D patches with the complete
BOLD time series; sliding-window blending may still soften fine boundaries.
Uncertainty maps are research QC indicators, not clinical confidence scores.
Independent validation is required before any scientific claim is treated as
robust.
