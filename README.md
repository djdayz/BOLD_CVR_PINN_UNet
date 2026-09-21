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
robust Student-t reconstruction likelihood plus cross-paradigm consistency in
the mixed-condition stage.
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
   tissue-specific joint `(CVR, delay, T)` distributions to create
   partial-volume-aware GT parameter-map cases.
10. Generate synthetic 4D BOLD from GT `CVR`, `delay`, `T`, ETCO2 paradigms, and
    tCNR-dependent noise using the ODE forward model.
11. Train the self-supervised 1D-CNN + 3D-U-Net physiology model on complete
    3D brain volumes generated on the fly with mixed cases, ETCO2 paradigms,
    tCNR levels, seeds, drift, motion spikes, and ETCO2 noise.
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

Train the current full-volume self-supervised physiology model:

```bash
hybrid-cvr train \
  --config configs/train_cnn1d_unet3d_physiology.yaml \
  --sim-root data/simulated \
  --out data/models/self_supervised_cnn1d_profiled_cvr_v1
```

Export one synthetic prediction or run real inference:

```bash
hybrid-cvr predict-sim \
  --checkpoint data/models/self_supervised_cnn1d_profiled_cvr_v1/best.pt \
  --config configs/train_cnn1d_unet3d_physiology.yaml \
  --sim-root data/simulated \
  --split test --case-id case_086 --paradigm block --tcnr 10 \
  --out data/evaluation/example_prediction

hybrid-cvr infer-real \
  --checkpoint data/models/self_supervised_cnn1d_profiled_cvr_v1/best.pt \
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

MIDA tissue components are simulated separately and mixed using the 2.5 mm
tissue fractions so partial-volume effects occur in the signal domain. The
current simulations do not apply spatial smoothing or an artificial intensity
gradient. GT parameter maps are generated from tissue-specific joint
distributions, with within-tissue variation and fraction-weighted mixing at
tissue boundaries.

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

Training is performed on complete `94 x 94 x 50` brain volumes with all 480 time
points. Synthetic BOLD is generated on the GPU as each batch is requested; the
full training set is therefore reproducible from parameter-map cases, configs,
and random seeds without storing thousands of 4D NIfTIs.

Each training sample selects a parameter-map case, ETCO2 paradigm, tCNR level,
noise seed, drift, motion-spike setting, AR(1) coefficient, ETCO2 measurement
noise, and optional model mismatch. The model input contains only observable
quantities:

```text
baseline BOLD, mean PSC, PSC standard deviation, hypercapnic PSC change,
tCNR, brain mask, zero-lag CO2 beta/correlation, peak CO2 correlation,
and normalized peak-correlation lag
```

Tissue fractions, tissue identities, GT CVR, GT delay, GT `T`, and pooled
parameter distributions are not model inputs or training targets.

The non-causal voxelwise 1D CNN processes the complete normalized BOLD/ETCO2
history to represent timing and response shape. A full-volume 3D U-Net combines
those temporal embeddings with observable spatial features and predicts
voxelwise delay and `T`. CVR is not a free neural-network output: after delay and
`T` are predicted, a unit-CVR ODE response is generated and CVR is fitted from
the raw BOLD PSC and delta-ETCO2 amplitudes by differentiable least squares.
Only intercept and linear drift are removed during this amplitude fit.

The current config is:

```text
configs/train_cnn1d_unet3d_physiology.yaml
```

Key settings:

```text
spatial input: complete 94 x 94 x 50 volume at approximately 2.5 mm
temporal input: complete 480-point BOLD PSC + ETCO2 + time
temporal encoder: non-causal dilated 1D CNN, 48-channel voxel embedding
spatial model: 3D U-Net, 24 base channels, depth 2
predicted parameters: voxelwise delay and T
profiled parameter: voxelwise CVR from raw physical amplitudes
split fractions: 70% train, 15% validation, 15% test
samples per epoch: 8
batch size: 1 full volume, gradient accumulation: 4
epochs: 700
stage 1: 200 epochs, clean tCNR 2/5/10, balanced paradigms, LR 1e-3
stage 2: 500 epochs, all tCNR/paradigm conditions, LR 5e-4
stage 2 low-tCNR probabilities: 0.05 each for tCNR 0.1 and 0.2
stage 2 loss: reconstruction + 0.05 cross-paradigm consistency
learning rate scheduler: none
validation: 21 fixed samples every 10 epochs
checkpoint metric: lowest self-supervised validation total
```

The reconstruction objective is a robust Student-t negative log-likelihood.
There is no supervised parameter-map loss, tissue-ranking loss, parameter-
distribution prior, or spatial smoothness loss. Validation uses held-out cases
and deterministic seeds. GT maps are permitted only after training for final
validation/test metrics and figures; they are never used for checkpoint
selection.

## Current Model Outputs

The current training run writes:

```text
data/models/self_supervised_cnn1d_profiled_cvr_v1/
  best.pt
  last.pt
  stage_1_identifiable_clean_best.pt
  stage_2_mixed_all_conditions_best.pt
  training_history.csv
  logs/
```

`best.pt` is updated only on validation epochs when the label-free validation
total improves. Validation is intentionally skipped on other epochs; `nan` in
those validation log fields means "not evaluated", not a numerical model
failure.

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
open data/evaluation/current/test/case_086/pseudo_random_binary/tcnr_10.0/predicted_maps_qc.png
```

GT-error maps exist only for held-out synthetic validation/test outputs. For real
subjects, use `predicted_uncertainty_sigma.nii.gz` and `residual_rms.nii.gz` as
QC indicators because real GT parameter maps do not exist.

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
paradigms. A single effective ODE per voxel is an approximation at partial-volume
boundaries where multiple tissue responses are mixed. The model processes the
complete 3D volume and time series, but spatial context and noise-robust
optimization can still soften fine boundaries. Profiled CVR depends on accurate
delay/`T`, BOLD PSC scaling, and ETCO2 amplitude calibration. Uncertainty maps
are research QC indicators, not clinical confidence scores. Independent
validation is required before any scientific claim is treated as robust.
