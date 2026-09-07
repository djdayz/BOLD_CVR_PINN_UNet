# hybrid-cvr-unet-pinn

Research code for unsupervised / self-supervised BOLD-MRI cerebrovascular reactivity
(CVR) mapping with a hybrid U-Net/PINN. The model estimates CVR magnitude, CO2/BOLD
delay, vascular time constant `T`, uncertainty/QC, ODE-reconstructed BOLD percent
signal change (PSC), and residual/QC maps.

This is a research project and is not a clinically validated medical device.

## Scientific Idea

The physiological forward model is the first-order exponential-HRF ODE:

```text
dy_v(t)/dt = [CVR_v * u(t - tau_v) - y_v(t)] / T_v
```

where `y_v(t)` is BOLD PSC, `u(t)` is baseline-subtracted ETCO2, `CVR_v` is
percent BOLD per mmHg, `tau_v` is delay, and `T_v` is the vascular response time
constant. The U-Net predicts parameter maps; the differentiable ODE solver
reconstructs BOLD PSC; training optimizes reconstruction, ODE residual,
uncertainty, smoothness, prior, and consistency losses.

Ground-truth simulated `GT_CVR`, `GT_delay`, and `GT_T` maps are evaluation-only.
They must not be used as model inputs, training losses, validation checkpoint
criteria, or adaptation targets.

## Training/Data Strategy

Use on-the-fly mixed training as the main strategy. Do not train sequentially on
one tCNR/paradigm, delete it, then move to the next, because that risks
catastrophic forgetting and produces a checkpoint specialised to recent
conditions rather than one robust checkpoint.

Training generates BOLD PSC on the fly. Each training item samples a parameter-map
case ID from the training split, an axial slice/patch, ETCO2 paradigm, tCNR level,
noise seed, motion-spike setting, BOLD drift setting, and ETCO2 measurement-noise
setting. Training batches therefore mix paradigms and tCNR levels.

Validation uses held-out validation parameter-map cases only when multiple
parameter cases are available. Validation seeds/configs are fixed so validation
loss is comparable across epochs. Validation may be deterministic on-the-fly or
cached in compact npz/zarr/HDF5 form. Checkpoint selection must use
self-supervised validation losses only, never GT CVR/delay/T metrics.

Testing uses held-out test parameter-map cases only when available. Test
seeds/configs are fixed. GT CVR/delay/T maps may be used only in testing for final
metrics and plots.

Storage should keep GT parameter maps, tissue-fraction maps, ETCO2 traces,
configs, and seeds. Do not save every full 4D simulated BOLD NIfTI permanently;
save only small full-4D QC examples when configured, for example 1-3 examples per
paradigm/tCNR. For reproducible validation/test, prefer compressed npz/zarr/HDF5
or deterministic on-the-fly regeneration from saved seeds.

Current one-case debug runs with only `case_000` cannot provide true held-out
parameter-map validation/test splits. They are useful for pipeline checks and
prediction export, but publication-grade validation/testing requires additional
parameter-map cases.

## Data Layout

```text
data/raw/
  sub-01/
    ses-01/
      anat/sub-01_ses-01_T1w.nii.gz
      cvr/sub-01_ses-01_bold.nii.gz
      cvr/sub-01_ses-01_gas_traces.txt
```

The scanner supports any `sub-*` / `ses-*` pair with `anat/*_T1w.nii` or
`anat/*_T1w.nii.gz`, `cvr/*_bold.nii.gz`, and `cvr/*_gas_traces.txt`.

## Installation

Use Python 3.11 if possible. Python 3.10 is also supported for neuroimaging
dependency compatibility.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

## Main Commands

```bash
hybrid-cvr scan-data --bids-root data/raw --out data/processed/manifest.csv
hybrid-cvr preprocess-real --config configs/preprocessing.yaml --manifest data/processed/manifest.csv
hybrid-cvr segment-real --config configs/segmentation.yaml --manifest data/processed/manifest.csv
hybrid-cvr fit-real-cvr --config configs/real_cvr_fit.yaml --manifest data/processed/manifest.csv
hybrid-cvr pool-distributions --config configs/simulation.yaml --real-fit-dir data/derivatives/real_cvr --seg-dir data/derivatives/segmentation --out data/distributions/tissue_parameter_samples.parquet
hybrid-cvr simulate --config configs/simulation.yaml --out data/simulated
hybrid-cvr train --config configs/train_unet_pinn.yaml --sim-root data/simulated --out data/models/unet_pinn
hybrid-cvr evaluate --checkpoint data/models/unet_pinn/best.pt --sim-root data/simulated/test --out data/derivatives/evaluation
hybrid-cvr infer-real --checkpoint data/models/unet_pinn/best.pt --config configs/train_unet_pinn.yaml --subject sub-01 --session ses-01 --bids-root data/raw --out data/derivatives/real_inference
hybrid-cvr adapt-real --checkpoint data/models/unet_pinn/best.pt --config configs/train_unet_pinn.yaml --subject sub-01 --session ses-01 --out data/derivatives/real_adaptation
```

## Pipeline

1. Scan BIDS-like raw data into `manifest.csv`.
2. Preprocess BOLD: discard initial volumes, estimate brain mask, convert to PSC,
   and write QC maps.
3. Process gas traces: parse CO2, detect end-expiration peaks, convert percent to
   mmHg when needed, smooth, baseline-correct, and resample to TR.
4. Segment anatomy where external tools are available; otherwise fail clearly or
   create test-only fallbacks. BOLD-derived vessel-likelihood is not anatomical
   vessel segmentation.
5. Fit conventional lagged GLM and exponential-HRF/ODE baselines for comparison,
   features, distribution pooling, and QC.
6. Pool real-data parameter distributions by high-confidence tissue/region.
7. Generate simulated BOLD using phantom/MIDA-like tissue maps, smooth CO2
   paradigms, the ODE forward model, and realistic noise/artifacts.
8. Train the hybrid U-Net/PINN with self-supervised losses only.
9. Evaluate with simulation ground truth and apply checkpoints to real data.
10. Optionally adapt on real data using reconstruction, residual, uncertainty,
    smoothness, prior, and anchor losses only.

## Vessel-Likelihood Policy

With only T1w, BOLD, and ETCO2, this project computes a BOLD-derived
`vessel_likelihood` map from fit quality, temporal variation, correlation,
response timing, and intensity features. It is a QC/input channel and a region
for separate evaluation. It is not a true anatomical vessel mask unless a
vascular image such as TOF-MRA, SWI, or QSM is supplied.

## Limitations

Real-data baseline CVR maps used for pooling are not true ground truth.
Simulation-to-real domain shift is expected. ETCO2 is a proxy for arterial CO2.
BOLD-CVR is affected by motion, physiological noise, baseline signal, and
vascular artifacts. Delay and `T` may be partially non-identifiable under simple
block paradigms. The first model is 2D/2.5D, not full 3D. Uncertainty maps are QC
indicators, not clinical confidence scores. Independent validation is required.
