# Training Optimization Audit

## Current Diagnosis

The current full-brain training pipeline uses on-the-fly simulated BOLD PSC and a 3D U-Net/PINN model. The U-Net receives observable spatial feature maps, predicts CVR, delay, T, and uncertainty, and the differentiable ODE reconstructs the BOLD PSC time series.

The audit found that the previous 15-channel input did not give the image encoder enough direct temporal alignment evidence. Temporal information reached the loss through ODE reconstruction, but the encoder mostly saw summary feature channels. The optimized input now uses 20 observable channels: the original 15 channels plus five BOLD/ETCO2 temporal-response summaries.

Poor recovery of delay or T may still come from:

- true practical non-identifiability under the ETCO2 paradigm and noise level,
- parameterization or prior collapse in the output heads,
- insufficient temporal information in the network input,
- or the training objective preferring reconstruction over parameter disentanglement.

## Phase 1: Oracle Identifiability

Before changing the architecture again, run an oracle single-voxel recovery experiment. This removes the U-Net from the problem and fits CVR, delay, and T directly from the known ETCO2 trace and simulated BOLD response.

Command:

```bash
hybrid-cvr oracle-identifiability --out data/qc/oracle_identifiability
```

Quick smoke command:

```bash
hybrid-cvr oracle-identifiability --out data/qc/oracle_identifiability_quick --quick --no-plot
```

Outputs:

- `oracle_predictions.csv`: true and recovered CVR, delay, and T for each paradigm, tCNR, and seed.
- `oracle_metrics.csv`: RMSE, MAE, bias, Pearson, and Spearman by parameter and condition.
- `profile_fixed_T.csv`: loss profiles when T is fixed and CVR/delay are refit.
- `profile_fixed_delay.csv`: loss profiles when delay is fixed and CVR/T are refit.
- `oracle_summary.json`: configuration and interpretation notes.

Interpretation:

- If oracle recovery fails or the profile losses are flat, the parameter is not identifiable under that stimulus/noise condition.
- If oracle recovery works but U-Net/PINN recovery fails, the issue is the model, parameter heads, priors, or training curriculum.
- Do not use GLM superiority claims until the simulated validation/test metrics support them.

## Next Optimization Gates

1. Check whether T and delay are identifiable from block, multi-step, and pseudo-random binary paradigms across tCNR.
2. Inspect output-head saturation and prior collapse by comparing predicted parameter histograms with GT distributions.
3. Use temporal-response feature channels so the model sees compact dynamic BOLD/ETCO2 alignment evidence directly.
4. Keep the physics-only/self-supervised training path intact; synthetic-supervised losses should be implemented only as an ablation.
5. Run GLM/exponential-HRF baselines on the same synthetic test set before claiming the PINN improves recovery.
