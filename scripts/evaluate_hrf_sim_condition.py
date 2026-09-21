#!/usr/bin/env python3
"""Fit conventional exponential-HRF maps to one deterministic test simulation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from hybrid_cvr.cvr.fit_real import grid_values, nuisance_basis, precompute_flat_ode_regressors
from hybrid_cvr.inference.predict_sim import _merged_config
from hybrid_cvr.simulation.mida_bold import load_mida_parameter_case
from hybrid_cvr.training.on_the_fly_dataset import OnTheFlyCVRDataset, OnTheFlyDatasetConfig
from hybrid_cvr.training.train import _batch_to_device, _materialize_gpu_simulated_batch


def _save(data: np.ndarray, ref: nib.spatialimages.SpatialImage, path: Path) -> None:
    image = nib.Nifti1Image(data.astype(np.float32), ref.affine, ref.header)
    image.set_data_dtype(np.float32)
    nib.save(image, path)


def _fit_grid_gpu(
    bold: torch.Tensor,
    regressors: np.ndarray,
    mask: torch.Tensor,
    candidate_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return best beta and candidate index after removing intercept and drift."""
    device = bold.device
    timepoints = bold.shape[0]
    y = bold[:, mask].float()
    nuisance = torch.as_tensor(nuisance_basis(timepoints, True), device=device, dtype=torch.float32)
    q, _ = torch.linalg.qr(nuisance, mode="reduced")
    y_res = y - q @ (q.T @ y)
    y_ss = torch.sum(y_res.square(), dim=0)
    best_ssr = torch.full((y.shape[1],), torch.inf, device=device)
    best_beta = torch.zeros_like(best_ssr)
    best_index = torch.zeros(y.shape[1], dtype=torch.int64, device=device)
    regs_cpu = torch.from_numpy(regressors.astype(np.float32, copy=False))
    for start in range(0, regressors.shape[0], candidate_batch):
        regs = regs_cpu[start : start + candidate_batch].to(device, non_blocking=True)
        regs_res = regs - (regs @ q) @ q.T
        reg_ss = torch.sum(regs_res.square(), dim=1).clamp_min(1e-12)
        cross = regs_res @ y_res
        ssr = y_ss.unsqueeze(0) - cross.square() / reg_ss.unsqueeze(1)
        local_ssr, local_index = torch.min(ssr, dim=0)
        improved = local_ssr < best_ssr
        chosen = local_index[improved]
        voxels = torch.nonzero(improved, as_tuple=False).squeeze(1)
        best_ssr[improved] = local_ssr[improved]
        best_beta[improved] = cross[chosen, voxels] / reg_ss[chosen]
        best_index[improved] = start + chosen
    return best_beta.cpu().numpy(), best_index.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--paradigm", required=True)
    parser.add_argument("--tcnr", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delay-min", type=float, default=0.0)
    parser.add_argument("--delay-max", type=float, default=80.0)
    parser.add_argument("--delay-step", type=float, default=1.55)
    parser.add_argument("--T-min", type=float, default=2.0)
    parser.add_argument("--T-max", type=float, default=100.0)
    parser.add_argument("--T-step", type=float, default=2.0)
    parser.add_argument("--candidate-batch", type=int, default=128)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    supplied = yaml.safe_load(args.config.read_text())
    run_config = _merged_config(payload.get("config") or {}, supplied)
    dataset_cfg = dict(run_config.get("dataset", {}))
    dataset_cfg.update(
        sim_root=args.sim_root,
        split="test",
        samples_per_epoch=1,
        slice_mode="full_volume",
        paradigms=(args.paradigm,),
        tcnr_levels=(args.tcnr,),
        temporal_mode="full",
        sampling_strategy="balanced_grid",
        randomize_artifacts_for_training=False,
    )
    dataset = OnTheFlyCVRDataset(OnTheFlyDatasetConfig(**dataset_cfg))
    case_dir = args.sim_root / "mida_parameters" / args.case_id
    dataset.case_dirs = [case_dir]
    maps, ref = load_mida_parameter_case(case_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    simulation_start = time.perf_counter()
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))
    batch = _batch_to_device(batch, device)
    batch = _materialize_gpu_simulated_batch(batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    simulation_seconds = time.perf_counter() - simulation_start

    bold = batch["bold_psc"][0]  # T,X,Y,Z
    mask = batch["mask"][0] > 0.5
    time_grid = batch["time_grid"].detach().cpu().numpy()
    etco2 = batch["etco2_for_model"][0].detach().cpu().numpy()
    delays = grid_values(args.delay_min, args.delay_max, args.delay_step)
    T_values = grid_values(args.T_min, args.T_max, args.T_step)
    regressors = precompute_flat_ode_regressors(etco2, time_grid, delays, T_values).astype(np.float32)

    if device.type == "cuda":
        torch.cuda.synchronize()
    fit_start = time.perf_counter()
    beta, best = _fit_grid_gpu(bold, regressors, mask, args.candidate_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - fit_start

    delay_index = best // T_values.size
    T_index = best % T_values.size
    shape = tuple(mask.shape)
    mask_np = mask.cpu().numpy()
    outputs = {}
    for name, values in (
        ("CVR", beta),
        ("delay", delays[delay_index]),
        ("T", T_values[T_index]),
    ):
        volume = np.zeros(shape, dtype=np.float32)
        volume[mask_np] = values.astype(np.float32)
        path = args.out / f"hrf_{name}.nii.gz"
        _save(volume, ref, path)
        outputs[name] = str(path)

    metadata = {
        "case_id": args.case_id,
        "split": "test",
        "paradigm": args.paradigm,
        "target_tcnr": args.tcnr,
        "simulation_generation_seconds": simulation_seconds,
        "hrf_parameter_fit_seconds": fit_seconds,
        "n_brain_voxels": int(mask.sum().item()),
        "n_delay_candidates": int(delays.size),
        "n_T_candidates": int(T_values.size),
        "n_joint_candidates": int(regressors.shape[0]),
        "etco2_input": "measured",
        "outputs": outputs,
    }
    (args.out / "hrf_fit_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
