from __future__ import annotations

from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time
from typing import Any

from hybrid_cvr.models.constraints import ParameterRanges
from hybrid_cvr.models.cnn1d_unet3d_physiology import CNN1DUNet3DPhysiologyModel
from hybrid_cvr.physiology.losses import (
    LossWeights,
    assert_no_supervised_parameter_loss,
    consistency_loss,
    self_supervised_loss,
)
from hybrid_cvr.physiology.torch_ode import simulate_ode_bold_torch
from hybrid_cvr.simulation.mida_bold import TISSUE_FRACTION_NAMES
from hybrid_cvr.training.checkpointing import save_checkpoint
from hybrid_cvr.training.on_the_fly_dataset import (
    ON_THE_FLY_FEATURE_NAMES,
    TEMPORAL_RESPONSE_LAG_SECONDS,
    OnTheFlyCVRDataset,
    OnTheFlyDatasetConfig,
)


def validate_training_batch_contract(batch: dict, loss_config: dict | None = None) -> None:
    assert_no_supervised_parameter_loss(batch, loss_config)


def validate_training_config(config: dict) -> None:
    training = config.get("training", {})
    features = config.get("features", {})
    if "use_supervised_parameter_loss" in training:
        raise ValueError("Supervised parameter loss configuration is not supported.")
    if bool(features.get("use_gt_maps_as_input", False)):
        raise ValueError("features.use_gt_maps_as_input=true is forbidden.")
    losses = config.get("losses", {})
    assert_no_supervised_parameter_loss({}, losses)
    for stage in config.get("stages", {}).values():
        assert_no_supervised_parameter_loss({}, dict(stage).get("losses", {}))


def train_physiology_model(
    config: dict[str, Any],
    sim_root: str | Path,
    out_dir: str | Path,
    *,
    max_epochs: int | None = None,
    samples_per_epoch: int | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    torch = __import__("torch")

    validate_training_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs_dir = out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(logs_dir / "training_config_resolved.yaml", config)
    case_split_metadata = _load_case_split_metadata(config)
    seed = int(config.get("dataset", {}).get("seed", 17))
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = int(config.get("training", {}).get("batch_size", 2))
    training_cfg = config.get("training", {})
    first_stage_name, first_stage_cfg = _stage_plan(config, max_epochs)[0]
    train_dataset = _make_dataset(config, sim_root, "train", samples_per_epoch, first_stage_cfg)
    model = _build_model(config, in_channels=len(train_dataset.feature_names)).to(device)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer)
    resume_payload = _load_resume_checkpoint(out, model, optimizer, scheduler, device, config)
    use_amp = bool(config.get("training", {}).get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    clip_norm = float(config.get("training", {}).get("gradient_clip_norm", 1.0))
    grad_accum_steps = max(1, int(config.get("training", {}).get("gradient_accumulation_steps", 1)))
    best_loss = float("inf")
    best_path = out / "best.pt"
    stage_best_losses: dict[str, float] = {}
    history: list[dict[str, float | int | str]] = _load_training_history_csv(
        out / "training_history.csv"
    )
    if resume_payload is not None:
        best_loss = _best_validation_loss_from_history(history, default=best_loss)
    condition_rows: list[dict[str, Any]] = []
    global_epoch = int(resume_payload.get("epoch", 0)) if resume_payload is not None else 0
    if (
        resume_payload is not None
        and _is_stage_boundary_epoch(config, global_epoch, max_epochs)
        and bool(training_cfg.get("reset_optimizer_on_stage_boundary_resume", True))
    ):
        optimizer = _build_optimizer(config, model)
        scheduler = _build_scheduler(config, optimizer)
    final_train_dataset = train_dataset
    final_val_dataset = train_dataset
    completed_epochs_before_stage = 0
    for stage_name, stage_cfg in _stage_plan(config, max_epochs):
        stage_epochs = int(stage_cfg["epochs"])
        if global_epoch >= completed_epochs_before_stage + stage_epochs:
            completed_epochs_before_stage += stage_epochs
            continue
        start_stage_epoch = max(1, global_epoch - completed_epochs_before_stage + 1)
        model.T_mode = str(stage_cfg.get("T_mode", config.get("model", {}).get("T_mode", model.T_mode)))
        train_dataset = _make_dataset(config, sim_root, "train", samples_per_epoch, stage_cfg)
        condition_rows.extend(
            {
                "stage": stage_name,
                "paradigm": paradigm,
                "target_tcnr": tcnr,
            }
            for paradigm, tcnr in train_dataset.condition_grid
        )
        try:
            val_dataset = _make_dataset(
                config,
                sim_root,
                "val",
                max(1, min(len(train_dataset), int(config.get("validation", {}).get("samples", 8)))),
                stage_cfg,
            )
        except FileNotFoundError:
            val_dataset = train_dataset
        final_train_dataset = train_dataset
        final_val_dataset = val_dataset
        train_num_workers = int(training_cfg.get("num_workers", 0))
        val_num_workers = int(training_cfg.get("validation_num_workers", 0))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_num_workers,
            pin_memory=bool(training_cfg.get("pin_memory", False)),
            **_loader_worker_options(training_cfg, train_num_workers),
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            pin_memory=bool(training_cfg.get("pin_memory", False)) and val_num_workers > 0,
            **_loader_worker_options(training_cfg, val_num_workers),
        )
        loss_weights = _loss_weights_from_config(
            _stage_loss_config(config.get("losses", {}), stage_cfg.get("losses"))
        )
        stage_lr = float(stage_cfg.get("lr", config.get("optimizer", {}).get("lr", 1e-4)))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = stage_lr
        for stage_epoch in range(start_stage_epoch, stage_epochs + 1):
            global_epoch += 1
            epoch_start = time.time()
            train_dataset.set_epoch(global_epoch)
            val_dataset.set_epoch(0)
            train_metrics = _run_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                loss_weights,
                device,
                clip_norm,
                use_amp,
                train=True,
                gradient_accumulation_steps=grad_accum_steps,
            )
            validate_every = max(1, int(training_cfg.get("validate_every_epochs", 1)))
            should_validate = bool(validate) and (
                stage_epoch == 1
                or stage_epoch == stage_epochs
                or stage_epoch % validate_every == 0
            )
            val_metrics = (
                _run_epoch(
                    model,
                    val_loader,
                    None,
                    scaler,
                    loss_weights,
                    device,
                    clip_norm,
                    use_amp=False,
                    train=False,
                    gradient_accumulation_steps=1,
                )
                if should_validate
                else {key: float("nan") for key in train_metrics}
            )
            val_total = float(val_metrics["total"])
            if scheduler is not None and should_validate:
                scheduler.step(val_total)
            lr = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": stage_epoch,
                "T_mode": model.T_mode,
                "learning_rate": lr,
                "epoch_time_seconds": time.time() - epoch_start,
                "gpu_memory_mb": _gpu_memory_mb(torch, device),
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            }
            history.append(row)
            print(_format_epoch_log(row), flush=True)
            _write_history_csv(out / "training_history.csv", history)
            _write_history_csv(logs_dir / "train_loss.csv", history)
            _write_history_csv(logs_dir / "val_loss.csv", history)
            _write_stage_history(logs_dir / "stage_history.json", history)
            _save_training_checkpoint(
                out / "last.pt",
                model,
                optimizer,
                scheduler,
                global_epoch,
                val_metrics,
                config,
                train_dataset,
                seed,
                model.T_mode,
                stage_name,
                case_split_metadata,
            )
            if should_validate and val_total < best_loss:
                best_loss = val_total
                _save_training_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    global_epoch,
                    val_metrics,
                    config,
                    train_dataset,
                    seed,
                    model.T_mode,
                    stage_name,
                    case_split_metadata,
                )
            stage_best = stage_best_losses.get(stage_name, float("inf"))
            if should_validate and val_total < stage_best:
                stage_best_losses[stage_name] = val_total
                _save_training_checkpoint(
                    out / _stage_checkpoint_name(stage_name),
                    model,
                    optimizer,
                    scheduler,
                    global_epoch,
                    val_metrics,
                    config,
                    train_dataset,
                    seed,
                    model.T_mode,
                    stage_name,
                    case_split_metadata,
                )
        _save_training_checkpoint(
            out / f"{stage_name}_end.pt",
            model,
            optimizer,
            scheduler,
            global_epoch,
            val_metrics,
            config,
            train_dataset,
            seed,
            model.T_mode,
            stage_name,
            case_split_metadata,
        )
        completed_epochs_before_stage += stage_epochs
    history_path = out / "training_history.csv"
    _write_history_csv(history_path, history)
    condition_grid_path = out / "training_condition_grid.json"
    condition_grid_path.write_text(json.dumps(condition_rows, indent=2), encoding="utf-8")
    status_path = out / "STATUS.txt"
    status_path.write_text(
        (
            f"trained_epochs={global_epoch}\n"
            f"best_self_supervised_validation_loss={best_loss:.8g}\n"
            f"best_checkpoint={best_path}\n"
            f"device={device}\n"
            f"train_cases={len(final_train_dataset.case_dirs)}\n"
            f"val_cases={len(final_val_dataset.case_dirs)}\n"
            f"feature_channels={len(final_train_dataset.feature_names)}\n"
            f"condition_grid={condition_grid_path}\n"
        ),
        encoding="utf-8",
    )
    return {
        "out_dir": out,
        "best_checkpoint": best_path,
        "history": history_path,
        "condition_grid": condition_grid_path,
        "status": status_path,
        "best_loss": best_loss,
        "epochs": global_epoch,
        "device": str(device),
        "train_cases": len(final_train_dataset.case_dirs),
        "val_cases": len(final_val_dataset.case_dirs),
    }


def _run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    loss_weights: LossWeights,
    device,
    clip_norm: float,
    use_amp: bool,
    *,
    train: bool,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float]:
    torch = __import__("torch")
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(loader):
        batch = _batch_to_device(batch, device)
        if "simulate_on_gpu" in batch:
            batch = _materialize_gpu_simulated_batch(batch, device)
        time_grid = batch["time_grid"][0] if batch["time_grid"].ndim == 2 else batch["time_grid"]
        with torch.set_grad_enabled(train):
            autocast_context = (
                torch.amp.autocast("cuda", enabled=True) if use_amp else nullcontext()
            )
            with autocast_context:
                pred = model(
                    batch["features"],
                    etco2=batch["etco2_for_model"],
                    time_grid=time_grid,
                    mask=batch["mask"],
                    tissue_maps=None,
                    bold_psc=batch["bold_psc"],
                    valid_time_mask=batch.get("valid_time_mask"),
                )
                loss_dict = self_supervised_loss(pred, {**batch, "time_grid": time_grid}, loss_weights)
                if loss_weights.lambda_view > 0 and "consistency_features" in batch:
                    pred_pair = model(
                        batch["consistency_features"],
                        etco2=batch["consistency_etco2_for_model"],
                        time_grid=time_grid,
                        mask=batch["mask"],
                        tissue_maps=None,
                        bold_psc=batch["consistency_bold_psc"],
                        valid_time_mask=batch.get("valid_time_mask"),
                    )
                    loss_dict["view"] = consistency_loss(pred, pred_pair, mask=batch["mask"])
                    loss_dict["total"] = (
                        loss_dict["total"]
                        + loss_weights.lambda_view * loss_dict["view"]
                        / max(loss_weights.view_scale, 1e-8)
                    )
                loss = loss_dict["total"]
                if train and gradient_accumulation_steps > 1:
                    loss = loss / float(gradient_accumulation_steps)
        if train:
            scaler.scale(loss).backward()
            should_step = ((batch_idx + 1) % int(gradient_accumulation_steps) == 0) or (batch_idx + 1 == len(loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        for key in ("cvr", "delay", "T", "sigma_y"):
            if key in pred:
                values = pred[key].detach()
                brain = batch["mask"] > 0.5
                while brain.ndim < values.ndim:
                    brain = brain.unsqueeze(1)
                selected = values[brain.expand_as(values)]
                totals[f"mean_{key}"] = totals.get(f"mean_{key}", 0.0) + float(
                    selected.mean().cpu() if selected.numel() else values.mean().cpu()
                )
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def _materialize_gpu_simulated_batch(batch: dict[str, Any], device) -> dict[str, Any]:
    torch = __import__("torch")
    mask = batch["mask"].to(device=device).float()
    mask_bool = mask > 0.5
    fractions = batch["tissue_maps"].to(device=device).float()
    time_grid = batch["time_grid"][0] if batch["time_grid"].ndim == 2 else batch["time_grid"]
    time_grid = time_grid.to(device=device).float()
    etco2_clean = batch["etco2_clean"].to(device=device).float()
    cvr_components = batch["sim_cvr_components"].to(device=device).float()
    delay_components = batch["sim_delay_components"].to(device=device).float()
    T_components = batch["sim_T_components"].to(device=device).float()
    bsz, _, *spatial = fractions.shape
    with torch.no_grad():
        clean = _gpu_simulate_pve_clean(
            cvr_components,
            delay_components,
            T_components,
            etco2_clean,
            time_grid,
            fractions,
            mask_bool,
        )
        clean = _gpu_apply_model_mismatch(
            clean, batch.get("model_mismatch_strength"), mask_bool
        )
        noise = _gpu_seeded_noise_batch(
            clean,
            mask_bool,
            fractions,
            batch["target_tcnr"],
            batch["noise_seed"],
            batch.get("ar1_rho"),
            batch.get("drift_fraction_of_noise"),
            batch.get("motion_spike_probability"),
            batch.get("motion_spike_scale"),
            device,
        )
        noisy = torch.where(mask_bool.unsqueeze(1), clean + noise, torch.zeros_like(clean))
        baseline = _gpu_baseline_intensity(fractions, mask_bool)
        tcnr_map = _gpu_estimate_tcnr(noisy, mask_bool)
        features = _gpu_build_observable_features(
            noisy,
            baseline,
            tcnr_map,
            mask,
            fractions,
            batch["vessel_likelihood"].to(device=device).float(),
            batch["boundary_uncertainty"].to(device=device).float(),
            batch["etco2_for_model"].to(device=device).float(),
            time_grid,
        )
        if "consistency_etco2_clean" in batch:
            pair_etco2_clean = batch["consistency_etco2_clean"].to(device=device).float()
            pair_clean = _gpu_simulate_pve_clean(
                cvr_components,
                delay_components,
                T_components,
                pair_etco2_clean,
                time_grid,
                fractions,
                mask_bool,
            )
            pair_noise = _gpu_seeded_noise_batch(
                pair_clean,
                mask_bool,
                fractions,
                batch["target_tcnr"],
                batch["consistency_noise_seed"],
                batch.get("consistency_ar1_rho"),
                batch.get("consistency_drift_fraction_of_noise"),
                batch.get("consistency_motion_spike_probability"),
                batch.get("consistency_motion_spike_scale"),
                device,
            )
            pair_noisy = torch.where(
                mask_bool.unsqueeze(1), pair_clean + pair_noise, torch.zeros_like(pair_clean)
            )
            pair_tcnr = _gpu_estimate_tcnr(pair_noisy, mask_bool)
            batch["consistency_features"] = _gpu_build_observable_features(
                pair_noisy,
                baseline,
                pair_tcnr,
                mask,
                fractions,
                batch["vessel_likelihood"].to(device=device).float(),
                batch["boundary_uncertainty"].to(device=device).float(),
                batch["consistency_etco2_for_model"].to(device=device).float(),
                time_grid,
            )
            batch["consistency_bold_psc"] = pair_noisy
    batch["features"] = features
    batch["bold_psc"] = noisy
    batch["time_grid"] = time_grid
    batch["etco2_clean"] = etco2_clean
    for key in (
        "simulate_on_gpu",
        "sim_cvr_components",
        "sim_delay_components",
        "sim_T_components",
        "target_tcnr",
        "noise_seed",
        "drift_fraction_of_noise",
        "motion_spike_probability",
        "motion_spike_scale",
        "ar1_rho",
    ):
        batch.pop(key, None)
    return batch


def _gpu_tcnr_noise(
    clean,
    mask_bool,
    fractions,
    target_tcnr,
    ar1_rho=None,
    drift_fraction_of_noise=None,
    motion_spike_probability=None,
    motion_spike_scale=None,
):
    torch = __import__("torch")
    bsz, n_time = clean.shape[:2]
    device = clean.device
    white = torch.randn_like(clean)
    spatial = _gpu_spatially_correlated_noise(white, mask_bool)
    tissue = _gpu_tissue_correlated_noise(clean, fractions)
    noise = 0.35 * white + 0.45 * spatial + 0.20 * tissue
    rho = _batch_scalar(ar1_rho, bsz, device, default=0.35).clamp(0.0, 0.95)
    ar = torch.zeros_like(clean[:, 0])
    innovation_scale = torch.sqrt((1.0 - rho.square()).clamp_min(1e-6))
    for t in range(n_time):
        ar = (
            rho.view(bsz, *([1] * (clean.ndim - 2))) * ar
            + innovation_scale.view(bsz, *([1] * (clean.ndim - 2))) * torch.randn_like(ar)
        )
        noise[:, t] = 0.65 * noise[:, t] + 0.35 * ar
    valid = mask_bool.unsqueeze(1).expand_as(noise)
    if bool(valid.any()):
        vals = noise[valid]
        noise = (noise - vals.mean()) / vals.std(unbiased=False).clamp_min(1e-6)
    temporal_std = clean.std(dim=1, unbiased=False)
    scale_vals = temporal_std[mask_bool]
    signal_scale = (
        torch.median(scale_vals[scale_vals > 0])
        if bool((scale_vals > 0).any())
        else torch.ones((), device=device, dtype=clean.dtype)
    )
    sigma = signal_scale / target_tcnr.clamp_min(1e-3)
    view_shape = (bsz, 1) + (1,) * (clean.ndim - 2)
    noise = noise * sigma.view(view_shape)
    drift_frac = _batch_scalar(drift_fraction_of_noise, bsz, device, default=0.0).clamp_min(0.0)
    if bool((drift_frac > 0).any()):
        drift_time = torch.linspace(-1.0, 1.0, n_time, device=device, dtype=clean.dtype).view(
            1, n_time, *([1] * (clean.ndim - 2))
        )
        drift_map = torch.randn_like(clean[:, 0]) * (drift_frac * sigma).view(
            bsz, *([1] * (clean.ndim - 2))
        )
        noise = noise + drift_time * drift_map.unsqueeze(1)
    spike_prob = _batch_scalar(motion_spike_probability, bsz, device, default=0.0).clamp_min(0.0)
    spike_scale = _batch_scalar(motion_spike_scale, bsz, device, default=0.0).clamp_min(0.0)
    for b in range(bsz):
        spike_count = int(round(float(spike_prob[b].detach().cpu()) * n_time))
        if spike_count <= 0:
            continue
        times = torch.randperm(n_time, device=device)[:spike_count]
        for t in times:
            frame = noise[b, int(t)]
            frame[mask_bool[b]] = frame[mask_bool[b]] + torch.randn(
                (), device=device, dtype=clean.dtype
            ) * sigma[b] * spike_scale[b]
    return torch.where(mask_bool.unsqueeze(1), noise, torch.zeros_like(noise))


def _gpu_spatially_correlated_noise(noise, mask_bool, kernel_size: int = 3):
    torch = __import__("torch")
    nnf = torch.nn.functional
    if noise.ndim != 5:
        return noise
    bsz, n_time, x, y, z = noise.shape
    mask = mask_bool.float().unsqueeze(1)
    flat = noise.reshape(bsz * n_time, 1, x, y, z)
    flat_mask = mask.repeat_interleave(n_time, dim=0)
    pad = int(kernel_size) // 2
    num = nnf.avg_pool3d(flat * flat_mask, kernel_size, stride=1, padding=pad)
    den = nnf.avg_pool3d(flat_mask, kernel_size, stride=1, padding=pad).clamp_min(1e-6)
    return (num / den).reshape_as(noise)


def _gpu_simulate_pve_clean(
    cvr_components,
    delay_components,
    T_components,
    etco2,
    time_grid,
    fractions,
    mask_bool,
):
    torch = __import__("torch")
    bsz, _, *spatial = fractions.shape
    clean = torch.zeros(
        (bsz, int(time_grid.numel()), *spatial),
        device=fractions.device,
        dtype=torch.float32,
    )
    fraction_sum = torch.zeros((bsz, *spatial), device=fractions.device, dtype=torch.float32)
    for tissue_idx in range(len(TISSUE_FRACTION_NAMES)):
        frac = fractions[:, tissue_idx].clamp_min(0.0)
        tissue_mask = mask_bool & (frac > 1e-6)
        if not bool(tissue_mask.any()):
            continue
        response = simulate_ode_bold_torch(
            cvr_components[:, tissue_idx],
            delay_components[:, tissue_idx],
            T_components[:, tissue_idx],
            etco2,
            time_grid,
            mask=tissue_mask.float(),
        )
        clean = clean + response * frac.unsqueeze(1)
        fraction_sum = fraction_sum + torch.where(
            tissue_mask, frac, torch.zeros_like(frac)
        )
    clean = clean / fraction_sum.clamp_min(1e-6).unsqueeze(1)
    return torch.where(mask_bool.unsqueeze(1), clean, torch.zeros_like(clean))


def _gpu_apply_model_mismatch(clean, strength, mask_bool):
    torch = __import__("torch")
    if strength is None:
        return clean
    strength = strength.to(device=clean.device, dtype=clean.dtype).flatten()
    if not bool((strength > 0).any()):
        return clean
    dispersed = clean.clone()
    for time_idx in range(1, clean.shape[1]):
        dispersed[:, time_idx] = 0.82 * dispersed[:, time_idx - 1] + 0.18 * clean[:, time_idx]
    delayed = torch.zeros_like(clean)
    shift = min(12, clean.shape[1] - 1)
    delayed[:, shift:] = dispersed[:, :-shift]
    view = (clean.shape[0], 1) + (1,) * (clean.ndim - 2)
    alpha = strength.view(view).clamp(0.0, 0.4)
    mismatched = (1.0 - alpha) * clean + alpha * dispersed - 0.08 * alpha * delayed
    return torch.where(mask_bool.unsqueeze(1), mismatched, torch.zeros_like(mismatched))


def _gpu_seeded_noise_batch(
    clean,
    mask_bool,
    fractions,
    target_tcnr,
    noise_seeds,
    ar1_rho,
    drift_fraction,
    motion_probability,
    motion_scale,
    device,
):
    torch = __import__("torch")
    rng_devices = [
        device.index if device.index is not None else torch.cuda.current_device()
    ] if device.type == "cuda" else []
    parts = []
    for sample_idx, seed_tensor in enumerate(noise_seeds.flatten()):
        seed = int(seed_tensor.item())
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            parts.append(
                _gpu_tcnr_noise(
                    clean[sample_idx : sample_idx + 1],
                    mask_bool[sample_idx : sample_idx + 1],
                    fractions[sample_idx : sample_idx + 1],
                    target_tcnr[sample_idx : sample_idx + 1]
                    .flatten()
                    .to(device=device)
                    .float(),
                    _slice_optional_batch(ar1_rho, sample_idx),
                    _slice_optional_batch(drift_fraction, sample_idx),
                    _slice_optional_batch(motion_probability, sample_idx),
                    _slice_optional_batch(motion_scale, sample_idx),
                )
            )
    return torch.cat(parts, dim=0)


def _gpu_tissue_correlated_noise(clean, fractions):
    torch = __import__("torch")
    bsz, n_time = clean.shape[:2]
    n_tissue = fractions.shape[1]
    tissue_noise = torch.randn((bsz, n_time, n_tissue), device=clean.device, dtype=clean.dtype)
    for t in range(1, n_time):
        tissue_noise[:, t] = 0.80 * tissue_noise[:, t - 1] + 0.60 * tissue_noise[:, t]
    return (tissue_noise.view(bsz, n_time, n_tissue, *([1] * (clean.ndim - 2))) * fractions.unsqueeze(1)).sum(dim=2)


def _batch_scalar(value, bsz: int, device, default: float):
    torch = __import__("torch")
    if value is None:
        return torch.full((bsz,), float(default), device=device)
    out = value.to(device=device).float().flatten()
    if out.numel() == 1 and bsz > 1:
        out = out.repeat(bsz)
    return out[:bsz]


def _slice_optional_batch(value, index: int):
    if value is None:
        return None
    return value[index : index + 1]


def _gpu_baseline_intensity(fractions, mask_bool):
    torch = __import__("torch")
    offsets = torch.tensor([40.0, 30.0, -40.0, 120.0, 80.0], device=fractions.device)
    baseline = 320.0 + (fractions * offsets.view(1, -1, *([1] * (fractions.ndim - 2)))).sum(dim=1)
    return torch.where(mask_bool, baseline, torch.zeros_like(baseline))


def _gpu_estimate_tcnr(observed, mask_bool):
    torch = __import__("torch")
    temporal_std = observed.std(dim=1, unbiased=False).clamp_min(1e-6)
    tcnr = torch.nan_to_num(1.0 / temporal_std, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.where(mask_bool, tcnr, torch.zeros_like(tcnr))


def _gpu_build_observable_features(
    noisy,
    baseline,
    tcnr_map,
    mask,
    fractions,
    vessel_likelihood,
    boundary_uncertainty,
    etco2_for_model,
    time_grid,
):
    torch = __import__("torch")
    n_time = noisy.shape[1]
    baseline_n = max(4, n_time // 5)
    positive = etco2_for_model > torch.maximum(
        torch.ones_like(etco2_for_model) * 1.0,
        etco2_for_model.amax(dim=1, keepdim=True) * 0.25,
    )
    hyper = torch.zeros_like(noisy[:, 0])
    for b in range(noisy.shape[0]):
        if bool(positive[b].any()):
            hyper[b] = noisy[b, positive[b]].mean(dim=0)
        else:
            hyper[b] = noisy[b, n_time // 2 :].mean(dim=0)
    feature_maps = {
        "baseline_bold": baseline,
        "psc_mean_baseline": noisy[:, :baseline_n].mean(dim=1),
        "psc_mean_hypercapnia": hyper,
        "psc_mean": noisy.mean(dim=1),
        "psc_std": noisy.std(dim=1, unbiased=False),
        "psc_delta": hyper - noisy[:, :baseline_n].mean(dim=1),
        "tcnr": tcnr_map,
        "mask": mask,
        "vessel_likelihood": vessel_likelihood,
        "boundary_uncertainty": boundary_uncertainty,
    }
    for tissue_idx, tissue in enumerate(TISSUE_FRACTION_NAMES):
        feature_maps[f"fraction_{tissue}"] = fractions[:, tissue_idx]
    feature_maps.update(_gpu_temporal_response_features(noisy, etco2_for_model, mask, time_grid))
    features = torch.stack([feature_maps[name] for name in ON_THE_FLY_FEATURE_NAMES], dim=1)
    return torch.where(mask.unsqueeze(1) > 0.5, features, torch.zeros_like(features))


def _gpu_temporal_response_features(noisy, etco2, mask, time_grid):
    torch = __import__("torch")
    bsz, n_time = noisy.shape[:2]
    if etco2.ndim == 1:
        etco2 = etco2.view(1, -1).expand(bsz, -1)
    etco2 = etco2[:, :n_time]
    if time_grid.numel() > 1:
        tr = float(torch.median(torch.diff(time_grid.detach())).detach().cpu())
    else:
        tr = 1.55
    y_centered = noisy - noisy.mean(dim=1, keepdim=True)
    y_std = noisy.std(dim=1, unbiased=False)
    corr_maps = []
    beta_maps = []
    for lag_seconds in TEMPORAL_RESPONSE_LAG_SECONDS:
        lag_frames = int(round(float(lag_seconds) / max(tr, 1e-6)))
        u_lag = _gpu_lag_vector_left_constant(etco2, lag_frames)
        u_centered = u_lag - u_lag.mean(dim=1, keepdim=True)
        u_var = (u_centered * u_centered).mean(dim=1).clamp_min(1e-12)
        u_std = torch.sqrt(u_var)
        view = (bsz, n_time) + (1,) * (noisy.ndim - 2)
        cov = (y_centered * u_centered.view(view)).mean(dim=1)
        beta = cov / u_var.view((bsz,) + (1,) * (noisy.ndim - 2))
        corr = cov / (y_std * u_std.view((bsz,) + (1,) * (noisy.ndim - 2))).clamp_min(1e-6)
        beta_maps.append(torch.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0))
        corr_maps.append(torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
    beta_stack = torch.stack(beta_maps, dim=1)
    corr_stack = torch.stack(corr_maps, dim=1)
    peak_idx = torch.argmax(corr_stack, dim=1, keepdim=True)
    peak_corr = torch.gather(corr_stack, 1, peak_idx).squeeze(1)
    peak_beta = torch.gather(beta_stack, 1, peak_idx).squeeze(1)
    lag_values = torch.tensor(
        TEMPORAL_RESPONSE_LAG_SECONDS,
        device=noisy.device,
        dtype=noisy.dtype,
    ).view((1, len(TEMPORAL_RESPONSE_LAG_SECONDS)) + (1,) * (noisy.ndim - 2))
    lag_norm = torch.gather(
        lag_values.expand((bsz, -1) + tuple(noisy.shape[2:])),
        1,
        peak_idx,
    ).squeeze(1) / max(float(max(TEMPORAL_RESPONSE_LAG_SECONDS)), 1e-6)
    out = {
        "psc_co2_beta_0lag": beta_stack[:, 0],
        "psc_co2_corr_0lag": corr_stack[:, 0],
        "psc_co2_beta_peak": peak_beta,
        "psc_co2_corr_peak": peak_corr,
        "psc_co2_lag_peak_norm": lag_norm,
    }
    return {key: torch.where(mask > 0.5, value, torch.zeros_like(value)) for key, value in out.items()}


def _gpu_lag_vector_left_constant(values, lag_frames: int):
    torch = __import__("torch")
    lag = max(0, int(lag_frames))
    if lag == 0:
        return values
    if lag >= values.shape[1]:
        return values[:, :1].expand_as(values)
    left = values[:, :1].expand(-1, lag)
    return torch.cat([left, values[:, :-lag]], dim=1)


def _batch_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    torch = __import__("torch")
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "metadata":
            out[key] = value
        elif torch.is_tensor(value):
            out[key] = value.to(device=device, dtype=torch.float32)
        else:
            out[key] = value
    return out


def _loader_worker_options(training_cfg: dict[str, Any], num_workers: int) -> dict[str, Any]:
    if int(num_workers) <= 0:
        return {}
    return {
        "persistent_workers": bool(training_cfg.get("persistent_workers", False)),
        "prefetch_factor": int(training_cfg.get("prefetch_factor", 2)),
    }


def _make_dataset(
    config: dict[str, Any],
    sim_root: str | Path,
    split: str,
    samples_per_epoch: int | None,
    stage_cfg: dict[str, Any] | None = None,
) -> OnTheFlyCVRDataset:
    dataset_cfg = dict(config.get("dataset", {}))
    if stage_cfg:
        for key in (
            "tcnr_levels",
            "paradigms",
            "etco2_input",
            "temporal_mode",
            "temporal_window_length",
            "temporal_windows_per_sample",
            "prefer_transition_windows",
            "randomize_artifacts_for_training",
            "etco2_noise_sd_mmhg_range",
            "etco2_drift_sd_mmhg_range",
            "drift_fraction_of_noise_range",
            "motion_spike_probability_options",
            "motion_spike_scale_range",
            "ar1_rho_range",
            "tissue_balanced_patches",
            "model_mismatch_probability",
            "model_mismatch_strength_range",
            "consistency_mode",
            "sampling_strategy",
            "tcnr_probabilities",
        ):
            if key in stage_cfg:
                dataset_cfg[key] = stage_cfg[key]
    dataset_cfg["sim_root"] = Path(sim_root)
    dataset_cfg["split"] = split
    if split in {"val", "validation"}:
        dataset_cfg["sampling_strategy"] = "balanced_grid"
        dataset_cfg["tcnr_probabilities"] = None
    if samples_per_epoch is not None:
        dataset_cfg["samples_per_epoch"] = int(samples_per_epoch)
    stage_loss_cfg = _stage_loss_config(config.get("losses", {}), stage_cfg.get("losses") if stage_cfg else None)
    dataset_cfg["return_consistency_pair"] = (
        float(stage_loss_cfg.get("lambda_view", 0.0)) > 0
    )
    return OnTheFlyCVRDataset(OnTheFlyDatasetConfig(**dataset_cfg))


def _stage_plan(config: dict[str, Any], max_epochs: int | None = None) -> list[tuple[str, dict[str, Any]]]:
    if max_epochs is not None:
        return [("debug_override", {"epochs": int(max_epochs), **config.get("dataset", {})})]
    stages = config.get("stages", {})
    if not stages:
        return [("single_stage", {"epochs": int(config.get("training", {}).get("max_epochs", 1))})]
    return [
        (str(name), dict(stage_cfg))
        for name, stage_cfg in stages.items()
        if bool(dict(stage_cfg).get("enabled", True))
    ]


def _stage_loss_config(base_loss_cfg: dict[str, Any], stage_losses: Any) -> dict[str, Any]:
    if not isinstance(stage_losses, dict):
        return base_loss_cfg
    mapping = {
        "data": "lambda_data",
        "view": "lambda_view",
        "nuisance": "lambda_nuisance",
        "weak": "lambda_weak",
        "transition_alpha": "transition_alpha",
    }
    out = dict(base_loss_cfg)
    for short_name, lambda_name in mapping.items():
        if short_name in stage_losses:
            out[lambda_name] = stage_losses[short_name]
    return out


def _build_model(config: dict[str, Any], in_channels: int):
    model_cfg = config.get("model", {})
    ranges = ParameterRanges(**config.get("parameter_ranges", {}))
    architecture = str(model_cfg.get("architecture", "cnn1d_unet3d_physiology")).lower()
    supported = {"cnn1d_unet3d_physiology", "cnn1d_hybrid_3d"}
    if architecture not in supported:
        raise ValueError(
            "Only model.architecture=cnn1d_unet3d_physiology is supported"
        )
    return CNN1DUNet3DPhysiologyModel(
        in_channels=in_channels,
        base_channels=int(model_cfg.get("base_channels", 24)),
        depth=int(model_cfg.get("depth", 3)),
        norm=str(model_cfg.get("norm", "instance")),
        dropout=float(model_cfg.get("dropout", 0.05)),
        temporal_embedding_channels=int(model_cfg.get("temporal_embedding_channels", 32)),
        temporal_voxel_chunk_size=int(model_cfg.get("temporal_voxel_chunk_size", 16384)),
        parameter_ranges=ranges,
        parameterization=str(model_cfg.get("parameterization", "direct")),
        joint_parameter_head=bool(model_cfg.get("joint_parameter_head", False)),
        initial_parameter_values=dict(model_cfg.get("initial_parameter_values", {})),
    )


def _build_optimizer(config: dict[str, Any], model):
    torch = __import__("torch")
    opt_cfg = config.get("optimizer", {})
    name = str(opt_cfg.get("name", "AdamW")).lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    if name != "adamw":
        raise ValueError("Only AdamW is currently supported")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_scheduler(config: dict[str, Any], optimizer):
    torch = __import__("torch")
    sched_cfg = config.get("scheduler", {})
    if str(sched_cfg.get("name", "ReduceLROnPlateau")).lower() != "reducelronplateau":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(sched_cfg.get("factor", 0.5)),
        patience=int(sched_cfg.get("patience", 10)),
    )


def _loss_weights_from_config(cfg: dict[str, Any]) -> LossWeights:
    return LossWeights(
        lambda_data=float(cfg.get("lambda_data", 1.0)),
        lambda_view=float(cfg.get("lambda_view", 0.0)),
        lambda_nuisance=float(cfg.get("lambda_nuisance", 0.001)),
        lambda_weak=float(cfg.get("lambda_weak", 0.05)),
        transition_alpha=float(cfg.get("transition_alpha", 1.0)),
        student_t_dof=float(cfg.get("student_t_dof", 4.0)),
        data_scale=float(cfg.get("data_scale", 1.0)),
        view_scale=float(cfg.get("view_scale", 1.0)),
        nuisance_scale=float(cfg.get("nuisance_scale", 1.0)),
        weak_scale=float(cfg.get("weak_scale", 1.0)),
    )


def _write_history_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "epoch",
        "stage",
        "stage_epoch",
        "T_mode",
        "learning_rate",
        "epoch_time_seconds",
        "gpu_memory_mb",
    ]
    all_keys = {key for row in rows for key in row.keys()}
    fieldnames = [key for key in preferred if key in all_keys]
    fieldnames.extend(sorted(key for key in all_keys if key not in fieldnames))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_training_history_csv(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | int | str] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                if key in {"epoch", "stage_epoch"}:
                    parsed[key] = int(float(value))
                elif key in {"stage", "T_mode"}:
                    parsed[key] = value
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            if parsed:
                rows.append(parsed)
    return rows


def _best_validation_loss_from_history(
    history: list[dict[str, float | int | str]],
    default: float,
) -> float:
    vals = []
    for row in history:
        try:
            vals.append(float(row["val_total"]))
        except (KeyError, TypeError, ValueError):
            pass
    return min(vals) if vals else default


def _is_stage_boundary_epoch(
    config: dict[str, Any],
    epoch: int,
    max_epochs: int | None = None,
) -> bool:
    if int(epoch) <= 0:
        return False
    total = 0
    boundaries = []
    for _, stage_cfg in _stage_plan(config, max_epochs):
        total += int(stage_cfg["epochs"])
        boundaries.append(total)
    return int(epoch) in set(boundaries[:-1])


def _load_resume_checkpoint(
    out: Path,
    model,
    optimizer,
    scheduler,
    device,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(config.get("training", {}).get("resume_if_available", True)):
        return None
    path = out / "last.pt"
    if not path.exists():
        return None
    torch = __import__("torch")
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload.get("model"))
    training_cfg = config.get("training", {})
    if state:
        incompatible = model.load_state_dict(
            state, strict=not bool(training_cfg.get("allow_partial_resume", False))
        )
        if bool(training_cfg.get("allow_partial_resume", False)):
            print(
                "partial checkpoint load: "
                f"missing={list(incompatible.missing_keys)} "
                f"unexpected={list(incompatible.unexpected_keys)}",
                flush=True,
            )
    if bool(training_cfg.get("resume_weights_only", False)):
        payload = dict(payload)
        payload["epoch"] = 0
        return payload
    opt_state = payload.get("optimizer_state_dict", payload.get("optimizer"))
    if opt_state and optimizer is not None:
        optimizer.load_state_dict(opt_state)
    sched_state = payload.get("scheduler_state_dict")
    if sched_state and scheduler is not None:
        scheduler.load_state_dict(sched_state)
    return payload


def _save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    val_metrics: dict[str, float],
    config: dict[str, Any],
    train_dataset: OnTheFlyCVRDataset,
    seed: int,
    T_mode: str,
    stage: str,
    case_split_metadata: dict[str, Any],
) -> None:
    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metrics={f"val_{k}": float(v) for k, v in val_metrics.items()},
        config=config,
        input_channel_names=list(train_dataset.feature_names),
        normalisation_statistics={},
        parameter_ranges=config.get("parameter_ranges", {}),
        T_mode=T_mode,
        stage=stage,
        random_seed=seed,
        git_commit=_git_commit(),
        case_split_metadata=case_split_metadata,
    )


def _load_case_split_metadata(config: dict[str, Any]) -> dict[str, Any]:
    split_path = config.get("dataset", {}).get("split_json_path")
    if not split_path:
        return {}
    try:
        return json.loads(Path(split_path).read_text(encoding="utf-8"))
    except Exception:
        return {"split_json_path": str(split_path), "status": "unavailable_at_train_start"}


def _write_resolved_config(path: Path, config: dict[str, Any]) -> None:
    try:
        yaml = __import__("yaml")
        text = yaml.safe_dump(config, sort_keys=False)
    except Exception:
        text = json.dumps(config, indent=2, default=str)
    path.write_text(text, encoding="utf-8")


def _write_stage_history(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    stages: dict[str, dict[str, Any]] = {}
    for row in rows:
        stage = str(row["stage"])
        stages.setdefault(stage, {"epochs": 0, "best_val_total": float("inf")})
        stages[stage]["epochs"] += 1
        stages[stage]["best_val_total"] = min(
            float(stages[stage]["best_val_total"]), float(row.get("val_total", float("inf")))
        )
    path.write_text(json.dumps(stages, indent=2), encoding="utf-8")


def _gpu_memory_mb(torch, device) -> float:
    if str(device) != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def _stage_checkpoint_name(stage_name: str) -> str:
    mapping = {
        "stage_1_clean": "stage_1_best.pt",
        "stage_2_physics": "stage_2_best.pt",
        "stage_2_refine": "stage_2_refine_best.pt",
        "stage_3_robust": "stage_3_best.pt",
        "stage_4_stress": "stage_4_best.pt",
    }
    return mapping.get(stage_name, f"{stage_name}_best.pt")


def _format_epoch_log(row: dict[str, float | int | str]) -> str:
    return (
        f"epoch: {int(row['epoch'])} | "
        f"stage: {row['stage']} | "
        f"stage_epoch: {int(row['stage_epoch'])} | "
        f"T_mode: {row['T_mode']} | "
        f"training_total: {_fmt(row.get('train_total'))} | "
        f"validation_total: {_fmt(row.get('val_total'))} | "
        f"training_data: {_fmt(row.get('train_data'))} | "
        f"validation_data: {_fmt(row.get('val_data'))} | "
        f"validation_view: {_fmt(row.get('val_view', 0.0))} | "
        f"validation_nuisance: {_fmt(row.get('val_nuisance'))} | "
        f"validation_residual_structure: {_fmt(row.get('val_residual_structure'))} | "
        f"validation_saturation: {_fmt(row.get('val_saturation'))} | "
        f"mean_CVR: {_fmt(row.get('val_mean_cvr'))} | "
        f"mean_delay: {_fmt(row.get('val_mean_delay'))} | "
        f"mean_T: {_fmt(row.get('val_mean_T'))} | "
        f"lr: {_fmt(row.get('learning_rate'))} | "
        f"time_taken: {_fmt(row.get('epoch_time_seconds'))}s | "
        f"gpu_memory_mb: {_fmt(row.get('gpu_memory_mb'))}"
    )


def _fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.2f}"
    if abs(number) >= 10:
        return f"{number:.3f}"
    if number != 0 and abs(number) < 0.001:
        return f"{number:.3e}"
    return f"{number:.4f}"


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""
