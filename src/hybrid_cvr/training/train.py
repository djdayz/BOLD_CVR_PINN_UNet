from __future__ import annotations

from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time
from typing import Any

from hybrid_cvr.models.constraints import ParameterRanges
from hybrid_cvr.models.hybrid_unet_pinn import HybridUNetPINN
from hybrid_cvr.pinn.losses import (
    LossWeights,
    assert_no_supervised_parameter_loss,
    consistency_loss,
    self_supervised_loss,
)
from hybrid_cvr.training.checkpointing import save_checkpoint
from hybrid_cvr.training.on_the_fly_dataset import OnTheFlyCVRDataset, OnTheFlyDatasetConfig


def validate_training_batch_contract(batch: dict, loss_config: dict | None = None) -> None:
    assert_no_supervised_parameter_loss(batch, loss_config)


def validate_training_config(config: dict) -> None:
    training = config.get("training", {})
    features = config.get("features", {})
    if bool(training.get("use_supervised_parameter_loss", False)):
        raise ValueError("training.use_supervised_parameter_loss=true is forbidden.")
    if bool(features.get("use_gt_maps_as_input", False)):
        raise ValueError("features.use_gt_maps_as_input=true is forbidden.")
    losses = config.get("losses", {})
    assert_no_supervised_parameter_loss({}, losses)


def train_unet_pinn(
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
                )
                if validate
                else train_metrics
            )
            val_total = float(val_metrics["total"])
            if scheduler is not None:
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
            if val_total < best_loss:
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
            if val_total < stage_best:
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
) -> dict[str, float]:
    torch = __import__("torch")
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        time_grid = batch["time_grid"][0] if batch["time_grid"].ndim == 2 else batch["time_grid"]
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
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
                    tissue_maps=batch.get("tissue_maps"),
                )
                loss_dict = self_supervised_loss(pred, {**batch, "time_grid": time_grid}, loss_weights)
                if loss_weights.lambda_consistency > 0 and "consistency_features" in batch:
                    pred_pair = model(
                        batch["consistency_features"],
                        mask=batch["mask"],
                        tissue_maps=batch.get("tissue_maps"),
                    )
                    loss_dict["consistency"] = consistency_loss(
                        pred,
                        pred_pair,
                        mask=batch["mask"],
                    )
                    loss_dict["total"] = (
                        loss_dict["total"]
                        + loss_weights.lambda_consistency * loss_dict["consistency"]
                    )
                loss = loss_dict["total"]
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        for key in ("cvr", "delay", "T", "sigma"):
            if key in pred:
                totals[f"mean_{key}"] = totals.get(f"mean_{key}", 0.0) + float(
                    pred[key].detach().mean().cpu()
                )
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def _batch_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    torch = __import__("torch")
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key in {"metadata", "gt_cvr", "gt_delay", "gt_T"}:
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
        ):
            if key in stage_cfg:
                dataset_cfg[key] = stage_cfg[key]
    dataset_cfg["sim_root"] = Path(sim_root)
    dataset_cfg["split"] = split
    if samples_per_epoch is not None:
        dataset_cfg["samples_per_epoch"] = int(samples_per_epoch)
    stage_loss_cfg = _stage_loss_config(config.get("losses", {}), stage_cfg.get("losses") if stage_cfg else None)
    dataset_cfg["return_consistency_pair"] = float(stage_loss_cfg.get("lambda_consistency", 0.0)) > 0
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
        "recon": "lambda_recon",
        "residual": "lambda_residual",
        "uncertainty": "lambda_uncertainty",
        "smooth": "lambda_smooth",
        "prior": "lambda_prior",
        "consistency": "lambda_consistency",
    }
    out = dict(base_loss_cfg)
    for short_name, lambda_name in mapping.items():
        if short_name in stage_losses:
            out[lambda_name] = stage_losses[short_name]
    return out


def _build_model(config: dict[str, Any], in_channels: int):
    model_cfg = config.get("model", {})
    ranges = ParameterRanges(**config.get("parameter_ranges", {}))
    return HybridUNetPINN(
        in_channels=in_channels,
        base_channels=int(model_cfg.get("base_channels", 32)),
        depth=int(model_cfg.get("depth", 3)),
        norm=str(model_cfg.get("norm", "instance")),
        dropout=float(model_cfg.get("dropout", 0.05)),
        parameter_ranges=ranges,
        T_mode=str(model_cfg.get("T_mode", "tissuewise")),
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
        lambda_recon=float(cfg.get("lambda_recon", 1.0)),
        lambda_residual=float(cfg.get("lambda_residual", 0.25)),
        lambda_uncertainty=float(cfg.get("lambda_uncertainty", 0.1)),
        lambda_smooth=float(cfg.get("lambda_smooth", 0.05)),
        lambda_prior=float(cfg.get("lambda_prior", 0.01)),
        lambda_consistency=float(cfg.get("lambda_consistency", 0.0)),
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
    if state:
        model.load_state_dict(state)
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
        f"training_recon: {_fmt(row.get('train_recon'))} | "
        f"validation_recon: {_fmt(row.get('val_recon'))} | "
        f"validation_residual: {_fmt(row.get('val_residual'))} | "
        f"validation_uncertainty: {_fmt(row.get('val_uncertainty'))} | "
        f"training_consistency: {_fmt(row.get('train_consistency', 0.0))} | "
        f"validation_consistency: {_fmt(row.get('val_consistency', 0.0))} | "
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
