from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    epoch: int = 0,
    metrics: dict | None = None,
    scheduler=None,
    config: dict[str, Any] | None = None,
    input_channel_names: list[str] | tuple[str, ...] | None = None,
    normalisation_statistics: dict[str, Any] | None = None,
    parameter_ranges: dict[str, Any] | None = None,
    T_mode: str | None = None,
    stage: str | None = None,
    random_seed: int | None = None,
    git_commit: str | None = None,
    case_split_metadata: dict[str, Any] | None = None,
) -> Path:
    torch = __import__("torch")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model": model.state_dict(),
        "epoch": int(epoch),
        "metrics": metrics or {},
        "config": config or {},
        "input_channel_names": list(input_channel_names or []),
        "normalisation_statistics": normalisation_statistics or {},
        "parameter_ranges": parameter_ranges or {},
        "T_mode": T_mode,
        "training_stage": stage,
        "random_seed": random_seed,
        "git_commit": git_commit,
        "case_split_metadata": case_split_metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
    return path
