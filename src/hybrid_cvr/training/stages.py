from __future__ import annotations


def stage_loss_names(stage_config: dict) -> set[str]:
    return set(stage_config.get("losses", []))
