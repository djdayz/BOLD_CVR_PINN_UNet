from __future__ import annotations


def validation_selection_policy() -> str:
    return "Select checkpoints by self-supervised validation loss only, never GT_CVR/GT_delay/GT_T."
