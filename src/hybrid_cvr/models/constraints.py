from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterRanges:
    cvr_min: float = -0.4
    cvr_max: float = 1.8
    delay_min: float = 0.0
    delay_max: float = 80.0
    T_min: float = 2.0
    T_max: float = 100.0
    sigma_eps: float = 1e-4


def constrain_parameter_maps(raw: Any, ranges: ParameterRanges | None = None) -> dict[str, Any]:
    torch = __import__("torch")
    ranges = ranges or ParameterRanges()
    if raw.shape[1] != 4:
        raise ValueError(f"Expected raw output with 4 channels, got {raw.shape}")
    raw_cvr, raw_delay, raw_T, raw_log_sigma = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    cvr = ranges.cvr_min + (ranges.cvr_max - ranges.cvr_min) * torch.sigmoid(raw_cvr)
    delay = ranges.delay_min + (ranges.delay_max - ranges.delay_min) * torch.sigmoid(raw_delay)
    T = ranges.T_min + (ranges.T_max - ranges.T_min) * torch.sigmoid(raw_T)
    sigma = torch.nn.functional.softplus(raw_log_sigma) + ranges.sigma_eps
    return {"cvr": cvr, "delay": delay, "T": T, "sigma": sigma}
