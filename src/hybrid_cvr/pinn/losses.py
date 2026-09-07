from __future__ import annotations

from dataclasses import dataclass

from hybrid_cvr.pinn.delay_interp import interpolate_delayed_1d

FORBIDDEN_SUPERVISED_KEYS = {"gt_cvr", "gt_delay", "gt_T", "GT_CVR", "GT_delay", "GT_T"}
FORBIDDEN_FEATURE_NAMES = {key.lower() for key in FORBIDDEN_SUPERVISED_KEYS}


@dataclass(frozen=True)
class LossWeights:
    lambda_recon: float = 1.0
    lambda_residual: float = 0.25
    lambda_uncertainty: float = 0.1
    lambda_smooth: float = 0.05
    lambda_prior: float = 0.01
    lambda_consistency: float = 0.0


def _masked_mean(x, mask=None):
    if mask is None:
        return x.mean()
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(1)
    weighted = x * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def _masked_std(x, mask=None):
    torch = __import__("torch")
    if mask is None:
        return x.std()
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=x.device, dtype=x.dtype)
    denom = mask.sum().clamp_min(1.0)
    mean = (x * mask).sum() / denom
    var = (((x - mean) ** 2) * mask).sum() / denom
    return torch.sqrt(var.clamp_min(1e-12))


def reconstruction_loss(y_hat, y_obs, mask=None, weights=None, huber_delta: float | None = 1.0):
    torch = __import__("torch")
    err = y_hat - y_obs
    if huber_delta is None:
        loss = err * err
    else:
        loss = torch.nn.functional.huber_loss(y_hat, y_obs, reduction="none", delta=huber_delta)
    if weights is not None:
        loss = loss * weights
    return _masked_mean(loss, mask)


def combine_loss_weights(local_weights=None, transition_weight=None, target_ndim: int = 4):
    torch = __import__("torch")
    if local_weights is None and transition_weight is None:
        return None
    weights = local_weights
    if transition_weight is not None:
        tw = transition_weight
        if tw.ndim == 1:
            tw = tw.unsqueeze(0)
        while tw.ndim < target_ndim:
            tw = tw.unsqueeze(-1)
        weights = tw if weights is None else weights * tw.to(weights.device)
    if weights is None:
        return None
    return torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)


def ode_residual_loss(cvr, delay, T, y_obs_smooth, etco2, time_grid, mask=None, weights=None):
    torch = __import__("torch")
    shifted = interpolate_delayed_1d(etco2, time_grid.to(cvr.device), delay)
    dt = torch.diff(time_grid.to(cvr.device)).view(1, -1, 1, 1).clamp_min(1e-6)
    dy = (y_obs_smooth[:, 1:] - y_obs_smooth[:, :-1]) / dt
    rhs = (cvr.unsqueeze(1) * shifted[:, :-1] - y_obs_smooth[:, :-1]) / T.clamp_min(1e-4).unsqueeze(1)
    residual = dy - rhs
    loss = torch.nn.functional.huber_loss(residual, torch.zeros_like(residual), reduction="none")
    if weights is not None:
        loss = loss * weights[:, :-1].to(loss.device)
    return _masked_mean(loss, mask)


def heteroscedastic_uncertainty_loss(y_hat, y_obs, sigma, mask=None):
    torch = __import__("torch")
    sigma_t = sigma.unsqueeze(1).clamp(0.05, 10.0)
    scaled_residual = (y_hat - y_obs) / sigma_t
    loss = torch.nn.functional.huber_loss(
        scaled_residual,
        torch.zeros_like(scaled_residual),
        reduction="none",
        delta=1.0,
    )
    loss = loss + 0.01 * sigma_t
    return _masked_mean(loss, mask)


def total_variation_loss(x, edge_weight=None):
    dx = (x[..., 1:, :] - x[..., :-1, :]).abs()
    dy = (x[..., :, 1:] - x[..., :, :-1]).abs()
    if edge_weight is not None:
        edge_weight = edge_weight.to(device=x.device, dtype=x.dtype)
        wx = 1.0 - edge_weight[..., 1:, :].clamp(0, 1)
        wy = 1.0 - edge_weight[..., :, 1:].clamp(0, 1)
        dx = dx * wx
        dy = dy * wy
    return dx.mean() + dy.mean()


def physiological_prior_loss(cvr, delay, T, mask=None):
    # Outputs are range-constrained; this discourages collapse and implausibly tiny T.
    torch = __import__("torch")
    spread = torch.stack(
        [
            _masked_std(cvr, mask) / 1.0,
            _masked_std(delay, mask) / 80.0,
            _masked_std(T, mask) / 100.0,
        ]
    )
    collapse_penalty = torch.relu(0.01 - spread).mean()
    T_low = _masked_mean((torch.relu(8.0 - T) / 8.0) ** 2, mask)
    T_high = _masked_mean((torch.relu(T - 70.0) / 30.0) ** 2, mask)
    delay_high = _masked_mean((torch.relu(delay - 75.0) / 10.0) ** 2, mask)
    cvr_high = _masked_mean((torch.relu(cvr - 1.4) / 0.4) ** 2, mask)
    return collapse_penalty + 0.5 * T_low + 0.1 * T_high + 0.05 * delay_high + 0.05 * cvr_high


def consistency_loss(
    pred_a: dict,
    pred_b: dict,
    mask=None,
    keys: tuple[str, ...] = ("cvr", "delay", "T"),
):
    torch = __import__("torch")
    losses = []
    for key in keys:
        if key in pred_a and key in pred_b:
            loss = torch.nn.functional.huber_loss(pred_a[key], pred_b[key], reduction="none")
            losses.append(_masked_mean(loss, mask))
    if not losses:
        device = None
        for pred in (pred_a, pred_b):
            for value in pred.values():
                if torch.is_tensor(value):
                    device = value.device
                    break
            if device is not None:
                break
        return torch.tensor(0.0, device=device)
    return sum(losses) / len(losses)


def assert_no_supervised_parameter_loss(batch: dict, loss_config: dict | None = None) -> None:
    if loss_config and any(str(k).lower().startswith("supervised") for k in loss_config):
        raise ValueError("Supervised parameter-map losses are forbidden for training.")
    if batch.get("features_use_gt_maps_as_input"):
        raise ValueError("GT parameter maps are forbidden as model input features.")
    feature_names = batch.get("feature_names")
    if feature_names and {str(name).lower() for name in feature_names}.intersection(FORBIDDEN_FEATURE_NAMES):
        raise ValueError("GT parameter maps are forbidden as model input features.")


def self_supervised_loss(pred: dict, batch: dict, weights: LossWeights | None = None) -> dict:
    torch = __import__("torch")
    weights = weights or LossWeights()
    assert_no_supervised_parameter_loss(batch)
    mask = batch.get("mask")
    local_weights = combine_loss_weights(
        batch.get("local_uncertainty_weights"), batch.get("transition_weight")
    )
    etco2 = batch.get("etco2_for_model", batch.get("etco2"))
    if etco2 is None:
        raise KeyError("Batch must contain etco2_for_model or etco2")
    losses = {}
    losses["recon"] = reconstruction_loss(pred["bold_psc_hat"], batch["bold_psc"], mask, local_weights)
    losses["residual"] = ode_residual_loss(
        pred["cvr"],
        pred["delay"],
        pred["T"],
        batch["bold_psc"],
        etco2,
        batch["time_grid"],
        mask,
        local_weights,
    )
    losses["uncertainty"] = heteroscedastic_uncertainty_loss(
        pred["bold_psc_hat"], batch["bold_psc"], pred["sigma"], mask
    )
    edge_weight = edge_suppression_weight(batch.get("boundary_uncertainty"), batch.get("vessel_likelihood"))
    losses["smooth"] = (
        total_variation_loss(pred["cvr"], edge_weight)
        + total_variation_loss(pred["delay"], edge_weight)
        + total_variation_loss(pred["T"], edge_weight)
    )
    losses["prior"] = physiological_prior_loss(pred["cvr"], pred["delay"], pred["T"], mask)
    total = (
        weights.lambda_recon * losses["recon"]
        + weights.lambda_residual * losses["residual"]
        + weights.lambda_uncertainty * losses["uncertainty"]
        + weights.lambda_smooth * losses["smooth"]
        + weights.lambda_prior * losses["prior"]
    )
    losses["total"] = total if torch.is_tensor(total) else torch.as_tensor(total)
    return losses


def edge_suppression_weight(boundary_uncertainty=None, vessel_likelihood=None):
    torch = __import__("torch")
    if boundary_uncertainty is None and vessel_likelihood is None:
        return None
    if boundary_uncertainty is None:
        base = torch.zeros_like(vessel_likelihood)
    else:
        base = boundary_uncertainty
    if vessel_likelihood is None:
        vessel = torch.zeros_like(base)
    else:
        vessel = vessel_likelihood.to(device=base.device, dtype=base.dtype)
    smooth_weight = torch.exp(-3.0 * base.clamp(0, 1)) * torch.exp(-2.0 * vessel.clamp(0, 1))
    return 1.0 - smooth_weight
