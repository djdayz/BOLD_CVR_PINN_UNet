from __future__ import annotations

from dataclasses import dataclass

FORBIDDEN_KEYS = {"gt_cvr", "gt_delay", "gt_t", "supervised", "prior", "tissue_order", "smooth"}


@dataclass(frozen=True)
class LossWeights:
    lambda_data: float = 1.0
    lambda_view: float = 0.0
    lambda_nuisance: float = 0.001
    lambda_weak: float = 0.05
    transition_alpha: float = 1.0
    student_t_dof: float = 4.0
    data_scale: float = 1.0
    view_scale: float = 1.0
    nuisance_scale: float = 1.0
    weak_scale: float = 1.0


def assert_no_supervised_parameter_loss(batch: dict, loss_config: dict | None = None) -> None:
    config_keys = {str(key).lower() for key in (loss_config or {})}
    if config_keys.intersection(FORBIDDEN_KEYS) or any(key.startswith("lambda_supervised") for key in config_keys):
        raise ValueError("Supervised, distribution-prior, tissue-order and smoothness losses are forbidden")
    if batch.get("features_use_gt_maps_as_input"):
        raise ValueError("GT parameter maps are forbidden as model inputs")
    if {str(key).lower() for key in batch}.intersection({"gt_cvr", "gt_delay", "gt_t"}):
        raise ValueError("GT parameter maps must not be present in training batches")
    names = {str(name).lower() for name in batch.get("feature_names", ())}
    if names.intersection({"gt_cvr", "gt_delay", "gt_t"}):
        raise ValueError("GT parameter maps are forbidden as model inputs")


def _masked_weighted_mean(values, mask=None, weights=None):
    torch = __import__("torch")
    weight = torch.ones_like(values)
    if mask is not None:
        m = mask
        while m.ndim < values.ndim:
            m = m.unsqueeze(1)
        weight = weight * m.to(values)
    if weights is not None:
        w = weights
        while w.ndim < values.ndim:
            w = w.unsqueeze(-1)
        weight = weight * w.to(values)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def student_t_reconstruction_loss(y_hat, y_obs, sigma, mask=None, weights=None, dof: float = 4.0):
    torch = __import__("torch")
    scale = sigma.unsqueeze(1).clamp(0.03, 20.0)
    z2 = ((y_obs - y_hat) / scale).square()
    nll = torch.log(scale) + 0.5 * (float(dof) + 1.0) * torch.log1p(z2 / float(dof))
    return _masked_weighted_mean(nll, mask, weights)


def consistency_loss(pred_a: dict, pred_b: dict, mask=None):
    torch = __import__("torch")
    scales = {"cvr": 1.8, "delay": 80.0, "T": 98.0}
    terms = []
    for name, scale in scales.items():
        difference = (pred_a[name] - pred_b[name]) / scale
        log_var_a = pred_a.get("parameter_log_var", {}).get(name)
        log_var_b = pred_b.get("parameter_log_var", {}).get(name)
        if log_var_a is not None and log_var_b is not None:
            variance = (log_var_a.exp() + log_var_b.exp()).clamp_min(1e-5)
            term = 0.5 * difference.square() / variance + 0.5 * torch.log(variance)
        else:
            term = difference.abs()
        terms.append(_masked_weighted_mean(term, mask))
    return sum(terms) / len(terms)


def residual_structure_loss(y_hat, y_obs, etco2, mask=None):
    torch = __import__("torch")
    residual = y_obs - y_hat
    centered = residual - residual.mean(1, keepdim=True)
    lag1 = (centered[:, 1:] * centered[:, :-1]).mean(1)
    variance = centered.square().mean(1).clamp_min(1e-6)
    autocorrelation = lag1.abs() / variance
    u = etco2 - etco2.mean(1, keepdim=True)
    shape = (u.shape[0], u.shape[1]) + (1,) * (residual.ndim - 2)
    covariance = (centered * u.view(shape)).mean(1)
    correlation = covariance.abs() / (centered.std(1) * u.std(1).view((u.shape[0],) + (1,) * (residual.ndim - 2))).clamp_min(1e-5)
    return _masked_weighted_mean(autocorrelation + correlation, mask)


def saturation_loss(pred: dict, mask=None, margin: float = 0.03):
    torch = __import__("torch")
    terms = []
    for raw in pred["raw"].values():
        probability = torch.sigmoid(raw)
        penalty = torch.relu(float(margin) - probability) + torch.relu(probability - (1.0 - float(margin)))
        terms.append(_masked_weighted_mean(penalty, mask))
    return sum(terms) / len(terms)


def nuisance_loss(pred: dict, mask=None):
    coefficients = pred.get("nuisance_coefficients")
    if coefficients is None:
        return pred["cvr"].new_zeros(())
    scaled = coefficients.clone()
    scaled[:, 0] = scaled[:, 0] / 5.0
    scaled[:, 1] = scaled[:, 1] / 2.0
    return _masked_weighted_mean(scaled.square().mean(1), mask)


def transition_weights(batch: dict, alpha: float):
    weights = batch.get("transition_weight")
    if weights is None:
        return None
    return 1.0 + float(alpha) * (weights - 1.0).clamp_min(0.0)


def self_supervised_loss(pred: dict, batch: dict, weights: LossWeights | None = None) -> dict:
    weights = weights or LossWeights()
    assert_no_supervised_parameter_loss(batch)
    mask = batch.get("mask")
    temporal_weights = transition_weights(batch, weights.transition_alpha)
    zero = pred["cvr"].new_zeros(())
    losses = {
        "data": student_t_reconstruction_loss(
            pred["bold_psc_hat"], batch["bold_psc"], pred["sigma_y"], mask,
            temporal_weights, weights.student_t_dof,
        ),
        "nuisance": nuisance_loss(pred, mask) if weights.lambda_nuisance > 0 else zero,
        "residual_structure": (
            residual_structure_loss(
                pred["bold_psc_hat"], batch["bold_psc"], batch["etco2_for_model"], mask
            )
            if weights.lambda_weak > 0
            else zero
        ),
        "saturation": saturation_loss(pred, mask) if weights.lambda_weak > 0 else zero,
    }
    losses["weak"] = losses["residual_structure"] + losses["saturation"]
    losses["total"] = (
        weights.lambda_data * losses["data"] / max(weights.data_scale, 1e-8)
        + weights.lambda_nuisance * losses["nuisance"] / max(weights.nuisance_scale, 1e-8)
        + weights.lambda_weak * losses["weak"] / max(weights.weak_scale, 1e-8)
    )
    return losses
