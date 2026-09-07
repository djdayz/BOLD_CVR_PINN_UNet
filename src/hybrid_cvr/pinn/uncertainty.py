from __future__ import annotations


def local_temporal_uncertainty(bold_psc, kernel_size: int = 3, eps: float = 1e-4):
    torch = __import__("torch")
    pad = kernel_size // 2
    if bold_psc.ndim != 4:
        raise ValueError("bold_psc must have shape B,T,H,W")
    b, t, h, w = bold_psc.shape
    x = bold_psc.reshape(b * t, 1, h, w)
    mean = torch.nn.functional.avg_pool2d(x, kernel_size, stride=1, padding=pad)
    mean_sq = torch.nn.functional.avg_pool2d(x * x, kernel_size, stride=1, padding=pad)
    var = (mean_sq - mean * mean).clamp_min(0.0)
    return torch.sqrt(var + eps).reshape(b, t, h, w)


def inverse_uncertainty_weight(local_sigma, eps: float = 1e-4):
    return 1.0 / (local_sigma + eps)
