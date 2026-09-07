from __future__ import annotations

from hybrid_cvr.pinn.delay_interp import interpolate_delayed_1d


def simulate_ode_bold_torch(
    cvr_map,
    delay_map,
    T_map,
    etco2,
    time_grid,
    y0=None,
    mask=None,
    method: str = "exact",
):
    torch = __import__("torch")
    if method not in {"exact", "euler"}:
        raise ValueError("method must be 'exact' or 'euler'")
    if cvr_map.ndim != 3:
        raise ValueError("cvr_map must have shape B,H,W")
    time = time_grid.to(device=cvr_map.device, dtype=cvr_map.dtype)
    delay = delay_map.to(device=cvr_map.device, dtype=cvr_map.dtype)
    tau = T_map.to(device=cvr_map.device, dtype=cvr_map.dtype).clamp_min(1e-4)
    if tau.ndim == 0:
        tau = tau.view(1, 1, 1)
    tau = torch.broadcast_to(tau, cvr_map.shape)
    shifted = interpolate_delayed_1d(etco2, time, delay)
    y_t = torch.zeros_like(cvr_map) if y0 is None else y0.to(device=cvr_map.device, dtype=cvr_map.dtype)
    if mask is not None:
        mask = mask.to(device=cvr_map.device, dtype=cvr_map.dtype)
        y_t = y_t * mask
    outputs = [y_t]
    for t in range(time.numel() - 1):
        dt = (time[t + 1] - time[t]).clamp_min(1e-6)
        if method == "exact":
            a = torch.exp(-dt / tau)
            y_t = a * y_t + (1.0 - a) * cvr_map * shifted[:, t]
        else:
            dydt = (cvr_map * shifted[:, t] - y_t) / tau
            y_t = y_t + dt * dydt
        if mask is not None:
            y_t = y_t * mask
        outputs.append(y_t)
    return torch.stack(outputs, dim=1)
