from __future__ import annotations


def interpolate_delayed_1d(etco2, time_grid, delay_map):
    torch = __import__("torch")
    if etco2.ndim == 1:
        etco2 = etco2.unsqueeze(0)
    if time_grid.ndim != 1:
        raise ValueError("time_grid must be 1D")
    batch = delay_map.shape[0]
    if etco2.shape[0] == 1 and batch > 1:
        etco2 = etco2.expand(batch, -1)
    if etco2.shape[0] != batch:
        raise ValueError("etco2 batch must be 1 or match delay_map batch")
    n_time = time_grid.numel()
    if n_time == 0:
        raise ValueError("time_grid must contain at least one time point")
    time = time_grid.to(device=delay_map.device, dtype=delay_map.dtype)
    source = etco2.to(device=delay_map.device, dtype=delay_map.dtype)
    if n_time == 1:
        query = time.view(1, 1, 1, 1) - delay_map.unsqueeze(1)
        valid = (query >= time[0]) & (query <= time[0])
        return torch.where(valid, source[:, 0].view(batch, 1, 1, 1), torch.zeros_like(query))

    dt = (time[-1] - time[0]) / max(n_time - 1, 1)
    query = time.view(1, n_time, 1, 1) - delay_map.unsqueeze(1)
    idx_float = (query - time[0]) / dt.clamp_min(1e-6)
    idx0_raw = torch.floor(idx_float)
    alpha = (idx_float - idx0_raw).clamp(0.0, 1.0)
    valid = (idx_float >= 0.0) & (idx_float <= float(n_time - 1))
    idx0 = idx0_raw.clamp(0, n_time - 1)
    idx1 = (idx0_raw + 1).clamp(0, n_time - 1)
    flat0 = idx0.long().reshape(batch, -1)
    flat1 = idx1.long().reshape(batch, -1)
    gather_source = source
    y0 = torch.gather(gather_source, 1, flat0).reshape_as(idx_float)
    y1 = torch.gather(gather_source, 1, flat1).reshape_as(idx_float)
    interpolated = y0 * (1.0 - alpha) + y1 * alpha
    return torch.where(valid, interpolated, torch.zeros_like(interpolated))
