from __future__ import annotations


def sliding_window_parameter_inference(
    model,
    features,
    bold_psc,
    etco2,
    time_grid,
    mask,
    tissue_maps=None,
    patch_size=(32, 32, 24),
    overlap: float = 0.5,
):
    """Blend overlapping temporal-model patches into seamless full-volume maps."""
    torch = __import__("torch")
    if features.ndim != 5 or bold_psc.ndim != 5:
        raise ValueError("Expected features B,C,X,Y,Z and bold_psc B,T,X,Y,Z")
    if features.shape[0] != 1:
        raise ValueError("Sliding-window inference currently supports batch size one")
    spatial = tuple(int(v) for v in features.shape[-3:])
    patch = tuple(min(int(p), spatial[i]) for i, p in enumerate(patch_size))
    stride = tuple(max(1, int(round(p * (1.0 - float(overlap))))) for p in patch)
    starts = [_axis_starts(spatial[i], patch[i], stride[i]) for i in range(3)]
    weight = _blend_weight(patch, features.device, features.dtype)
    output_keys = ("cvr", "delay", "T", "sigma", "sigma_cvr", "sigma_delay", "sigma_T")
    accum = {key: torch.zeros(spatial, device=features.device, dtype=features.dtype) for key in output_keys}
    denominator = torch.zeros(spatial, device=features.device, dtype=features.dtype)
    for x in starts[0]:
        for y in starts[1]:
            for z in starts[2]:
                sl = (slice(x, x + patch[0]), slice(y, y + patch[1]), slice(z, z + patch[2]))
                patch_mask = mask[(slice(None),) + sl]
                if not bool((patch_mask > 0.5).any()):
                    continue
                pred = model(
                    features[(slice(None), slice(None)) + sl],
                    etco2=etco2,
                    time_grid=time_grid,
                    mask=patch_mask,
                    tissue_maps=tissue_maps[(slice(None), slice(None)) + sl]
                    if tissue_maps is not None
                    else None,
                    bold_psc=bold_psc[(slice(None), slice(None)) + sl],
                )
                local_weight = weight * patch_mask[0]
                for key in accum:
                    accum[key][sl] += pred[key][0] * local_weight
                denominator[sl] += local_weight
    valid = denominator > 0
    result = {
        key: torch.where(valid, value / denominator.clamp_min(1e-6), torch.zeros_like(value))
        for key, value in accum.items()
    }
    result["coverage"] = valid
    return result


def _axis_starts(length: int, patch: int, stride: int) -> list[int]:
    if patch >= length:
        return [0]
    values = list(range(0, length - patch + 1, stride))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values


def _blend_weight(shape, device, dtype):
    torch = __import__("torch")
    axes = []
    for length in shape:
        coordinate = torch.linspace(-1.0, 1.0, int(length), device=device, dtype=dtype)
        axes.append((1.0 - coordinate.abs()).clamp_min(0.1))
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
