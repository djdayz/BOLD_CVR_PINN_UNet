from __future__ import annotations

from hybrid_cvr.models.constraints import ParameterRanges
from hybrid_cvr.models.unet3d import UNet3D
from hybrid_cvr.pinn.torch_ode import simulate_ode_bold_torch


def _groups(channels: int) -> int:
    return next((g for g in (8, 4, 2) if channels % g == 0), 1)


class TemporalBlock(__import__("torch").nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        nn = __import__("torch").nn
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class VoxelwiseTemporalTCN(__import__("torch").nn.Module):
    """Dilated shared TCN using complete BOLD and explicit stimulus histories."""

    def __init__(self, channels: int = 24, voxel_chunk_size: int = 2048):
        super().__init__()
        nn = __import__("torch").nn
        self.channels = int(channels)
        self.voxel_chunk_size = int(voxel_chunk_size)
        self.input = nn.Sequential(
            nn.Conv1d(5, channels, 5, padding=2),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[TemporalBlock(channels, d) for d in (1, 2, 4, 8)])
        self.readout = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.SiLU(inplace=True),
                                     nn.AdaptiveAvgPool1d(1))

    def forward(self, bold, etco2, time_grid, valid_time_mask=None):
        torch = __import__("torch")
        if bold.ndim != 5:
            raise ValueError("Expected BOLD PSC with shape B,T,X,Y,Z")
        batch, n_time, *spatial = bold.shape
        u = etco2 if etco2.ndim == 2 else etco2.unsqueeze(0)
        u = u.to(device=bold.device, dtype=bold.dtype)[:, :n_time]
        u = (u - u.mean(1, keepdim=True)) / u.std(1, keepdim=True).clamp_min(1e-4)
        du = torch.zeros_like(u)
        du[:, 1:] = u[:, 1:] - u[:, :-1]
        t = time_grid.to(device=bold.device, dtype=bold.dtype)
        t = 2 * (t - t.min()) / (t.max() - t.min()).clamp_min(1e-6) - 1
        valid = torch.ones((batch, n_time), device=bold.device, dtype=bold.dtype)
        if valid_time_mask is not None:
            valid = valid_time_mask.to(device=bold.device, dtype=bold.dtype)[:, :n_time]
        signal = bold.movedim(1, -1).reshape(batch, -1, n_time)
        signal = (signal - signal.mean(-1, keepdim=True)) / signal.std(-1, keepdim=True).clamp_min(0.05)
        voxels = signal.shape[1]
        x = torch.stack([
            signal, u[:, None].expand(batch, voxels, n_time),
            du[:, None].expand(batch, voxels, n_time),
            t.view(1, 1, -1).expand(batch, voxels, n_time),
            valid[:, None].expand(batch, voxels, n_time),
        ], dim=2).reshape(-1, 5, n_time)
        def encode(chunk):
            return self.readout(self.blocks(self.input(chunk))).squeeze(-1)

        embeddings = []
        checkpoint = __import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint
        for start in range(0, x.shape[0], self.voxel_chunk_size):
            chunk = x[start : start + self.voxel_chunk_size]
            if self.training and torch.is_grad_enabled():
                encoded = checkpoint(encode, chunk, use_reentrant=False)
            else:
                encoded = encode(chunk)
            embeddings.append(encoded)
        x = torch.cat(embeddings, dim=0)
        return x.reshape(batch, *spatial, self.channels).movedim(-1, 1).contiguous()


class SpatialHead(__import__("torch").nn.Module):
    def __init__(self, in_channels: int, hidden: int, norm: str):
        super().__init__()
        nn = __import__("torch").nn
        normalise = (nn.InstanceNorm3d(hidden, affine=True) if norm == "instance"
                     else nn.GroupNorm(_groups(hidden), hidden))
        self.net = nn.Sequential(nn.Conv3d(in_channels, hidden, 3, padding=1), normalise,
                                 nn.SiLU(inplace=True), nn.Conv3d(hidden, 1, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class TemporalHybridUNetPINN(__import__("torch").nn.Module):
    """Full-time TCN + 3D U-Net + separate coarse and full-resolution heads."""

    def __init__(self, in_channels: int, base_channels: int = 24, depth: int = 2,
                 norm: str = "instance", dropout: float = 0.05,
                 temporal_embedding_channels: int = 24,
                 parameter_ranges: ParameterRanges | None = None,
                 parameterization: str = "direct", **_: object):
        super().__init__()
        if parameterization != "direct":
            raise ValueError("Principal model requires direct bounded parameter heads")
        self.parameter_ranges = parameter_ranges or ParameterRanges()
        self.parameterization, self.T_mode = parameterization, "voxelwise"
        self.temporal_encoder = VoxelwiseTemporalTCN(temporal_embedding_channels)
        self.spatial_channels = int(base_channels)
        self.spatial_unet = UNet3D(
            in_channels=in_channels + temporal_embedding_channels, out_channels=base_channels,
            base_channels=base_channels, depth=depth, norm=norm, dropout=dropout,
        )
        self.coarse_heads = __import__("torch").nn.ModuleDict({
            p: SpatialHead(base_channels, base_channels, norm) for p in ("cvr", "delay", "T")
        })
        refine_in = base_channels + temporal_embedding_channels + 1
        self.refine_heads = __import__("torch").nn.ModuleDict({
            p: SpatialHead(refine_in, max(8, base_channels // 2), norm) for p in ("cvr", "delay", "T")
        })
        self.uncertainty_heads = __import__("torch").nn.ModuleDict({
            p: SpatialHead(base_channels, max(8, base_channels // 2), norm) for p in ("cvr", "delay", "T")
        })
        self.observation_uncertainty_head = SpatialHead(base_channels, max(8, base_channels // 2), norm)

    def forward(self, features, etco2=None, time_grid=None, mask=None, tissue_maps=None,
                bold_psc=None, valid_time_mask=None):
        torch = __import__("torch")
        if bold_psc is None or etco2 is None or time_grid is None:
            raise ValueError("bold_psc, measured etco2 and time_grid are required")
        temporal = self.temporal_encoder(bold_psc, etco2, time_grid, valid_time_mask)
        shared = self.spatial_unet(torch.cat([features, temporal], dim=1))
        limits = {"cvr": (self.parameter_ranges.cvr_min, self.parameter_ranges.cvr_max),
                  "delay": (self.parameter_ranges.delay_min, self.parameter_ranges.delay_max),
                  "T": (self.parameter_ranges.T_min, self.parameter_ranges.T_max)}
        raw, coarse, values = {}, {}, {}
        for name in ("cvr", "delay", "T"):
            coarse_raw = self.coarse_heads[name](shared)
            refine = self.refine_heads[name](torch.cat([shared, temporal, coarse_raw[:, None]], 1))
            raw[name] = coarse_raw + 0.75 * torch.tanh(refine)
            lo, hi = limits[name]
            coarse[name] = lo + (hi - lo) * torch.sigmoid(coarse_raw)
            values[name] = lo + (hi - lo) * torch.sigmoid(raw[name])
        physiology = simulate_ode_bold_torch(values["cvr"], values["delay"], values["T"],
                                              etco2, time_grid, mask=mask, method="exact")
        reconstruction, nuisance = self._profile_nuisance(physiology, bold_psc, time_grid, mask)
        log_vars = {p: self.uncertainty_heads[p](shared).clamp(-8, 8) for p in values}
        log_scale = self.observation_uncertainty_head(shared).clamp(-5, 3)
        parameter_sigma = {p: torch.exp(0.5 * log_vars[p]) for p in values}
        combined_sigma = (
            parameter_sigma["cvr"] / 1.8
            + parameter_sigma["delay"] / 80.0
            + parameter_sigma["T"] / 98.0
        ) / 3.0
        return {
            "shared": shared, "temporal_embedding": temporal, "raw": raw, "coarse": coarse,
            **values, "bold_psc_physiology": physiology, "bold_psc_hat": reconstruction,
            "nuisance_coefficients": nuisance, "observation_log_scale": log_scale,
            "sigma_y": torch.nn.functional.softplus(log_scale) + 0.03,
            "parameter_log_var": log_vars,
            "sigma": combined_sigma,
            **{f"sigma_{p}": parameter_sigma[p] for p in values},
        }

    @staticmethod
    def _profile_nuisance(physiology, observed, time_grid, mask):
        time = time_grid.to(device=observed.device, dtype=observed.dtype)
        t = 2 * (time - time.min()) / (time.max() - time.min()).clamp_min(1e-6) - 1
        residual = observed - physiology
        intercept = residual.mean(1)
        shape = (1, -1) + (1,) * (observed.ndim - 2)
        slope = (residual * t.view(shape)).mean(1) / t.square().mean().clamp_min(1e-6)
        nuisance = intercept[:, None] + slope[:, None] * t.view(shape)
        if mask is not None:
            nuisance = nuisance * mask[:, None]
        return physiology + nuisance, __import__("torch").stack([intercept, slope], 1)
