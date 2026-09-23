from __future__ import annotations

from hybrid_cvr.models.constraints import ParameterRanges
from hybrid_cvr.models.unet3d import UNet3D
from hybrid_cvr.physiology.torch_ode import simulate_ode_bold_torch


def _logit(probability: float) -> float:
    import math

    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


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


class VoxelwiseCNN1D(__import__("torch").nn.Module):
    """Shared non-causal 1D CNN using complete BOLD and stimulus histories."""

    def __init__(self, channels: int = 24, voxel_chunk_size: int = 2048):
        super().__init__()
        nn = __import__("torch").nn
        self.channels = int(channels)
        self.voxel_chunk_size = int(voxel_chunk_size)
        self.input = nn.Sequential(
            nn.Conv1d(5, channels, 5, padding=2),
            nn.GroupNorm(_groups(channels), channels), nn.SiLU(inplace=True),
        )
        # At TR=1.55 s this spans roughly +/-195 s, covering the configured
        # 80 s delay range and slow first-order responses.
        self.blocks = nn.Sequential(*[TemporalBlock(channels, d) for d in (1, 2, 4, 8, 16, 32)])
        self.readout = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.SiLU(inplace=True),
                                     nn.AdaptiveAvgPool1d(1))

    def forward(self, bold, etco2, time_grid, valid_time_mask=None, spatial_mask=None):
        torch = __import__("torch")
        if bold.ndim != 5:
            raise ValueError("Expected BOLD PSC with shape B,T,X,Y,Z")
        batch, n_time, *spatial = bold.shape
        raw_u = etco2 if etco2.ndim == 2 else etco2.unsqueeze(0)
        raw_u = raw_u.to(device=bold.device, dtype=bold.dtype)[:, :n_time]
        u = (raw_u - raw_u.mean(1, keepdim=True)) / raw_u.std(1, keepdim=True).clamp_min(1e-4)
        du = torch.zeros_like(u)
        du[:, 1:] = u[:, 1:] - u[:, :-1]
        t = time_grid.to(device=bold.device, dtype=bold.dtype)
        t = 2 * (t - t.min()) / (t.max() - t.min()).clamp_min(1e-6) - 1
        valid = torch.ones((batch, n_time), device=bold.device, dtype=bold.dtype)
        if valid_time_mask is not None:
            valid = valid_time_mask.to(device=bold.device, dtype=bold.dtype)[:, :n_time]
        signal = bold.movedim(1, -1).reshape(batch, -1, n_time)
        voxels = signal.shape[1]
        if spatial_mask is None:
            active = torch.ones(batch * voxels, device=bold.device, dtype=torch.bool)
        else:
            active = spatial_mask.reshape(batch, voxels).to(device=bold.device) > 0.5
            active = active.reshape(-1)
        sample_index = torch.arange(batch, device=bold.device).repeat_interleave(voxels)[active]
        signal = signal.reshape(batch * voxels, n_time)[active]
        signal = (signal - signal.mean(-1, keepdim=True)) / signal.std(-1, keepdim=True).clamp_min(0.05)
        active_voxels = signal.shape[0]
        x = torch.stack([
            signal,
            u[sample_index],
            du[sample_index],
            t.view(1, -1).expand(active_voxels, n_time),
            valid[sample_index],
        ], dim=1)
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
        encoded = torch.cat(embeddings, dim=0)
        full = torch.zeros(
            (batch * voxels, self.channels), device=bold.device, dtype=encoded.dtype
        )
        full[active] = encoded
        return full.reshape(batch, *spatial, self.channels).movedim(-1, 1).contiguous()


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


class JointPhysiologyHead(__import__("torch").nn.Module):
    """Predict all physiological parameters and uncertainties from one latent field."""

    def __init__(self, in_channels: int, hidden: int, norm: str):
        super().__init__()
        nn = __import__("torch").nn
        normalise = (
            nn.InstanceNorm3d(hidden, affine=True)
            if norm == "instance"
            else nn.GroupNorm(_groups(hidden), hidden)
        )
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 3, padding=1),
            normalise,
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, 6, 1),
        )

    def forward(self, x):
        return self.net(x)


class CVRResidualHead(__import__("torch").nn.Module):
    """Locally refine physically profiled CVR without normalising its amplitude."""

    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        nn = __import__("torch").nn
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, 2, 1),
        )
        # Begin exactly at the physical profile. The correction learns first;
        # the reliability gate opens only where reconstruction supports it.
        with __import__("torch").no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias[0] = 0.0
            self.net[-1].bias[1] = -2.0

    def forward(self, x):
        return self.net(x)


class CNN1DUNet3DPhysiologyModel(__import__("torch").nn.Module):
    """Full-time 1D CNN + 3D U-Net with differentiable profiled CVR."""

    def __init__(self, in_channels: int, base_channels: int = 24, depth: int = 2,
                 norm: str = "instance", dropout: float = 0.05,
                 temporal_embedding_channels: int = 24,
                 parameter_ranges: ParameterRanges | None = None,
                 parameterization: str = "direct", joint_parameter_head: bool = False,
                 temporal_voxel_chunk_size: int = 16384,
                 initial_parameter_values: dict[str, float] | None = None,
                 cvr_residual_head: bool = True,
                 cvr_residual_channels: int = 24,
                 cvr_log_correction_limit: float = 0.6931471805599453,
                 **_: object):
        super().__init__()
        if parameterization != "direct":
            raise ValueError("Principal model requires direct bounded parameter heads")
        self.parameter_ranges = parameter_ranges or ParameterRanges()
        self.parameterization, self.T_mode = parameterization, "voxelwise"
        self.joint_parameter_head = bool(joint_parameter_head)
        self.use_cvr_residual_head = bool(cvr_residual_head)
        self.cvr_log_correction_limit = float(cvr_log_correction_limit)
        self.temporal_encoder = VoxelwiseCNN1D(
            temporal_embedding_channels, voxel_chunk_size=temporal_voxel_chunk_size
        )
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
        self.joint_head = JointPhysiologyHead(base_channels, base_channels, norm)
        # Five physical-unit maps: profiled CVR, covariance, response RMS,
        # observed-BOLD RMS and residual RMS. No normalization is used here.
        cvr_inputs = base_channels + temporal_embedding_channels + 5
        self.cvr_residual_head = CVRResidualHead(
            cvr_inputs, max(8, int(cvr_residual_channels))
        )
        self._initialize_joint_head(initial_parameter_values or {})

    def _initialize_joint_head(self, initial_values: dict[str, float]) -> None:
        """Start direct heads at broad physiological values, not range midpoints."""
        torch = __import__("torch")
        defaults = {"delay": 25.0, "T": 25.0}
        limits = {
            "delay": (self.parameter_ranges.delay_min, self.parameter_ranges.delay_max),
            "T": (self.parameter_ranges.T_min, self.parameter_ranges.T_max),
        }
        final = self.joint_head.net[-1]
        with torch.no_grad():
            final.weight[:2].normal_(mean=0.0, std=1e-3)
            for index, name in enumerate(("delay", "T")):
                lo, hi = limits[name]
                value = min(max(float(initial_values.get(name, defaults[name])), lo), hi)
                final.bias[index] = _logit((value - lo) / max(hi - lo, 1e-8))

    def forward(self, features, etco2=None, time_grid=None, mask=None, tissue_maps=None,
                bold_psc=None, valid_time_mask=None):
        torch = __import__("torch")
        if bold_psc is None or etco2 is None or time_grid is None:
            raise ValueError("bold_psc, measured etco2 and time_grid are required")
        temporal = self.temporal_encoder(
            bold_psc, etco2, time_grid, valid_time_mask, spatial_mask=mask
        )
        shared = self.spatial_unet(torch.cat([features, temporal], dim=1))
        limits = {"delay": (self.parameter_ranges.delay_min, self.parameter_ranges.delay_max),
                  "T": (self.parameter_ranges.T_min, self.parameter_ranges.T_max)}
        raw, coarse, values = {}, {}, {}
        if self.joint_parameter_head:
            joint = self.joint_head(shared)
            for index, name in enumerate(("delay", "T")):
                raw[name] = joint[:, index]
                lo, hi = limits[name]
                values[name] = lo + (hi - lo) * torch.sigmoid(raw[name])
                coarse[name] = values[name]
            log_vars = {
                name: joint[:, index + 2].clamp(-8, 8)
                for index, name in enumerate(("cvr", "delay", "T"))
            }
            log_scale = joint[:, 5].clamp(-5, 3)
        else:
            for name in ("delay", "T"):
                coarse_raw = self.coarse_heads[name](shared)
                refine = self.refine_heads[name](torch.cat([shared, temporal, coarse_raw[:, None]], 1))
                raw[name] = coarse_raw + 0.75 * torch.tanh(refine)
                lo, hi = limits[name]
                coarse[name] = lo + (hi - lo) * torch.sigmoid(coarse_raw)
                values[name] = lo + (hi - lo) * torch.sigmoid(raw[name])
            log_vars = {
                p: self.uncertainty_heads[p](shared).clamp(-8, 8)
                for p in ("cvr", "delay", "T")
            }
            log_scale = self.observation_uncertainty_head(shared).clamp(-5, 3)
        # The recurrent ODE and least-squares ratios are numerically sensitive
        # in fp16. Keep the expensive learned encoders under AMP, but always run
        # the physical inverse problem in fp32 so GradScaler does not skip steps.
        with torch.autocast(device_type=bold_psc.device.type, enabled=False):
            delay_fp32 = values["delay"].float()
            T_fp32 = values["T"].float()
            bold_fp32 = bold_psc.float()
            etco2_fp32 = etco2.float()
            time_fp32 = time_grid.float()
            mask_fp32 = mask.float() if mask is not None else None
            valid_fp32 = valid_time_mask.float() if valid_time_mask is not None else None
            unit_response = simulate_ode_bold_torch(
                torch.ones_like(delay_fp32), delay_fp32, T_fp32,
                etco2_fp32, time_fp32, mask=mask_fp32, method="exact",
            )
            cvr_profile, amplitude_maps = self._profile_cvr(
                unit_response, bold_fp32, time_fp32, valid_fp32, mask_fp32
            )
        cvr_gate = torch.zeros_like(cvr_profile)
        cvr_factor = torch.ones_like(cvr_profile)
        if self.use_cvr_residual_head:
            correction = self.cvr_residual_head(torch.cat(
                [shared, temporal, amplitude_maps.to(shared.dtype)], dim=1
            ))
            delta_log_cvr = torch.tanh(correction[:, 0])
            cvr_gate = torch.sigmoid(correction[:, 1])
            cvr_factor = torch.exp(
                self.cvr_log_correction_limit * cvr_gate * delta_log_cvr
            )
        values["cvr"] = (cvr_profile * cvr_factor).clamp(
            self.parameter_ranges.cvr_min, self.parameter_ranges.cvr_max
        )
        if mask is not None:
            values["cvr"] = values["cvr"] * mask.to(values["cvr"])
        coarse["cvr"] = values["cvr"]
        with torch.autocast(device_type=bold_psc.device.type, enabled=False):
            physiology = unit_response * values["cvr"].float().unsqueeze(1)
            reconstruction, nuisance = self._profile_nuisance(
                physiology, bold_psc.float(), time_grid.float(), mask_fp32
            )
        parameter_sigma = {p: torch.exp(0.5 * log_vars[p]) for p in values}
        combined_sigma = (
            parameter_sigma["cvr"] / 1.8
            + parameter_sigma["delay"] / 80.0
            + parameter_sigma["T"] / 98.0
        ) / 3.0
        return {
            "shared": shared, "temporal_embedding": temporal, "raw": raw, "coarse": coarse,
            "cvr_profile": cvr_profile, "cvr_correction_factor": cvr_factor,
            "cvr_correction_gate": cvr_gate,
            **values, "bold_psc_physiology": physiology, "bold_psc_hat": reconstruction,
            "nuisance_coefficients": nuisance, "observation_log_scale": log_scale,
            "sigma_y": torch.nn.functional.softplus(log_scale) + 0.03,
            "parameter_log_var": log_vars,
            "sigma": combined_sigma,
            **{f"sigma_{p}": parameter_sigma[p] for p in values},
        }

    def _profile_cvr(self, unit_response, observed, time_grid, valid_time_mask, mask):
        """Estimate CVR amplitude by differentiable least squares in physical units."""
        torch = __import__("torch")
        batch, n_time = observed.shape[:2]
        time = time_grid.to(device=observed.device, dtype=observed.dtype)[:n_time]
        time = time.view(1, n_time).expand(batch, -1)
        valid = torch.ones_like(time)
        if valid_time_mask is not None:
            valid = valid_time_mask.to(device=observed.device, dtype=observed.dtype)[:, :n_time]
        weight_shape = (batch, n_time) + (1,) * (observed.ndim - 2)
        weight = valid.view(weight_shape)
        denominator = weight.sum(dim=1).clamp_min(1.0)
        time_mean = (valid * time).sum(1) / valid.sum(1).clamp_min(1.0)
        centered_time = time - time_mean[:, None]
        time_view = centered_time.view(weight_shape)
        time_energy = (weight * time_view.square()).sum(1).clamp_min(1e-6)

        def remove_intercept_and_drift(values):
            mean = (weight * values).sum(1) / denominator
            centered = values - mean.unsqueeze(1)
            slope = (weight * centered * time_view).sum(1) / time_energy
            return centered - slope.unsqueeze(1) * time_view

        x = remove_intercept_and_drift(unit_response)
        y = remove_intercept_and_drift(observed)
        covariance = (weight * x * y).sum(1) / denominator
        response_energy = (weight * x.square()).sum(1) / denominator
        observed_energy = (weight * y.square()).sum(1) / denominator
        cvr = covariance / response_energy.clamp_min(1e-6)
        cvr = cvr.clamp(self.parameter_ranges.cvr_min, self.parameter_ranges.cvr_max)
        residual = y - cvr.unsqueeze(1) * x
        residual_energy = (weight * residual.square()).sum(1) / denominator
        amplitude_maps = torch.stack(
            [
                cvr,
                covariance,
                (response_energy.clamp_min(0.0) + 1e-6).sqrt(),
                (observed_energy.clamp_min(0.0) + 1e-6).sqrt(),
                (residual_energy.clamp_min(0.0) + 1e-6).sqrt(),
            ],
            dim=1,
        )
        if mask is not None:
            spatial_mask = mask.to(device=cvr.device, dtype=cvr.dtype)
            cvr = cvr * spatial_mask
            amplitude_maps = amplitude_maps * spatial_mask.unsqueeze(1)
        return cvr, amplitude_maps

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
