from __future__ import annotations

from hybrid_cvr.models.constraints import ParameterRanges
from hybrid_cvr.models.unet2d import UNet2D
from hybrid_cvr.models.unet3d import UNet3D
from hybrid_cvr.pinn.torch_ode import simulate_ode_bold_torch


class ParameterHead(__import__("torch").nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, spatial_dims: int = 2):
        super().__init__()
        torch = __import__("torch")
        nn = torch.nn
        conv = nn.Conv3d if int(spatial_dims) == 3 else nn.Conv2d
        self.net = nn.Sequential(
            conv(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            conv(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class HybridUNetPINN(__import__("torch").nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 3,
        norm: str = "instance",
        dropout: float = 0.0,
        parameter_ranges: ParameterRanges | None = None,
        T_mode: str = "tissuewise",
        spatial_dims: int = 2,
        T_residual_scale: float = 25.0,
    ):
        super().__init__()
        torch = __import__("torch")
        nn = torch.nn
        if T_mode not in {"global", "tissuewise", "voxelwise"}:
            raise ValueError("T_mode must be one of: global, tissuewise, voxelwise")
        if int(spatial_dims) not in {2, 3}:
            raise ValueError("spatial_dims must be 2 or 3")
        self.T_mode = T_mode
        self.spatial_dims = int(spatial_dims)
        self.T_residual_scale = float(T_residual_scale)
        self.parameter_ranges = parameter_ranges or ParameterRanges()
        self.shared_channels = int(base_channels)
        unet_cls = UNet3D if self.spatial_dims == 3 else UNet2D
        self.unet = unet_cls(
            in_channels=in_channels,
            out_channels=self.shared_channels,
            base_channels=base_channels,
            depth=depth,
            norm=norm,
            dropout=dropout,
        )
        self.cvr_head = ParameterHead(self.shared_channels, 1, spatial_dims=self.spatial_dims)
        self.delay_head = ParameterHead(self.shared_channels, 1, spatial_dims=self.spatial_dims)
        self.log_var_head = ParameterHead(self.shared_channels, 1, spatial_dims=self.spatial_dims)
        self.T_voxel_head = ParameterHead(self.shared_channels, 1, spatial_dims=self.spatial_dims)
        self.T_pool = nn.AdaptiveAvgPool3d(1) if self.spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
        self.T_tissue_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.shared_channels, self.shared_channels),
            nn.SiLU(inplace=True),
            nn.Linear(self.shared_channels, 5),
        )
        self.register_buffer(
            "tissue_T_medians",
            torch.tensor([30.1, 21.75, 45.27, 50.08, 17.41], dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        features,
        etco2=None,
        time_grid=None,
        mask=None,
        tissue_maps=None,
        bold_psc=None,
    ):
        torch = __import__("torch")
        shared = self.unet(features)
        raw_cvr = self.cvr_head(shared).squeeze(1)
        raw_delay = self.delay_head(shared).squeeze(1)
        raw_log_var = self.log_var_head(shared).squeeze(1)
        cvr = self._scale_sigmoid(raw_cvr, self.parameter_ranges.cvr_min, self.parameter_ranges.cvr_max)
        delay = self._scale_sigmoid(
            raw_delay, self.parameter_ranges.delay_min, self.parameter_ranges.delay_max
        )
        T = self._predict_T(shared, features, tissue_maps)
        log_var = raw_log_var.clamp(-8.0, 6.0)
        sigma = torch.exp(0.5 * log_var).clamp_min(self.parameter_ranges.sigma_eps)
        out = {
            "shared": shared,
            "raw": {"cvr": raw_cvr, "delay": raw_delay, "T": T, "log_var": raw_log_var},
            "cvr": cvr,
            "delay": delay,
            "T": T,
            "log_var": log_var,
            "sigma": sigma,
        }
        if etco2 is not None and time_grid is not None:
            out["bold_psc_hat"] = simulate_ode_bold_torch(
                cvr, delay, T, etco2, time_grid, mask=mask, method="exact"
            )
        return out

    def _predict_T(self, shared, features, tissue_maps=None):
        torch = __import__("torch")
        if self.T_mode == "voxelwise":
            raw_T = self.T_voxel_head(shared).squeeze(1)
            base_T = self._tissue_T_baseline(raw_T, tissue_maps)
            if base_T is not None:
                T = base_T + self.T_residual_scale * torch.tanh(raw_T)
                return T.clamp(float(self.parameter_ranges.T_min), float(self.parameter_ranges.T_max))
            return self._scale_sigmoid(raw_T, self.parameter_ranges.T_min, self.parameter_ranges.T_max)

        raw = self.T_tissue_head(self.T_pool(shared))
        T_values = self._scale_sigmoid(raw, self.parameter_ranges.T_min, self.parameter_ranges.T_max)
        if self.T_mode == "global":
            return self._expand_scalar_map(T_values.mean(dim=1), features)

        if tissue_maps is None:
            return self._expand_scalar_map(T_values.mean(dim=1), features)
        tissue = tissue_maps.to(device=shared.device, dtype=shared.dtype)
        expected_ndim = 2 + self.spatial_dims
        if tissue.ndim != expected_ndim or tissue.shape[1] < 5:
            raise ValueError(f"tissue_maps must have shape B,5,{','.join(['*'] * self.spatial_dims)}")
        weights = tissue[:, :5]
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        view_shape = (T_values.shape[0], 5) + (1,) * self.spatial_dims
        T_map = (weights * T_values.view(view_shape)).sum(dim=1)
        fallback = self._expand_scalar_map(T_values.mean(dim=1), features)
        return torch.where(denom.squeeze(1) > 0, T_map, fallback)

    def _tissue_T_baseline(self, raw_T, tissue_maps=None):
        torch = __import__("torch")
        if tissue_maps is None:
            return None
        tissue = tissue_maps.to(device=raw_T.device, dtype=raw_T.dtype)
        expected_ndim = 2 + self.spatial_dims
        if tissue.ndim != expected_ndim or tissue.shape[1] < 5:
            return None
        weights = tissue[:, :5]
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        med = self.tissue_T_medians.to(device=raw_T.device, dtype=raw_T.dtype)
        view_shape = (1, 5) + (1,) * self.spatial_dims
        baseline = ((weights / denom) * med.view(view_shape)).sum(dim=1)
        fallback = torch.full_like(raw_T, float(med.mean()))
        return torch.where(denom.squeeze(1) > 0, baseline, fallback)

    def _expand_scalar_map(self, values, features):
        view_shape = (values.shape[0],) + (1,) * self.spatial_dims
        return values.view(view_shape).expand((values.shape[0],) + tuple(features.shape[-self.spatial_dims :]))

    @staticmethod
    def _scale_sigmoid(raw, lower: float, upper: float):
        torch = __import__("torch")
        return float(lower) + (float(upper) - float(lower)) * torch.sigmoid(raw)
