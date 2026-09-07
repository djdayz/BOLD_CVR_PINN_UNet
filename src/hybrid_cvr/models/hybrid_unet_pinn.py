from __future__ import annotations

from hybrid_cvr.models.constraints import ParameterRanges
from hybrid_cvr.models.unet2d import UNet2D
from hybrid_cvr.pinn.torch_ode import simulate_ode_bold_torch


class ParameterHead(__import__("torch").nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        torch = __import__("torch")
        nn = torch.nn
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
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
    ):
        super().__init__()
        torch = __import__("torch")
        nn = torch.nn
        if T_mode not in {"global", "tissuewise", "voxelwise"}:
            raise ValueError("T_mode must be one of: global, tissuewise, voxelwise")
        self.T_mode = T_mode
        self.parameter_ranges = parameter_ranges or ParameterRanges()
        self.shared_channels = int(base_channels)
        self.unet = UNet2D(
            in_channels=in_channels,
            out_channels=self.shared_channels,
            base_channels=base_channels,
            depth=depth,
            norm=norm,
            dropout=dropout,
        )
        self.cvr_head = ParameterHead(self.shared_channels, 1)
        self.delay_head = ParameterHead(self.shared_channels, 1)
        self.log_var_head = ParameterHead(self.shared_channels, 1)
        self.T_voxel_head = ParameterHead(self.shared_channels, 1)
        self.T_pool = nn.AdaptiveAvgPool2d(1)
        self.T_tissue_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.shared_channels, self.shared_channels),
            nn.SiLU(inplace=True),
            nn.Linear(self.shared_channels, 5),
        )

    def forward(self, features, etco2=None, time_grid=None, mask=None, tissue_maps=None):
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
            return self._scale_sigmoid(raw_T, self.parameter_ranges.T_min, self.parameter_ranges.T_max)

        raw = self.T_tissue_head(self.T_pool(shared))
        T_values = self._scale_sigmoid(raw, self.parameter_ranges.T_min, self.parameter_ranges.T_max)
        if self.T_mode == "global":
            return T_values.mean(dim=1).view(-1, 1, 1).expand(
                -1, features.shape[-2], features.shape[-1]
            )

        if tissue_maps is None:
            return T_values.mean(dim=1).view(-1, 1, 1).expand(
                -1, features.shape[-2], features.shape[-1]
            )
        tissue = tissue_maps.to(device=shared.device, dtype=shared.dtype)
        if tissue.ndim != 4 or tissue.shape[1] < 5:
            raise ValueError("tissue_maps must have shape B,5,H,W for tissuewise T")
        weights = tissue[:, :5]
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        T_map = (weights * T_values.view(T_values.shape[0], 5, 1, 1)).sum(dim=1)
        fallback = T_values.mean(dim=1).view(-1, 1, 1)
        return torch.where(denom.squeeze(1) > 0, T_map, fallback)

    @staticmethod
    def _scale_sigmoid(raw, lower: float, upper: float):
        torch = __import__("torch")
        return float(lower) + (float(upper) - float(lower)) * torch.sigmoid(raw)
