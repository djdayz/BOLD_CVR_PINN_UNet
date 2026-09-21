from __future__ import annotations


def _norm_layer(norm: str, channels: int):
    torch = __import__("torch")
    nn = torch.nn
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    return nn.Identity()


class ConvBlock3D(__import__("torch").nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        torch = __import__("torch")
        nn = torch.nn
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            _norm_layer(norm, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            _norm_layer(norm, out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout3d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet3D(__import__("torch").nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 4,
        base_channels: int = 16,
        depth: int = 3,
        norm: str = "instance",
        dropout: float = 0.0,
    ):
        super().__init__()
        torch = __import__("torch")
        nn = torch.nn
        self.pool = nn.MaxPool3d(2)
        self.down = nn.ModuleList()
        channels = []
        prev = in_channels
        for i in range(depth):
            ch = base_channels * 2**i
            self.down.append(ConvBlock3D(prev, ch, norm=norm, dropout=dropout))
            channels.append(ch)
            prev = ch
        self.bottleneck = ConvBlock3D(prev, prev * 2, norm=norm, dropout=dropout)
        self.up_transpose = nn.ModuleList()
        self.up = nn.ModuleList()
        current = prev * 2
        for ch in reversed(channels):
            self.up_transpose.append(nn.ConvTranspose3d(current, ch, kernel_size=2, stride=2))
            self.up.append(ConvBlock3D(ch * 2, ch, norm=norm, dropout=dropout))
            current = ch
        self.out = nn.Conv3d(current, out_channels, kernel_size=1)

    def forward(self, x):
        torch = __import__("torch")
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up_t, block, skip in zip(self.up_transpose, self.up, reversed(skips), strict=True):
            x = up_t(x)
            if x.shape[-3:] != skip.shape[-3:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return self.out(x)
