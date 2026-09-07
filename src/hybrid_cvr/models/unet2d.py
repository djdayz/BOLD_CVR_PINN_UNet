from __future__ import annotations


def _norm_layer(norm: str, channels: int):
    torch = __import__("torch")
    nn = torch.nn
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    return nn.Identity()


class ConvBlock(__import__("torch").nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance", dropout: float = 0.0):
        super().__init__()
        torch = __import__("torch")
        nn = torch.nn
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _norm_layer(norm, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _norm_layer(norm, out_channels),
            nn.SiLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet2D(__import__("torch").nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 4,
        base_channels: int = 32,
        depth: int = 4,
        norm: str = "instance",
        dropout: float = 0.0,
    ):
        super().__init__()
        torch = __import__("torch")
        nn = torch.nn
        self.pool = nn.MaxPool2d(2)
        self.down = nn.ModuleList()
        channels = []
        prev = in_channels
        for i in range(depth):
            ch = base_channels * 2**i
            self.down.append(ConvBlock(prev, ch, norm=norm, dropout=dropout))
            channels.append(ch)
            prev = ch
        self.bottleneck = ConvBlock(prev, prev * 2, norm=norm, dropout=dropout)
        self.up_transpose = nn.ModuleList()
        self.up = nn.ModuleList()
        current = prev * 2
        for ch in reversed(channels):
            self.up_transpose.append(nn.ConvTranspose2d(current, ch, kernel_size=2, stride=2))
            self.up.append(ConvBlock(ch * 2, ch, norm=norm, dropout=dropout))
            current = ch
        self.out = nn.Conv2d(current, out_channels, kernel_size=1)

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
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return self.out(x)
