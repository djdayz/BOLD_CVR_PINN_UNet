from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    ec: str = "#2f2f2f",
    fs: float = 9.0,
    weight: str = "normal",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight)
    return patch


def arrow(ax, start, end, *, color="#2f2f2f", lw=1.4, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(patch)


def main() -> None:
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis("off")

    c_input = "#e8f1fb"
    c_unet = "#dff0d8"
    c_bottleneck = "#fce5cd"
    c_head = "#eadcf8"
    c_pinn = "#fff2cc"
    c_loss = "#f4cccc"
    c_meta = "#eeeeee"

    ax.text(
        11,
        13.45,
        "Hybrid U-Net/PINN Architecture Used For BOLD-MRI CVR Mapping",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        11,
        13.05,
        "Current full-brain run: 2D axial slices, 15 observable input channels, 94 x 50 slice grid, 360 epochs; IN = InstanceNorm",
        ha="center",
        va="center",
        fontsize=11,
    )

    y_main = 10.15
    h_main = 1.25
    main_boxes = [
        (
            0.55,
            y_main,
            2.45,
            h_main,
            "Input layer\nB x 15 x 94 x 50\n15 observed feature maps",
            c_input,
        ),
        (
            3.45,
            y_main,
            2.25,
            h_main,
            "Encoder L1\n32 x 94 x 50\n150,400 activations\n2 x Conv3x3 + IN + SiLU",
            c_unet,
        ),
        (
            6.2,
            y_main,
            2.25,
            h_main,
            "Encoder L2\n64 x 47 x 25\n75,200 activations\n2 x Conv3x3 + IN + SiLU",
            c_unet,
        ),
        (
            8.95,
            y_main,
            2.25,
            h_main,
            "Encoder L3\n128 x 23 x 12\n35,328 activations\n2 x Conv3x3 + IN + SiLU",
            c_unet,
        ),
        (
            11.75,
            y_main,
            2.25,
            h_main,
            "Bottleneck\n256 x 11 x 6\n16,896 activations\n2 x Conv3x3 + IN + SiLU",
            c_bottleneck,
        ),
        (
            14.55,
            y_main,
            2.2,
            h_main,
            "Decoder L3\n128 x 23 x 12\nUpConv2x2 + skip\n2 x Conv3x3 + IN + SiLU",
            c_unet,
        ),
        (
            17.25,
            y_main,
            2.2,
            h_main,
            "Decoder L2\n64 x 47 x 25\nUpConv2x2 + skip\n2 x Conv3x3 + IN + SiLU",
            c_unet,
        ),
        (
            19.8,
            y_main,
            1.85,
            h_main,
            "Decoder L1\n32 x 94 x 50\nUpConv2x2 + skip\n2 x Conv3x3",
            c_unet,
        ),
    ]
    for x, y, w, h, text, fc in main_boxes:
        box(ax, x, y, w, h, text, fc=fc, fs=7.8)
    for left, right in zip(main_boxes[:-1], main_boxes[1:], strict=True):
        x1, y1, w1, h1 = left[:4]
        x2, y2, _, h2 = right[:4]
        arrow(ax, (x1 + w1, y1 + h1 / 2), (x2, y2 + h2 / 2))

    for x in (5.85, 8.6, 11.4):
        ax.text(x, 9.82, "MaxPool2d(2)\nspatial size / 2", ha="center", va="top", fontsize=7.4)
    for x in (14.2, 16.95, 19.6):
        ax.text(
            x,
            9.82,
            "ConvTranspose2d\nspatial size x 2",
            ha="center",
            va="top",
            fontsize=7.4,
        )

    skip_color = "#5b8c5a"
    arrow(ax, (4.55, 11.42), (20.7, 11.42), color=skip_color, lw=1.3, rad=-0.12)
    arrow(ax, (7.3, 11.42), (18.35, 11.42), color=skip_color, lw=1.3, rad=-0.1)
    arrow(ax, (10.05, 11.42), (15.65, 11.42), color=skip_color, lw=1.3, rad=-0.08)
    ax.text(
        15.1,
        12.05,
        "Skip connections: encoder detail is copied into decoder to preserve tissue boundaries",
        fontsize=9,
    )

    box(
        ax,
        0.55,
        7.0,
        4.15,
        2.25,
        "Input feature channels\n\nbaseline BOLD\nPSC baseline / hypercapnia / mean / std / delta\ntCNR and brain mask\nvessel likelihood and boundary uncertainty\nCGM, SGM, WM, VCSF, vessel-like fractions\n\nGT CVR/delay/T are NOT input channels",
        fc=c_input,
        fs=8.0,
    )
    arrow(ax, (2.6, 9.25), (1.8, 10.15), color="#4169aa")

    box(
        ax,
        17.35,
        7.95,
        3.2,
        0.85,
        "Shared decoder feature map\nB x 32 x 94 x 50",
        fc="#edf7ed",
        fs=8.5,
        weight="bold",
    )
    arrow(ax, (20.72, 10.15), (19.0, 8.8), rad=0.05)

    head_specs = [
        ("CVR head\nConv3x3 -> SiLU -> Conv1x1\nsigmoid scale: -0.4 to 1.8", 6.65),
        ("Delay head\nConv3x3 -> SiLU -> Conv1x1\nsigmoid scale: 0 to 80 s", 5.55),
        (
            "T head\nConv3x3 -> SiLU -> Conv1x1\nstage 1 global, stage 2 tissuewise,\nstage 3-4 voxelwise; 2 to 100 s",
            4.35,
        ),
        ("Uncertainty head\nConv3x3 -> SiLU -> Conv1x1\nlog variance -> sigma", 3.15),
    ]
    for text, y in head_specs:
        box(ax, 17.35, y, 3.2, 0.85, text, fc=c_head, fs=7.7)
        arrow(ax, (19.0, 7.95), (19.0, y + 0.85), rad=0.02)

    box(
        ax,
        13.2,
        4.95,
        3.25,
        1.8,
        "Network outputs\n\nCVR map: B x 94 x 50\nDelay map: B x 94 x 50\nT map: B x 94 x 50\nSigma map: B x 94 x 50",
        fc=c_head,
        fs=8.4,
    )
    for y in (7.08, 5.98, 4.78, 3.58):
        arrow(ax, (17.35, y), (16.45, 5.82), rad=0.03)

    box(
        ax,
        8.65,
        5.15,
        3.25,
        1.35,
        "ETCO2 time course\nu(t)\n480 frames, TR = 1.55 s\ntraining windows: 240 or 320 frames",
        fc=c_input,
        fs=8.2,
    )
    box(
        ax,
        8.4,
        2.95,
        4.0,
        1.65,
        "PINN / differentiable ODE forward model\n\ndy/dt = (CVR * u(t - delay) - y) / T\n\noutputs reconstructed BOLD PSC",
        fc=c_pinn,
        fs=8.5,
    )
    arrow(ax, (10.25, 5.15), (10.25, 4.6))
    arrow(ax, (13.2, 5.2), (12.4, 3.8))

    box(
        ax,
        0.55,
        3.0,
        3.75,
        1.55,
        "Observed BOLD PSC\non-the-fly synthetic data\n\nmixed tCNR, paradigms, ETCO2 noise,\ndrift, AR(1) noise, motion spikes",
        fc=c_input,
        fs=8.0,
    )
    box(
        ax,
        5.0,
        2.75,
        2.6,
        1.95,
        "Self-supervised losses\n\nL_recon: BOLD fit\nL_residual: ODE dynamics\nL_uncertainty\nL_smooth\nL_prior\nL_consistency",
        fc=c_loss,
        fs=8.2,
    )
    arrow(ax, (4.3, 3.78), (5.0, 3.78))
    arrow(ax, (8.4, 3.78), (7.6, 3.78))

    box(
        ax,
        8.75,
        0.75,
        4.0,
        1.45,
        "Automatic differentiation / backpropagation\n\nLoss gradients flow through the ODE solver,\nparameter heads, decoder, bottleneck, and encoder.\nOptimizer: AdamW, learning rate = 1e-4",
        fc=c_meta,
        fs=8.4,
    )
    arrow(ax, (6.3, 2.75), (8.75, 1.5), rad=-0.1)
    arrow(ax, (12.75, 1.5), (18.8, 7.95), color="#8a6d00", lw=1.5, rad=-0.15)

    box(
        ax,
        13.3,
        0.75,
        3.15,
        1.45,
        "Layer count summary\n\nU-Net: 15 Conv2d + 3 UpConv layers\nHeads: 8 Conv2d layers\nStage-2 tissuewise T also uses a 2-layer MLP",
        fc=c_meta,
        fs=8.2,
    )
    box(
        ax,
        0.55,
        0.75,
        7.25,
        1.65,
        "Training stages and updates\n\nStage 1: 40 epochs, clean ETCO2, high tCNR, global T\nStage 2: 80 epochs, measured ETCO2, physics residual, tissuewise T\nStage 3: 180 epochs, mixed paradigms/tCNR, voxelwise T, consistency\nStage 4: 60 epochs, low-tCNR stress, voxelwise T\nUpdates: full axial range [0,93], 100 GT cases split 60/20/20,\nfixed LR scheduler disabled, stage-best checkpoints, readable epoch logs",
        fc=c_meta,
        fs=7.45,
    )

    fig.savefig(out_dir / "hybrid_unet_pinn_architecture.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "hybrid_unet_pinn_architecture.pdf", bbox_inches="tight")
    print(out_dir / "hybrid_unet_pinn_architecture.png")
    print(out_dir / "hybrid_unet_pinn_architecture.pdf")


if __name__ == "__main__":
    main()
