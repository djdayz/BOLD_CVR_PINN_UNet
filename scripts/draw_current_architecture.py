#!/usr/bin/env python3
"""Draw the current self-supervised 1D-CNN/3D-U-Net CVR architecture."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "figures" / "presentation"

COLORS = {
    "ink": "#17212B",
    "muted": "#52616B",
    "input": "#EAF2F8",
    "input_edge": "#4F7D95",
    "train": "#DDF3EE",
    "train_edge": "#16806A",
    "physics": "#FFF1D6",
    "physics_edge": "#C27716",
    "output": "#FCE5E5",
    "output_edge": "#B74B4B",
    "loss": "#F1EAF7",
    "loss_edge": "#76528D",
    "white": "#FFFFFF",
}


def box(ax, x, y, w, h, face, edge, title, lines=(), title_size=13, text_size=10.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.8, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=COLORS["ink"])
    if lines:
        ax.text(x + w / 2, y + h * 0.37, "\n".join(lines), ha="center", va="center",
                fontsize=text_size, color=COLORS["ink"], linespacing=1.38)
    return patch


def arrow(ax, start, end, color=None, width=2.5, head=18, zorder=6):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=head,
        linewidth=width, color=color or COLORS["ink"],
        connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=0, zorder=zorder,
    ))


def line(ax, start, end, color=None, width=1.7, style="-"):
    ax.plot([start[0], end[0]], [start[1], end[1]], style,
            color=color or COLORS["ink"], linewidth=width, solid_capstyle="round")


def draw_unet(ax, x, y, w, h):
    box(ax, x, y, w, h, COLORS["train"], COLORS["train_edge"], "Full-volume 3D U-Net", ())
    ax.text(x + w / 2, y + h - 0.095, "joint spatial representation", ha="center",
            va="top", fontsize=10.5, color=COLORS["muted"])

    centers = [
        (x + 0.12 * w, y + 0.58 * h),
        (x + 0.31 * w, y + 0.40 * h),
        (x + 0.50 * w, y + 0.24 * h),
        (x + 0.69 * w, y + 0.40 * h),
        (x + 0.88 * w, y + 0.58 * h),
    ]
    labels = ["24", "48", "96", "48", "24"]
    sizes = [(0.11, 0.16), (0.10, 0.14), (0.10, 0.13), (0.10, 0.14), (0.11, 0.16)]
    patches = []
    for (cx, cy), label, (rw, rh) in zip(centers, labels, sizes, strict=True):
        p = FancyBboxPatch(
            (cx - rw * w / 2, cy - rh * h / 2), rw * w, rh * h,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            facecolor=COLORS["white"], edgecolor=COLORS["train_edge"], linewidth=1.35,
        )
        ax.add_patch(p)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=COLORS["ink"])
        patches.append((cx, cy, rw * w, rh * h))
    for left, right in zip(patches[:-1], patches[1:], strict=True):
        arrow(ax, (left[0] + left[2] / 2, left[1]), (right[0] - right[2] / 2, right[1]),
              color=COLORS["train_edge"], width=1.35, head=10)
    # Straight skip connections.
    y_skip_1 = y + 0.72 * h
    line(ax, (centers[0][0], centers[0][1] + 0.08 * h), (centers[0][0], y_skip_1), COLORS["train_edge"], 1.2)
    line(ax, (centers[0][0], y_skip_1), (centers[4][0], y_skip_1), COLORS["train_edge"], 1.2)
    arrow(ax, (centers[4][0], y_skip_1), (centers[4][0], centers[4][1] + 0.08 * h), COLORS["train_edge"], 1.2, 9)
    y_skip_2 = y + 0.59 * h
    line(ax, (centers[1][0], centers[1][1] + 0.07 * h), (centers[1][0], y_skip_2), COLORS["train_edge"], 1.2)
    line(ax, (centers[1][0], y_skip_2), (centers[3][0], y_skip_2), COLORS["train_edge"], 1.2)
    arrow(ax, (centers[3][0], y_skip_2), (centers[3][0], centers[3][1] + 0.07 * h), COLORS["train_edge"], 1.2, 9)
    ax.text(x + w / 2, y + 0.025, "Conv3D + IN3D + SiLU   |   MaxPool / transposed Conv",
            ha="center", va="bottom", fontsize=7.6, color=COLORS["muted"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Input branch.
    box(ax, 0.025, 0.665, 0.155, 0.19, COLORS["input"], COLORS["input_edge"], "", ())
    ax.text(
        0.1025, 0.76,
        "4D BOLD PSC\n$\\mathbf{[B,480,94,94,50]}$\nMeasured $\\mathbf{\\Delta ETCO_2}$\ntime + valid mask",
        ha="center", va="center", fontsize=10.7, fontweight="bold",
        color=COLORS["ink"], linespacing=1.42,
    )
    box(ax, 0.025, 0.43, 0.155, 0.17, COLORS["input"], COLORS["input_edge"],
        "Observable maps",
        ("10 spatial channels", "BOLD summaries, tCNR,", "mask, CO2 response features"), text_size=10.0)

    box(ax, 0.215, 0.645, 0.205, 0.23, COLORS["train"], COLORS["train_edge"],
        "Voxelwise 1D CNN",
        ("5 time-series inputs per voxel", "Conv1D: 5 -> 48, kernel 5", "6 residual dilated blocks", "d = 1, 2, 4, 8, 16, 32", "global average pooling"),
        text_size=9.7)
    ax.text(0.3175, 0.63, "shared, non-causal encoder -> 48 features / voxel",
            ha="center", va="top", fontsize=9.2, color=COLORS["muted"])
    arrow(ax, (0.18, 0.76), (0.215, 0.76))

    # Concatenation node.
    diamond = Polygon([(0.445, 0.62), (0.468, 0.66), (0.491, 0.62), (0.468, 0.58)],
                      closed=True, facecolor=COLORS["white"], edgecolor=COLORS["train_edge"], linewidth=1.8)
    ax.add_patch(diamond)
    ax.text(0.468, 0.62, "+", ha="center", va="center", fontsize=18,
            fontweight="bold", color=COLORS["train_edge"])
    ax.text(0.468, 0.555, "concatenate\n58 channels", ha="center", va="top",
            fontsize=9.5, color=COLORS["muted"])
    arrow(ax, (0.42, 0.76), (0.468, 0.66))
    line(ax, (0.18, 0.515), (0.468, 0.515), COLORS["ink"], 1.8)
    arrow(ax, (0.468, 0.515), (0.468, 0.58))

    draw_unet(ax, 0.515, 0.57, 0.285, 0.31)
    arrow(ax, (0.491, 0.62), (0.515, 0.62))

    box(ax, 0.835, 0.62, 0.14, 0.235, COLORS["train"], COLORS["train_edge"],
        "Joint 3D head",
        ("3x3x3 Conv", "InstanceNorm3D + SiLU", "1x1x1 Conv -> 6", "", r"$\hat{\tau}: 0-80\,s$", r"$\hat{T}: 2-100\,s$", "log-variances + noise scale"),
        text_size=9.1)
    arrow(ax, (0.80, 0.72), (0.835, 0.72))

    # Differentiable physiology and amplitude-preserving CVR branch.
    box(ax, 0.205, 0.245, 0.205, 0.19, COLORS["physics"], COLORS["physics_edge"],
        "Differentiable first-order physiology",
        (), title_size=9.8)
    ax.text(0.3075, 0.335, r"$\frac{dx(t)}{dt}=\frac{u(t-\hat{\tau})-x(t)}{\hat{T}}$",
            ha="center", va="center", fontsize=17.0, color=COLORS["ink"])
    ax.text(0.3075, 0.278, "unit-CVR response x(t)",
            ha="center", va="center", fontsize=11.5, color=COLORS["ink"])
    arrow(ax, (0.905, 0.62), (0.905, 0.485), COLORS["physics_edge"])
    line(ax, (0.905, 0.485), (0.3075, 0.485), COLORS["physics_edge"], 2.2)
    arrow(ax, (0.3075, 0.485), (0.3075, 0.435), COLORS["physics_edge"])
    line(ax, (0.1025, 0.665), (0.1025, 0.34), COLORS["input_edge"], 1.5)
    arrow(ax, (0.1025, 0.34), (0.205, 0.34), COLORS["input_edge"], 2.2)
    ax.text(0.155, 0.352, r"raw PSC + $\Delta$ETCO$_2$", ha="center", va="bottom",
            fontsize=8.8, color=COLORS["input_edge"])

    box(ax, 0.435, 0.245, 0.17, 0.19, COLORS["physics"], COLORS["physics_edge"],
        "Physical CVR profile",
        ("raw physical amplitudes", r"$\mathrm{CVR}_{p}=\frac{\langle x',y'\rangle}{\langle x',x'\rangle}$",
         "CVR, covariance and RMS maps"),
        title_size=10.3, text_size=11.8)
    arrow(ax, (0.410, 0.34), (0.435, 0.34), COLORS["physics_edge"])

    box(ax, 0.63, 0.245, 0.17, 0.19, COLORS["train"], COLORS["train_edge"],
        "Gated residual CVR head",
        ("1x1x1 + 3x3x3 + 1x1x1 Conv", "no pooling or normalization",
         r"$\widehat{\mathrm{CVR}}=\mathrm{CVR}_{p}$",
         r"$\times\exp[\ln(2)\,g\tanh(\Delta)]$"),
        title_size=9.7, text_size=9.3)
    arrow(ax, (0.605, 0.34), (0.63, 0.34), COLORS["train_edge"])
    # Shared spatial and temporal features also enter the residual CVR head.
    line(ax, (0.70, 0.57), (0.70, 0.455), COLORS["train_edge"], 2.0)
    arrow(ax, (0.70, 0.455), (0.70, 0.435), COLORS["train_edge"])
    ax.text(0.707, 0.46, "shared 3D + temporal features", ha="left", va="bottom",
            fontsize=8.3, color=COLORS["train_edge"])

    box(ax, 0.825, 0.245, 0.15, 0.19, COLORS["output"], COLORS["output_edge"],
        "Outputs and reconstruction",
        (r"$\widehat{\mathrm{CVR}},\ \hat{\tau},\ \hat{T}$", "parameter uncertainty maps", r"$\hat{y}(t)=\widehat{\mathrm{CVR}}\,x(t)+a+bt$", "reconstructed BOLD PSC"),
        title_size=9.5, text_size=8.9)
    arrow(ax, (0.80, 0.34), (0.825, 0.34), COLORS["output_edge"])

    # Loss and optimization.
    box(ax, 0.15, 0.035, 0.70, 0.145, COLORS["loss"], COLORS["loss_edge"],
        "", ())
    ax.text(0.50, 0.151, "Loss function", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.50, 0.112,
        r"$\mathcal{L}=\mathcal{L}_{\mathrm{Student\text{-}t\ reconstruction}}"
        r"+\lambda\,\mathcal{L}_{\mathrm{cross\text{-}paradigm\ consistency}}$",
        ha="center", va="center", fontsize=15.0, color=COLORS["ink"],
    )
    ax.text(
        0.50, 0.068,
        "Stage 1: clean identifiable signals   |   Stage 2: mixed tCNR, paradigms and artefacts",
        ha="center", va="center", fontsize=11.5, color=COLORS["ink"],
    )
    line(ax, (0.90, 0.245), (0.90, 0.112), COLORS["loss_edge"], 2.5)
    arrow(ax, (0.90, 0.112), (0.85, 0.112), COLORS["loss_edge"])
    ax.text(0.885, 0.122, r"compare $\hat{y}(t)$ with observed $y(t)$",
            ha="right", va="bottom", fontsize=9.2, color=COLORS["loss_edge"])
    # Back-propagation path to trainable blocks.
    line(ax, (0.15, 0.112), (0.115, 0.112), COLORS["loss_edge"], 1.6)
    line(ax, (0.115, 0.112), (0.115, 0.205), COLORS["loss_edge"], 1.6)
    line(ax, (0.115, 0.205), (0.615, 0.205), COLORS["loss_edge"], 1.6, "--")
    arrow(ax, (0.615, 0.205), (0.615, 0.57), COLORS["loss_edge"], 1.6, 11)
    ax.text(0.13, 0.215, "back-propagation updates the 1D CNN, 3D U-Net, parameter heads and CVR residual head",
            ha="left", va="bottom", fontsize=9.5, color=COLORS["loss_edge"])

    ax.text(0.5, 0.006,
            "Ground-truth CVR, delay and T maps are hidden during training and used only for final validation/test evaluation.",
            ha="center", va="bottom", fontsize=10.2, fontweight="bold", color=COLORS["ink"])

    png = OUT_DIR / "current_self_supervised_cvr_architecture.png"
    pdf = OUT_DIR / "current_self_supervised_cvr_architecture.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
