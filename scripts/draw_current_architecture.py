#!/usr/bin/env python3
"""Draw a concise presentation diagram of the current CVR architecture."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "figures" / "presentation"

COLORS = {
    "ink": "#17212B",
    "input": "#EAF2F8",
    "input_edge": "#4F7D95",
    "model": "#DDF3EE",
    "model_edge": "#16806A",
    "output": "#FCE5E5",
    "output_edge": "#B74B4B",
    "physics": "#FFF1D6",
    "physics_edge": "#C27716",
    "loss": "#F1EAF7",
    "loss_edge": "#76528D",
}


def box(ax, x, y, w, h, face, edge, title, body, *, title_size=14, body_size=11.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=2.0, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h - 0.045, title,
        ha="center", va="top", fontsize=title_size,
        fontweight="bold", color=COLORS["ink"], zorder=3,
    )
    ax.text(
        x + w / 2, y + h * 0.40, body,
        ha="center", va="center", fontsize=body_size,
        color=COLORS["ink"], linespacing=1.45, zorder=3,
    )


def arrow(ax, start, end, *, color=None, width=2.8, head=18):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=head,
        linewidth=width, color=color or COLORS["ink"],
        connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=0, zorder=5,
    ))


def line(ax, start, end, *, color=None, width=2.0):
    ax.plot(
        [start[0], end[0]], [start[1], end[1]],
        color=color or COLORS["ink"], linewidth=width,
        solid_capstyle="round", zorder=4,
    )


def draw_mini_unet(ax, x, y, w, h):
    centers = [
        (x + 0.13 * w, y + 0.48 * h),
        (x + 0.31 * w, y + 0.32 * h),
        (x + 0.50 * w, y + 0.18 * h),
        (x + 0.69 * w, y + 0.32 * h),
        (x + 0.87 * w, y + 0.48 * h),
    ]
    labels = ("24", "48", "96", "48", "24")
    node_w, node_h = 0.14 * w, 0.20 * h
    for center, label in zip(centers, labels, strict=True):
        node = FancyBboxPatch(
            (center[0] - node_w / 2, center[1] - node_h / 2),
            node_w,
            node_h,
            boxstyle="round,pad=0.002,rounding_size=0.003",
            facecolor="white",
            edgecolor=COLORS["model_edge"],
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(node)
        ax.text(center[0], center[1], label, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=COLORS["ink"], zorder=4)
    for left, right in zip(centers[:-1], centers[1:], strict=True):
        arrow(
            ax,
            (left[0] + node_w / 2, left[1]),
            (right[0] - node_w / 2, right[1]),
            color=COLORS["model_edge"],
            width=1.0,
            head=7,
        )
    skip_y = y + 0.67 * h
    line(ax, (centers[0][0], centers[0][1] + node_h / 2), (centers[0][0], skip_y),
         color=COLORS["model_edge"], width=0.9)
    line(ax, (centers[0][0], skip_y), (centers[4][0], skip_y),
         color=COLORS["model_edge"], width=0.9)
    arrow(ax, (centers[4][0], skip_y), (centers[4][0], centers[4][1] + node_h / 2),
          color=COLORS["model_edge"], width=0.9, head=6)
    ax.text(x + w / 2, y + 0.82 * h, "encoder   •   bottleneck   •   decoder",
            ha="center", va="center", fontsize=7.3, color=COLORS["ink"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 8), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y, h = 0.56, 0.29
    boxes = [
        (0.025, 0.145, COLORS["input"], COLORS["input_edge"], "Observable inputs",
         "4D BOLD PSC\nMeasured ETCO$_2$\nBOLD summary maps"),
        (0.215, 0.145, COLORS["model"], COLORS["model_edge"], "Temporal encoder",
         "Voxelwise 1D CNN\nLearns BOLD–CO$_2$\ndynamics"),
        (0.405, 0.155, COLORS["model"], COLORS["model_edge"], "Spatial model",
         ""),
        (0.605, 0.155, COLORS["output"], COLORS["output_edge"], "Parameter maps",
         r"$\widehat{\mathrm{CVR}}$   $\hat{\tau}$   $\hat{T}$" "\nUncertainty maps"),
        (0.805, 0.165, COLORS["physics"], COLORS["physics_edge"], "Physiological model",
         ""),
    ]
    for x, w, face, edge, title, body in boxes:
        box(ax, x, y, w, h, face, edge, title, body)

    ax.text(0.4825, y + 0.205, "Full-volume 3D U-Net", ha="center", va="center",
            fontsize=10.5, color=COLORS["ink"])
    draw_mini_unet(ax, 0.412, y + 0.025, 0.141, 0.16)

    ax.text(
        0.8875,
        y + 0.155,
        r"$\frac{d\hat y}{dt}=\frac{\widehat{\mathrm{CVR}}\,u(t-\hat\tau)-\hat y}{\hat T}$",
        ha="center",
        va="center",
        fontsize=17.5,
        color=COLORS["ink"],
    )
    ax.text(0.8875, y + 0.075, "Reconstructed BOLD PSC", ha="center", va="center",
            fontsize=11.5, color=COLORS["ink"])

    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        arrow(ax, (left[0] + left[1], y + h / 2), (right[0], y + h / 2))

    loss_x, loss_y, loss_w, loss_h = 0.60, 0.13, 0.365, 0.22
    box(
        ax, loss_x, loss_y, loss_w, loss_h,
        COLORS["loss"], COLORS["loss_edge"], "Self-supervised loss",
        r"$\mathcal{L}=\mathcal{L}_{\mathrm{Student\text{-}t\ reconstruction}}"
        r"+\lambda\mathcal{L}_{\mathrm{consistency}}$",
        title_size=14, body_size=16,
    )

    physics_center_x = boxes[-1][0] + boxes[-1][1] / 2
    arrow(
        ax, (physics_center_x, y), (physics_center_x, loss_y + loss_h),
        color=COLORS["loss_edge"],
    )

    input_center_x = boxes[0][0] + boxes[0][1] / 2
    route_y = 0.43
    line(ax, (input_center_x, y), (input_center_x, route_y), color=COLORS["input_edge"])
    line(ax, (input_center_x, route_y), (loss_x - 0.025, route_y), color=COLORS["input_edge"])
    line(
        ax, (loss_x - 0.025, route_y), (loss_x - 0.025, loss_y + loss_h / 2),
        color=COLORS["input_edge"],
    )
    arrow(
        ax, (loss_x - 0.025, loss_y + loss_h / 2), (loss_x, loss_y + loss_h / 2),
        color=COLORS["input_edge"], width=2.0, head=15,
    )
    ax.text(
        0.33, route_y + 0.018, "observed BOLD PSC",
        ha="center", va="bottom", fontsize=11, color=COLORS["input_edge"],
    )

    ax.text(
        0.30, 0.205,
        "Back-propagation updates the temporal encoder, 3D U-Net and parameter heads",
        ha="center", va="center", fontsize=11.5,
        color=COLORS["loss_edge"], fontweight="bold",
    )
    ax.text(
        0.50, 0.055,
        "Ground-truth parameter maps are hidden during training and used only for evaluation.",
        ha="center", va="center", fontsize=11.5,
        color=COLORS["ink"], fontweight="bold",
    )

    png = OUT_DIR / "current_self_supervised_cvr_architecture.png"
    pdf = OUT_DIR / "current_self_supervised_cvr_architecture.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
