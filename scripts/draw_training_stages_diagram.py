from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, title, subtitle, epoch_text, color, edge):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.05",
        facecolor=color,
        edgecolor=edge,
        linewidth=2.0,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=15, fontweight="bold", color="#17212b")
    ax.text(x + w / 2, y + h - 0.74, subtitle, ha="center", va="center", fontsize=9.0, color="#263238", linespacing=1.15)
    ax.text(x + w / 2, y + 0.28, epoch_text, ha="center", va="center", fontsize=10, fontweight="bold", color=edge)


def straight_arrow(ax, xy1, xy2, *, lw=4.2, color="#2f343b", mutation_scale=22):
    arrow = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0",
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(arrow)


def main() -> None:
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16.2, 3.5))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    # One continuous curriculum arrow behind the four close stage blocks.
    straight_arrow(ax, (0.42, 1.72), (15.80, 1.72), lw=5.0, color="#30343b", mutation_scale=24)

    stages = [
        ("Stage 1", "Clean signal", "high tCNR • simple paradigms", "40 epochs", "#e7f0fb", "#2f5f91"),
        ("Stage 2", "Physics", "stronger ODE constraint\n tissuewise T", "80 epochs", "#fff4c7", "#a57600"),
        ("Stage 3", "Robust mixed training", "all paradigms • broader tCNR\n voxelwise T", "180 epochs", "#e8f5e3", "#2e6f40"),
        ("Stage 4", "Stress test", "very low tCNR + artefacts", "60 epochs", "#f9e0df", "#9b2d26"),
    ]

    xs = [0.65, 4.45, 8.25, 12.05]
    w, h, y = 3.35, 1.75, 0.85
    for i, (stage, title, subtitle, epochs, color, edge) in enumerate(stages):
        box(ax, xs[i], y, w, h, title, subtitle, epochs, color, edge)
        ax.text(xs[i] + w / 2, y + h + 0.34, stage, ha="center", va="center", fontsize=11.5, fontweight="bold", color=edge)

    # Short visible arrow segments between boxes keep the flow explicit.
    for i in range(3):
        straight_arrow(ax, (xs[i] + w + 0.06, 1.72), (xs[i + 1] - 0.08, 1.72), lw=3.2, color="#30343b", mutation_scale=18)

    fig.savefig(out_dir / "training_stages_horizontal.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / "training_stages_horizontal.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Saved {out_dir / 'training_stages_horizontal.png'}")
    print(f"Saved {out_dir / 'training_stages_horizontal.pdf'}")


if __name__ == "__main__":
    main()
