from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str = "#30343b",
    linewidth: float = 1.8,
    radius: float = 0.045,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def straight_arrow(ax, xy1, xy2, *, lw: float = 2.0, color: str = "#30343b", ms: float = 15.0):
    patch = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0",
        shrinkA=6,
        shrinkB=6,
    )
    ax.add_patch(patch)
    return patch


def add_center_text(ax, x, y, text, *, size=10, weight="bold", color="#17212b", linespacing=1.25):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color=color,
        linespacing=linespacing,
    )


def draw_input_block(ax):
    x, y, w, h = 0.35, 3.25, 2.25, 2.55
    rounded_box(ax, x, y, w, h, facecolor="#e7f0fb")
    add_center_text(ax, x + w / 2, y + h - 0.38, "Observable image\nfeatures", size=10.5, linespacing=1.0)
    add_center_text(ax, x + w / 2, y + h - 0.94, "15 spatial channels", size=11)
    ax.text(
        x + 0.24,
        y + 0.95,
        "\n".join(
            [
                "BOLD signal statistics",
                "baseline / hypercapnic PSC",
                "tCNR",
                "tissue fractions",
                "vessel likelihood",
                "boundary uncertainty",
            ]
        ),
        ha="left",
        va="center",
        fontsize=8.2,
        color="#263238",
        linespacing=1.18,
    )
    return x + w, y + h / 2


def draw_unet_block(ax):
    x, y, w, h = 3.25, 2.75, 2.65, 3.25
    rounded_box(ax, x, y, w, h, facecolor="#e8f5e3", edgecolor="#2e6f40", linewidth=2.1)
    add_center_text(ax, x + w / 2, y + h - 0.38, "2D U-Net", size=12)
    add_center_text(ax, x + w / 2, y + 0.35, "shared spatial representation", size=8.5, weight="bold")

    enc_x = x + 0.34
    dec_x = x + w - 0.84
    levels_y = [y + 2.35, y + 1.75, y + 1.15]
    sizes = [0.50, 0.42, 0.34]
    enc_centers = []
    dec_centers = []
    for yy, s in zip(levels_y, sizes, strict=True):
        ax.add_patch(Rectangle((enc_x, yy - s / 2), s, s, facecolor="#7fc97f", edgecolor="#2e6f40", linewidth=1.2))
        ax.add_patch(Rectangle((dec_x, yy - s / 2), s, s, facecolor="#b2df8a", edgecolor="#2e6f40", linewidth=1.2))
        enc_centers.append((enc_x + s / 2, yy))
        dec_centers.append((dec_x + s / 2, yy))
        ax.plot([enc_x + s + 0.05, dec_x - 0.05], [yy, yy], color="#2e6f40", linewidth=1.1, alpha=0.65)

    bottleneck = (x + w / 2, y + 0.98)
    ax.add_patch(Rectangle((bottleneck[0] - 0.25, bottleneck[1] - 0.17), 0.50, 0.34, facecolor="#4daf4a", edgecolor="#2e6f40", linewidth=1.2))
    left_x = enc_centers[0][0]
    right_x = dec_centers[0][0]
    mid_y = bottleneck[1]
    ax.plot(
        [left_x, left_x, bottleneck[0], right_x, right_x],
        [enc_centers[0][1], mid_y, mid_y, mid_y, dec_centers[0][1]],
        color="#1b5e20",
        linewidth=2.2,
    )
    return x + w, y + h / 2


def draw_heads(ax):
    x, y = 6.65, 2.85
    w, h = 1.16, 0.62
    heads = [
        (x, y + 2.25, r"$\widehat{\mathrm{CVR}}$", ""),
        (x, y + 1.50, r"$\hat{\tau}$", ""),
        (x, y + 0.75, r"$\hat{T}$", ""),
        (x, y + 0.00, r"$\hat{\sigma}$", "uncertainty"),
    ]
    for hx, hy, label, subtitle in heads:
        rounded_box(ax, hx, hy, w, h, facecolor="#efe5fb", edgecolor="#6a3d9a", linewidth=1.7, radius=0.035)
        add_center_text(ax, hx + w / 2, hy + h / 2 + (0.08 if subtitle else 0.0), label, size=13)
        if subtitle:
            ax.text(hx + w / 2, hy + 0.13, subtitle, ha="center", va="center", fontsize=7.2, color="#3f2a56")
    ax.text(x + w / 2, y + 3.16, "Parameter heads", ha="center", va="center", fontsize=10, fontweight="bold")
    return x, x + w, y + 1.54


def draw_physics(ax):
    x, y, w, h = 8.55, 3.15, 2.95, 2.65
    rounded_box(ax, x, y, w, h, facecolor="#fff4c7", edgecolor="#a57600", linewidth=2.6, radius=0.045)
    add_center_text(ax, x + w / 2, y + h - 0.45, "Differentiable\nphysiological model", size=11, linespacing=1.1)
    ax.text(
        x + w / 2,
        y + 1.08,
        r"$\frac{d\hat{y}}{dt} = \frac{\widehat{\mathrm{CVR}}\,u(t-\hat{\tau})-\hat{y}}{\hat{T}}$",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#17212b",
    )

    ax.text(x + w / 2, y + h + 0.82, r"Measured ETCO$_2$", ha="center", va="center", fontsize=10, fontweight="bold")
    trace_x = np.linspace(x + 0.38, x + w - 0.38, 140)
    trace_y = y + h + 0.42 + 0.10 * np.sin(np.linspace(0, 4 * np.pi, 140))
    trace_y += 0.19 * (np.linspace(0, 1, 140) > 0.43)
    ax.plot(trace_x, trace_y, color="#1f77b4", linewidth=2.1)
    straight_arrow(ax, (x + w / 2, y + h + 0.30), (x + w / 2, y + h - 0.06), lw=1.9, color="#1f77b4")
    return x + w, y + h / 2


def draw_reconstruction(ax):
    x, y, w, h = 12.25, 3.00, 2.35, 2.85
    rounded_box(ax, x, y, w, h, facecolor="#f4f4f4", edgecolor="#30343b", linewidth=1.9)
    add_center_text(ax, x + w / 2, y + h - 0.34, r"$\hat{y}(t)$", size=14)
    add_center_text(ax, x + w / 2, y + h - 0.70, "Reconstructed BOLD PSC", size=8.5)

    tx = np.linspace(x + 0.32, x + w - 0.32, 130)
    yy = y + 1.38 + 0.13 * np.sin(np.linspace(0, 3.7 * np.pi, 130))
    yy += 0.45 * np.exp(-0.5 * ((np.linspace(-2, 2, 130) - 0.55) / 0.6) ** 2)
    ax.plot(tx, yy, color="#d62728", linewidth=2.2)

    ax.text(x + w / 2, y + 0.78, r"Observed BOLD $y(t)$", ha="center", va="center", fontsize=8.5, fontweight="bold")
    oy = y + 0.34 + 0.10 * np.sin(np.linspace(0.3, 4.0 * np.pi, 130))
    oy += 0.33 * np.exp(-0.5 * ((np.linspace(-2, 2, 130) - 0.55) / 0.75) ** 2)
    ax.plot(tx, oy, color="#555555", linewidth=1.8)
    return x, y + h / 2


def draw_loss(ax):
    x, y, w, h = 1.10, 0.45, 13.00, 1.05
    rounded_box(ax, x, y, w, h, facecolor="#f9e0df", edgecolor="#9b2d26", linewidth=2.0, radius=0.035)
    loss = (
        r"$\mathcal{L} = "
        r"\lambda_{\rm recon}\mathcal{L}_{\rm recon} + "
        r"\lambda_{\rm ODE}\mathcal{L}_{\rm ODE} + "
        r"\lambda_{\rm unc}\mathcal{L}_{\rm unc} + "
        r"\lambda_{\rm smooth}\mathcal{L}_{\rm smooth} + "
        r"\lambda_{\rm prior}\mathcal{L}_{\rm prior}$"
    )
    ax.text(x + w / 2, y + h / 2, loss, ha="center", va="center", fontsize=18, fontweight="bold")


def main() -> None:
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16.0, 8.0))
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    input_out = draw_input_block(ax)
    unet_out = draw_unet_block(ax)
    heads_in, heads_out, heads_mid_y = draw_heads(ax)
    physics_out = draw_physics(ax)
    recon_in = draw_reconstruction(ax)
    draw_loss(ax)

    straight_arrow(ax, input_out, (3.25, input_out[1]), lw=2.1)
    straight_arrow(ax, unet_out, (heads_in, heads_mid_y), lw=2.1)
    for y in [5.10, 4.35, 3.60, 2.85]:
        straight_arrow(ax, (heads_out, y), (8.55, 4.48), lw=1.6, color="#5d4377", ms=12.0)
    straight_arrow(ax, physics_out, recon_in, lw=2.2)

    fig.savefig(out_dir / "hybrid_unet_pinn_architecture_flow.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_dir / "hybrid_unet_pinn_architecture_flow.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Saved {out_dir / 'hybrid_unet_pinn_architecture_flow.png'}")
    print(f"Saved {out_dir / 'hybrid_unet_pinn_architecture_flow.pdf'}")


if __name__ == "__main__":
    main()
