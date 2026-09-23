from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARADIGMS = ("block", "multi_step", "pseudo_random_binary")
PARADIGM_LABELS = ("Block", "Multi-step", "Pseudo-random binary")
TCNRS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def matrix(rows: list[dict[str, str]], method: str) -> np.ndarray:
    values = np.full((len(PARADIGMS), len(TCNRS)), np.nan, dtype=np.float64)
    for row in rows:
        if row["split"] != "test" or row["method"] != method:
            continue
        paradigm = row["paradigm"]
        tcnr = float(row["tcnr"])
        if paradigm in PARADIGMS and tcnr in TCNRS:
            values[PARADIGMS.index(paradigm), TCNRS.index(tcnr)] = float(row["cvr_mae"])
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Incomplete test CVR MAE grid for {method}")
    return values


def annotate(ax, image, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            rgba = image.cmap(image.norm(values[row, col]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                col,
                row,
                f"{values[row, col]:.3f}",
                ha="center",
                va="center",
                color="black" if luminance > 0.55 else "white",
                fontsize=12,
                fontweight="bold",
            )


def main() -> None:
    args = parse_args()
    with args.metrics.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    neural = matrix(rows, "self_supervised")
    hrf = matrix(rows, "exponential_hrf")
    vmin = float(min(neural.min(), hrf.min()))
    vmax = float(max(neural.max(), hrf.max()))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8), constrained_layout=True)
    images = []
    for col, (title, values) in enumerate((("Neural network", neural), ("Exponential ODE-HRF", hrf))):
        ax = axes[col]
        image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        images.append(image)
        annotate(ax, image, values)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xticks(range(len(TCNRS)), [f"{value:g}" for value in TCNRS], fontsize=11)
        ax.set_yticks(range(len(PARADIGMS)), PARADIGM_LABELS, fontsize=11)
        ax.set_xlabel("tCNR", fontsize=12)
        if col == 0:
            ax.set_ylabel("ETCO2 paradigm", fontsize=12)
        else:
            ax.set_yticklabels([])
    colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("CVR MAE (%BOLD/mmHg)", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
