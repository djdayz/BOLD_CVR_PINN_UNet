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
METRICS = (
    ("cvr_mae", "CVR MAE", ".3f"),
    ("delay_mae", "Delay MAE (s)", ".2f"),
    ("T_mae", "T MAE (s)", ".2f"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["method"] == "self_supervised"]


def matrix(rows: list[dict[str, str]], split: str, metric: str) -> np.ndarray:
    values = np.full((len(PARADIGMS), len(TCNRS)), np.nan, dtype=np.float64)
    for row in rows:
        if row["split"] != split:
            continue
        paradigm = row["paradigm"]
        tcnr = float(row["tcnr"])
        if paradigm in PARADIGMS and tcnr in TCNRS:
            values[PARADIGMS.index(paradigm), TCNRS.index(tcnr)] = float(row[metric])
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Missing {split} values for {metric}")
    return values


def annotate(ax, image, values: np.ndarray, number_format: str) -> None:
    cmap = image.cmap
    norm = image.norm
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            rgba = cmap(norm(values[row, col]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.55 else "white"
            ax.text(
                col,
                row,
                format(values[row, col], number_format),
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics)
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10.2), constrained_layout=True)

    for metric_row, (metric, label, number_format) in enumerate(METRICS):
        validation = matrix(rows, "validation", metric)
        test = matrix(rows, "test", metric)
        vmin = float(min(validation.min(), test.min()))
        vmax = float(max(validation.max(), test.max()))
        for split_col, (split_title, values) in enumerate(
            (("Validation", validation), ("Test", test))
        ):
            ax = axes[metric_row, split_col]
            image = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            annotate(ax, image, values, number_format)
            ax.set_xticks(range(len(TCNRS)), [f"{value:g}" for value in TCNRS], fontsize=11)
            ax.set_yticks(range(len(PARADIGMS)), PARADIGM_LABELS, fontsize=11)
            ax.set_xlabel("tCNR", fontsize=12)
            if split_col == 0:
                ax.set_ylabel(label, fontsize=12, fontweight="bold")
            else:
                ax.set_yticklabels([])
            if metric_row == 0:
                ax.set_title(split_title, fontsize=15, fontweight="bold")
            colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            colorbar.ax.tick_params(labelsize=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
