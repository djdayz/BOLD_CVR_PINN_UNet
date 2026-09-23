#!/usr/bin/env python3
"""Plot the three ETCO2 paradigms used by the current simulation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hybrid_cvr.simulation.co2_paradigms import make_paradigm


PARADIGMS = (
    ("block", "Block"),
    ("multi_step", "Multi-step"),
    ("pseudo_random_binary", "Pseudo-random binary"),
)


def make_plot(out_dir: Path, *, seed: int, n_timepoints: int, tr_seconds: float) -> None:
    time = np.arange(n_timepoints, dtype=np.float64) * tr_seconds
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.0, 8.4),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.16},
    )

    for axis, (key, label) in zip(axes, PARADIGMS, strict=True):
        trace = np.asarray(make_paradigm(key, time, seed=seed), dtype=np.float64)
        axis.plot(time, trace, color="#1677b8", linewidth=2.5)
        axis.axhline(0.0, color="0.35", linewidth=1.1, linestyle="--")
        panel_title = f"{label} (example seed {seed})" if key == "pseudo_random_binary" else label
        axis.set_title(panel_title, fontweight="bold", fontsize=12, pad=5)
        axis.set_ylabel("ΔETCO2\n(mmHg)")
        axis.set_ylim(-0.8, 11.0)
        axis.grid(True, color="0.90", linewidth=0.8)
        axis.margins(x=0)

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(float(time[0]), float(time[-1]))
    axes[-1].set_xticks(np.arange(0.0, 701.0, 100.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.08, hspace=0.18)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "simulation_etco2_paradigms_3x1.png"
    pdf = out_dir / "simulation_etco2_paradigms_3x1.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=root / "data/figures/presentation")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--n-timepoints", type=int, default=480)
    parser.add_argument("--tr-seconds", type=float, default=1.55)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_plot(
        args.out_dir,
        seed=args.seed,
        n_timepoints=args.n_timepoints,
        tr_seconds=args.tr_seconds,
    )
