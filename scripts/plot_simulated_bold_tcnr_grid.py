#!/usr/bin/env python3
"""Plot simulated 4D BOLD axial slices across tCNR levels."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


TCNR_ORDER = ["0p1", "0p2", "0p5", "1", "2", "5", "10"]


def tcnr_label(token: str) -> str:
    return token.replace("p", ".")


def parse_tcnr(path: Path) -> str:
    match = re.search(r"tcnr-([^_]+)_bold", path.name)
    if not match:
        raise ValueError(f"Could not parse tCNR from {path.name}")
    return match.group(1)


def axial_axis(affine: np.ndarray) -> int:
    axcodes = tuple(str(code).upper() for code in nib.aff2axcodes(affine))
    return axcodes.index("S") if "S" in axcodes else 2


def display_slice(volume: np.ndarray, affine: np.ndarray, slice_index: int, *, flip_vertical: bool) -> np.ndarray:
    axis = axial_axis(affine)
    sl = np.take(volume, slice_index, axis=axis)
    sl = np.rot90(sl)
    if flip_vertical:
        sl = np.flipud(sl)
    sl = np.rot90(sl, k=-1)
    sl = np.flipud(sl)
    return sl


def crop_bounds(slices: list[np.ndarray], margin: int = 2) -> tuple[slice, slice]:
    support = np.zeros_like(slices[0], dtype=bool)
    for sl in slices:
        finite = np.isfinite(sl)
        vals = sl[finite]
        if vals.size == 0:
            continue
        threshold = max(1e-6, float(np.nanpercentile(np.abs(vals), 1)))
        support |= finite & (np.abs(sl) > threshold)
    coords = np.argwhere(support)
    if coords.size == 0:
        return slice(None), slice(None)
    row_min, col_min = coords.min(axis=0)
    row_max, col_max = coords.max(axis=0)
    row_min = max(0, int(row_min) - margin)
    col_min = max(0, int(col_min) - margin)
    row_max = min(support.shape[0] - 1, int(row_max) + margin)
    col_max = min(support.shape[1] - 1, int(col_max) + margin)
    return slice(row_min, row_max + 1), slice(col_min, col_max + 1)


def load_slice(path: Path, time_index: int, slice_index: int, flip_vertical: bool) -> np.ndarray:
    img = nib.load(str(path))
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4D NIfTI for {path}, got shape {img.shape}")
    if time_index < 0 or time_index >= img.shape[3]:
        raise ValueError(f"time_index {time_index} outside {path.name} time length {img.shape[3]}")
    volume = np.asanyarray(img.dataobj[..., time_index], dtype=np.float32)
    return display_slice(volume, img.affine, slice_index, flip_vertical=flip_vertical)


def make_plot(args: argparse.Namespace) -> None:
    pattern = f"sim_*_{args.paradigm}_tcnr-*_bold.nii.gz"
    paths = sorted(args.bold_dir.glob(pattern), key=lambda p: TCNR_ORDER.index(parse_tcnr(p)))
    if len(paths) != len(TCNR_ORDER):
        found = ", ".join(p.name for p in paths)
        raise FileNotFoundError(f"Expected {len(TCNR_ORDER)} {args.paradigm} simulations, found {len(paths)}: {found}")

    slices = [load_slice(p, args.time_index, args.slice_index, args.flip_vertical) for p in paths]
    crop = crop_bounds(slices, margin=1)
    cropped = [sl[crop] for sl in slices]

    vals = np.concatenate([sl[np.isfinite(sl)].ravel() for sl in cropped])
    vmin = float(np.nanpercentile(vals, args.low_percentile))
    vmax = float(np.nanpercentile(vals, args.high_percentile))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))

    fig = plt.figure(figsize=(4.2, 5.1))
    gs = fig.add_gridspec(3, 6, wspace=0.01, hspace=0.06)
    positions = [
        (0, slice(0, 2)),
        (0, slice(2, 4)),
        (0, slice(4, 6)),
        (1, slice(1, 3)),
        (1, slice(3, 5)),
        (2, slice(1, 3)),
        (2, slice(3, 5)),
    ]

    for path, sl, pos in zip(paths, cropped, positions, strict=True):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"tCNR {tcnr_label(parse_tcnr(path))}", fontweight="bold", labelpad=0, fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.045, wspace=0.0, hspace=0.02)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / f"simulated_{args.paradigm}_bold_axial{args.slice_index}_tcnr_grid.png"
    pdf = args.out_dir / f"simulated_{args.paradigm}_bold_axial{args.slice_index}_tcnr_grid.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.005)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)

    print(f"Saved {png}")
    print(f"Saved {pdf}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bold-dir", type=Path, default=root / "data/simulated/bold4d")
    ap.add_argument("--out-dir", type=Path, default=root / "data/vm_outputs/unet_pinn_fullbrain_T_recovery/time_series_concepts")
    ap.add_argument("--paradigm", default="block")
    ap.add_argument("--slice-index", type=int, default=54)
    ap.add_argument("--time-index", type=int, default=320)
    ap.add_argument("--low-percentile", type=float, default=2.0)
    ap.add_argument("--high-percentile", type=float, default=98.0)
    ap.add_argument("--flip-vertical", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


if __name__ == "__main__":
    make_plot(parse_args())
