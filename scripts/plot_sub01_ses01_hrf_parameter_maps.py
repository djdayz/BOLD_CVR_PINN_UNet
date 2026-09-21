#!/usr/bin/env python3
"""Create a tight three-panel HRF parameter-map figure for sub-01/ses-01."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 11,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def load_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=np.float32), img.affine


def axial_slice(data: np.ndarray, affine: np.ndarray, index: int, *, flip_vertical: bool) -> np.ndarray:
    axcodes = tuple(str(code).upper() for code in nib.aff2axcodes(affine))
    axis = axcodes.index("S") if "S" in axcodes else 2
    sl = np.take(data, index, axis=axis)
    sl = np.rot90(sl)
    if flip_vertical:
        sl = np.flipud(sl)
    return sl


def masked(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    out[(mask <= 0.5) | ~np.isfinite(out)] = np.nan
    return out


def draw_panel(
    ax,
    data: np.ndarray,
    affine: np.ndarray,
    slice_index: int,
    *,
    label: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    flip_vertical: bool,
):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    im = ax.imshow(
        axial_slice(data, affine, slice_index, flip_vertical=flip_vertical),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(label, fontweight="bold", labelpad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.015)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)


def make_plot(args: argparse.Namespace) -> None:
    cvr, affine = load_map(args.hrf_cvr)
    delay, _ = load_map(args.hrf_delay)
    response_t, _ = load_map(args.hrf_t)
    brain_mask, _ = load_map(args.brain_mask)

    cvr = masked(cvr, brain_mask)
    delay = masked(delay, brain_mask)
    response_t = masked(response_t, brain_mask)

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.65))
    draw_panel(
        axes[0],
        cvr,
        affine,
        args.slice_index,
        label="HRF CVR",
        cbar_label="%/mmHg",
        vmin=0.0,
        vmax=0.7,
        flip_vertical=args.flip_vertical,
    )
    draw_panel(
        axes[1],
        delay,
        affine,
        args.slice_index,
        label="Delay",
        cbar_label="s",
        vmin=0.0,
        vmax=80.0,
        flip_vertical=args.flip_vertical,
    )
    draw_panel(
        axes[2],
        response_t,
        affine,
        args.slice_index,
        label="T",
        cbar_label="s",
        vmin=2.0,
        vmax=100.0,
        flip_vertical=args.flip_vertical,
    )

    fig.subplots_adjust(left=0.01, right=0.995, top=0.995, bottom=0.13, wspace=0.03)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "sub-01_ses-01_hrf_parameter_maps_axial25.png"
    pdf = args.out_dir / "sub-01_ses-01_hrf_parameter_maps_axial25.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    print(f"Saved {png}")
    print(f"Saved {pdf}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hrf-cvr", type=Path, default=root / "data/derivatives/real_cvr/sub-01/ses-01/hrf_cvr.nii.gz")
    ap.add_argument("--hrf-delay", type=Path, default=root / "data/derivatives/real_cvr/sub-01/ses-01/hrf_delay.nii.gz")
    ap.add_argument("--hrf-t", type=Path, default=root / "data/derivatives/real_cvr/sub-01/ses-01/hrf_T.nii.gz")
    ap.add_argument("--brain-mask", type=Path, default=root / "data/processed/sub-01/ses-01/brain_mask.nii.gz")
    ap.add_argument("--out-dir", type=Path, default=root / "data/figures/presentation")
    ap.add_argument("--slice-index", type=int, default=25)
    ap.add_argument("--flip-vertical", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


if __name__ == "__main__":
    make_plot(parse_args())
