#!/usr/bin/env python3
"""Create a tight three-panel GT parameter-map figure for MIDA case_000."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 12,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def load_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(), dtype=np.float32), img.affine


def axial_axis(affine: np.ndarray) -> int:
    axcodes = tuple(str(code).upper() for code in nib.aff2axcodes(affine))
    return axcodes.index("S") if "S" in axcodes else 2


def slice2d(data: np.ndarray, axis: int, slice_index: int, *, flip_vertical: bool) -> np.ndarray:
    sl = np.take(data, slice_index, axis=axis)
    sl = np.rot90(sl)
    if flip_vertical:
        sl = np.flipud(sl)
    sl = np.rot90(sl, k=-1)
    sl = np.flipud(sl)
    return sl


def choose_axial_slice(support: np.ndarray, axis: int, requested: int | None) -> int:
    counts = support.sum(axis=tuple(i for i in range(support.ndim) if i != axis))
    if requested is not None:
        if requested < 0 or requested >= support.shape[axis]:
            raise ValueError(f"Requested axial slice {requested} is outside axis length {support.shape[axis]}")
        if counts[requested] > 0:
            return int(requested)
        print(
            f"Requested anatomical axial slice {requested} is empty; "
            f"using fullest axial slice {int(np.argmax(counts))} instead."
        )
    return int(np.argmax(counts))


def crop_bounds(support_slice: np.ndarray, margin: int = 2) -> tuple[slice, slice]:
    coords = np.argwhere(np.isfinite(support_slice) & (support_slice > 0))
    if coords.size == 0:
        return slice(None), slice(None)
    row_min, col_min = coords.min(axis=0)
    row_max, col_max = coords.max(axis=0)
    row_min = max(0, int(row_min) - margin)
    col_min = max(0, int(col_min) - margin)
    row_max = min(support_slice.shape[0] - 1, int(row_max) + margin)
    col_max = min(support_slice.shape[1] - 1, int(col_max) + margin)
    return slice(row_min, row_max + 1), slice(col_min, col_max + 1)


def masked(data: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    out[(support <= 0) | ~np.isfinite(out)] = np.nan
    return out


def draw_panel(
    ax,
    cax,
    data: np.ndarray,
    slice_index: int,
    *,
    label: str,
    cbar_label: str,
    vmin: float,
    vmax: float,
    flip_vertical: bool,
    crop: tuple[slice, slice],
    axis: int,
):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    sl = slice2d(data, axis, slice_index, flip_vertical=flip_vertical)
    im = ax.imshow(
        sl[crop],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(label, fontweight="bold", labelpad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)


def make_plot(args: argparse.Namespace) -> None:
    case_dir = args.case_dir
    cvr, affine = load_map(case_dir / "GT_CVR.nii.gz")
    delay, _ = load_map(case_dir / "GT_delay.nii.gz")
    response_t, _ = load_map(case_dir / "GT_T.nii.gz")
    region_labels, _ = load_map(case_dir / "GT_region_labels.nii.gz")

    support = region_labels > 0
    axis = axial_axis(affine)
    slice_index = choose_axial_slice(support, axis, args.slice_index)
    cvr = masked(cvr, support)
    delay = masked(delay, support)
    response_t = masked(response_t, support)

    support_display = slice2d(support.astype(np.float32), axis, slice_index, flip_vertical=args.flip_vertical)
    crop = crop_bounds(support_display, margin=1)

    fig = plt.figure(figsize=(7.0, 2.45))
    gs = fig.add_gridspec(
        1,
        8,
        width_ratios=[1.0, 0.035, 0.065, 1.0, 0.035, 0.065, 1.0, 0.035],
        wspace=0.0,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 3, 6)]
    caxes = [fig.add_subplot(gs[0, i]) for i in (1, 4, 7)]
    draw_panel(
        axes[0],
        caxes[0],
        cvr,
        slice_index,
        label="GT CVR",
        cbar_label="%/mmHg",
        vmin=0.0,
        vmax=0.7,
        flip_vertical=args.flip_vertical,
        crop=crop,
        axis=axis,
    )
    draw_panel(
        axes[1],
        caxes[1],
        delay,
        slice_index,
        label="GT Delay",
        cbar_label="s",
        vmin=0.0,
        vmax=80.0,
        flip_vertical=args.flip_vertical,
        crop=crop,
        axis=axis,
    )
    draw_panel(
        axes[2],
        caxes[2],
        response_t,
        slice_index,
        label="GT T",
        cbar_label="s",
        vmin=2.0,
        vmax=100.0,
        flip_vertical=args.flip_vertical,
        crop=crop,
        axis=axis,
    )

    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.12)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / f"case_000_gt_parameter_maps_anatomical_axial{slice_index:02d}.png"
    pdf = args.out_dir / f"case_000_gt_parameter_maps_anatomical_axial{slice_index:02d}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.005)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.005)
    plt.close(fig)

    print(f"Saved {png}")
    print(f"Saved {pdf}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir", type=Path, default=root / "data/simulated/mida_parameters/case_000")
    ap.add_argument("--out-dir", type=Path, default=root / "data/figures/presentation")
    ap.add_argument("--slice-index", type=int, default=54)
    ap.add_argument("--flip-vertical", action=argparse.BooleanOptionalAction, default=True)
    return ap.parse_args()


if __name__ == "__main__":
    make_plot(parse_args())
