from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create real-inference comparison diagrams from conventional CVR and PINN outputs."
    )
    parser.add_argument(
        "--real-inference-root",
        type=Path,
        default=Path(
            "data/vm_outputs/real_inference_unet_pinn_fullbrain_T_recovery_20260909_052143"
        ),
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--real-cvr-root", type=Path, default=Path("data/derivatives/real_cvr"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/vm_outputs/unet_pinn_fullbrain_T_recovery/real_inference_diagrams"),
    )
    parser.add_argument(
        "--conventional",
        choices=("glm", "hrf"),
        default="glm",
        help="Conventional CVR map to compare with PINN CVR.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional subject filter, e.g. sub-01. If omitted, all available sessions are plotted.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Optional session filter, e.g. ses-01. If omitted, all available sessions are plotted.",
    )
    parser.add_argument(
        "--save-difference-nifti",
        action="store_true",
        help="Also save observed minus reconstructed mean PSC NIfTI inside each real inference folder.",
    )
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--flip-vertical", action="store_true")
    parser.add_argument("--cvr-vmin", type=float, default=0.0)
    parser.add_argument("--cvr-vmax", type=float, default=0.7)
    return parser.parse_args()


def load(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32), img


def robust_limits(data: np.ndarray, mask: np.ndarray, lo: float = 2, hi: float = 98) -> tuple[float, float]:
    vals = np.asarray(data)[mask & np.isfinite(data)]
    if vals.size == 0:
        return 0.0, 1.0
    a, b = np.nanpercentile(vals, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or a == b:
        center = float(np.nanmedian(vals))
        return center - 1.0, center + 1.0
    return float(a), float(b)


def symmetric_limits(data: np.ndarray, mask: np.ndarray, percentile: float = 98) -> tuple[float, float]:
    vals = np.asarray(data)[mask & np.isfinite(data)]
    if vals.size == 0:
        return -1.0, 1.0
    bound = float(np.nanpercentile(np.abs(vals), percentile))
    bound = max(bound, 1e-3)
    return -bound, bound


def choose_axial_slice(mask: np.ndarray, affine: np.ndarray) -> tuple[int, int]:
    axcodes = tuple(str(code).upper() for code in nib.aff2axcodes(affine))
    axis = axcodes.index("S") if "S" in axcodes else 2
    counts = mask.sum(axis=tuple(i for i in range(mask.ndim) if i != axis))
    valid = np.flatnonzero(counts > 0)
    if valid.size == 0:
        return axis, mask.shape[axis] // 2
    return axis, int(valid[np.argmax(counts[valid])])


def slice2d(data: np.ndarray, axis: int, index: int, *, flip_vertical: bool = False) -> np.ndarray:
    sl = np.take(data, index, axis=axis)
    sl = np.rot90(sl)
    if flip_vertical:
        sl = np.flipud(sl)
    return sl


def draw_map(
    ax,
    data: np.ndarray,
    axis: int,
    index: int,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    flip_vertical: bool = False,
):
    im = ax.imshow(
        slice2d(data, axis, index, flip_vertical=flip_vertical),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=8)


def session_dirs(root: Path, subject: str | None, session: str | None) -> list[Path]:
    dirs = sorted(p for p in root.glob("sub-*/ses-*") if p.is_dir())
    if subject is not None:
        dirs = [p for p in dirs if p.parent.name == subject]
    if session is not None:
        dirs = [p for p in dirs if p.name == session]
    return dirs


def create_diagram(session_dir: Path, args: argparse.Namespace) -> Path:
    subject = session_dir.parent.name
    session = session_dir.name
    conventional_name = f"{args.conventional}_cvr.nii.gz"

    conventional_cvr, conventional_img = load(
        args.real_cvr_root / subject / session / conventional_name
    )
    pinn_cvr, _ = load(session_dir / "predicted_CVR.nii.gz")
    sigma, _ = load(session_dir / "predicted_uncertainty_sigma.nii.gz")
    residual, _ = load(session_dir / "residual_rms.nii.gz")
    recon_mean, _ = load(session_dir / "reconstructed_BOLD_PSC_mean.nii.gz")

    bold_img = nib.load(str(args.processed_root / subject / session / "bold_psc.nii.gz"))
    observed_mean = np.asarray(np.mean(np.asarray(bold_img.dataobj, dtype=np.float32), axis=3), dtype=np.float32)
    diff = observed_mean - recon_mean

    mask_path = args.processed_root / subject / session / "brain_mask.nii.gz"
    if mask_path.exists():
        mask = load(mask_path)[0] > 0
    else:
        mask = np.isfinite(pinn_cvr) & (sigma > 0)
    axis, auto_index = choose_axial_slice(mask, conventional_img.affine)
    index = int(args.slice_index) if args.slice_index is not None else auto_index

    if args.save_difference_nifti:
        out_img = nib.Nifti1Image(diff.astype(np.float32), bold_img.affine, bold_img.header)
        out_img.set_data_dtype(np.float32)
        nib.save(out_img, str(session_dir / "observed_minus_reconstructed_BOLD_PSC_mean.nii.gz"))

    cvr_vmin, cvr_vmax = float(args.cvr_vmin), float(args.cvr_vmax)
    sigma_vmin, sigma_vmax = robust_limits(sigma, mask, 1, 99)
    residual_vmin, residual_vmax = robust_limits(residual, mask, 1, 99)
    psc_vmin, psc_vmax = robust_limits(np.stack([observed_mean, recon_mean]), np.stack([mask, mask]), 2, 98)
    diff_vmin, diff_vmax = symmetric_limits(diff, mask, 98)

    fig = plt.figure(figsize=(15, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0])
    fig.suptitle(
        f"{subject}_{session} real inference comparison, axial slice {index}",
        fontsize=16,
    )

    draw_map(
        fig.add_subplot(gs[0, 0]),
        conventional_cvr,
        axis,
        index,
        title=f"Conventional {args.conventional.upper()} CVR",
        cmap="viridis",
        vmin=cvr_vmin,
        vmax=cvr_vmax,
        cbar_label="%BOLD/mmHg",
        flip_vertical=args.flip_vertical,
    )
    draw_map(
        fig.add_subplot(gs[0, 1]),
        pinn_cvr,
        axis,
        index,
        title="PINN CVR",
        cmap="viridis",
        vmin=cvr_vmin,
        vmax=cvr_vmax,
        cbar_label="%BOLD/mmHg",
        flip_vertical=args.flip_vertical,
    )
    draw_map(
        fig.add_subplot(gs[0, 2]),
        sigma,
        axis,
        index,
        title="Predicted uncertainty",
        cmap="magma",
        vmin=sigma_vmin,
        vmax=sigma_vmax,
        cbar_label="sigma",
        flip_vertical=args.flip_vertical,
    )
    draw_map(
        fig.add_subplot(gs[0, 3]),
        residual,
        axis,
        index,
        title="Residual RMS",
        cmap="inferno",
        vmin=residual_vmin,
        vmax=residual_vmax,
        cbar_label="PSC",
        flip_vertical=args.flip_vertical,
    )
    draw_map(
        fig.add_subplot(gs[1, 0]),
        observed_mean,
        axis,
        index,
        title="Observed BOLD PSC mean",
        cmap="viridis",
        vmin=psc_vmin,
        vmax=psc_vmax,
        cbar_label="PSC",
        flip_vertical=args.flip_vertical,
    )
    draw_map(
        fig.add_subplot(gs[1, 1]),
        recon_mean,
        axis,
        index,
        title="Reconstructed BOLD PSC mean",
        cmap="viridis",
        vmin=psc_vmin,
        vmax=psc_vmax,
        cbar_label="PSC",
        flip_vertical=args.flip_vertical,
    )
    draw_map(
        fig.add_subplot(gs[1, 2]),
        diff,
        axis,
        index,
        title="Observed - reconstructed BOLD",
        cmap="coolwarm",
        vmin=diff_vmin,
        vmax=diff_vmax,
        cbar_label="PSC",
        flip_vertical=args.flip_vertical,
    )
    note_ax = fig.add_subplot(gs[1, 3])
    note_ax.axis("off")
    note_ax.text(
        0.02,
        0.78,
        "Diagram notes",
        fontsize=12,
        fontweight="bold",
        transform=note_ax.transAxes,
    )
    note_ax.text(
        0.02,
        0.56,
        "Top row compares conventional CVR,\nPINN CVR, uncertainty, and residual.",
        fontsize=10,
        transform=note_ax.transAxes,
    )
    note_ax.text(
        0.02,
        0.34,
        "Bottom row checks whether the PINN\nreconstruction matches observed BOLD PSC.",
        fontsize=10,
        transform=note_ax.transAxes,
    )
    note_ax.text(
        0.02,
        0.16,
        "Difference map = observed mean PSC\nminus reconstructed mean PSC.",
        fontsize=10,
        transform=note_ax.transAxes,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / f"{subject}_{session}_real_inference_comparison.png"
    out_pdf = args.out_dir / f"{subject}_{session}_real_inference_comparison.pdf"
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_png


def main() -> None:
    args = parse_args()
    outputs = []
    for session_dir in session_dirs(args.real_inference_root, args.subject, args.session):
        try:
            outputs.append(create_diagram(session_dir, args))
            print(f"Saved {outputs[-1]}")
        except FileNotFoundError as exc:
            print(f"Skipping {session_dir}: {exc}")
    if not outputs:
        raise SystemExit("No diagrams were generated.")


if __name__ == "__main__":
    main()
