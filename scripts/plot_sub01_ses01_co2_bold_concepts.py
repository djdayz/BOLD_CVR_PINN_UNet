#!/usr/bin/env python3
"""Plot full-duration ETCO2 and cortical-GM BOLD PSC concept traces."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
    }
)


def load_etco2(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        rows.extend(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    return {
        "time": np.array([float(r["time"]) for r in rows], dtype=np.float64),
        "etco2": np.array([float(r["etco2_mmhg"]) for r in rows], dtype=np.float64),
        "delta": np.array([float(r["delta_etco2_mmhg"]) for r in rows], dtype=np.float64),
        "baseline": np.array([float(r["baseline_mmhg"]) for r in rows], dtype=np.float64),
    }


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def select_cortical_roi(
    mask: np.ndarray,
    cvr: np.ndarray,
    delay: np.ndarray,
    response_t: np.ndarray,
    r2: np.ndarray,
    slice_index: int,
    max_voxels: int,
) -> np.ndarray:
    roi = np.zeros(mask.shape, dtype=bool)
    roi[:, :, slice_index] = mask[:, :, slice_index] > 0.5

    valid = (
        roi
        & np.isfinite(cvr)
        & np.isfinite(delay)
        & np.isfinite(response_t)
        & np.isfinite(r2)
        & (cvr > 0.02)
        & (delay >= 5.0)
        & (delay <= 25.0)
        & (response_t >= 4.0)
        & (response_t <= 60.0)
        & (r2 > 0.10)
    )
    if int(valid.sum()) < 8:
        valid = (
            roi
            & np.isfinite(cvr)
            & np.isfinite(delay)
            & np.isfinite(response_t)
            & np.isfinite(r2)
            & (cvr > 0.0)
            & (delay >= 3.0)
            & (delay <= 35.0)
            & (response_t >= 3.0)
            & (response_t <= 80.0)
            & (r2 > 0.05)
        )
    if int(valid.sum()) == 0:
        raise ValueError(f"No cortical-GM ROI voxels found on slice {slice_index}")

    coords = np.argwhere(valid)
    scores = r2[valid]
    keep = np.argsort(scores)[-min(max_voxels, coords.shape[0]) :]
    selected = np.zeros(mask.shape, dtype=bool)
    selected[tuple(coords[keep].T)] = True
    return selected


def find_clear_co2_rise(time: np.ndarray, delta: np.ndarray) -> int:
    smoothed = gaussian_filter1d(delta, sigma=2.0)
    deriv = np.gradient(smoothed, time)
    candidate = (time > 60.0) & (time < time[-1] - 120.0)
    if not np.any(candidate):
        return int(np.argmax(deriv))
    candidate_indices = np.where(candidate)[0]
    return int(candidate_indices[np.argmax(deriv[candidate])])


def bold_at(time: np.ndarray, bold_trace: np.ndarray, target_time: float) -> float:
    return float(np.interp(target_time, time, bold_trace))


def make_plot(args: argparse.Namespace) -> None:
    et = load_etco2(args.etco2)
    time = et["time"]
    etco2 = et["etco2"]
    delta = et["delta"]
    baseline = et["baseline"]

    bold = load_nifti(args.bold_psc)
    cgm = load_nifti(args.cortical_gm_mask)
    cvr = load_nifti(args.hrf_cvr)
    delay = load_nifti(args.hrf_delay)
    response_t = load_nifti(args.hrf_t)
    r2 = load_nifti(args.hrf_r2)

    if bold.shape[:3] != cgm.shape:
        raise ValueError(f"BOLD shape {bold.shape[:3]} does not match mask shape {cgm.shape}")
    if bold.shape[3] != time.size:
        raise ValueError(f"BOLD time points {bold.shape[3]} do not match ETCO2 rows {time.size}")

    selected = select_cortical_roi(cgm, cvr, delay, response_t, r2, args.slice_index, args.roi_voxels)
    bold_roi = np.nanmean(bold[selected, :], axis=0)
    bold_roi = gaussian_filter1d(bold_roi, sigma=args.bold_smooth_sigma)

    roi_delay = float(np.nanmedian(delay[selected]))
    roi_t = float(np.nanmedian(response_t[selected]))
    roi_cvr = float(np.nanmedian(cvr[selected]))

    if args.transition_time is None:
        rise_idx = find_clear_co2_rise(time, delta)
    else:
        rise_idx = int(np.argmin(np.abs(time - args.transition_time)))
    t_co2 = float(time[rise_idx])
    t_bold_start = min(float(time[-1]), t_co2 + roi_delay)
    t_bold_late = min(float(time[-1]), t_bold_start + roi_t)

    pre_mask = (time >= max(float(time[0]), t_co2 - 45.0)) & (time < t_co2)
    y_pre = float(np.nanmedian(bold_roi[pre_mask])) if np.any(pre_mask) else bold_at(time, bold_roi, t_co2)
    y_start = bold_at(time, bold_roi, t_bold_start)
    y_late = bold_at(time, bold_roi, t_bold_late)

    # Keep the amplitude annotation within the first hypercapnic response while
    # allowing enough time to reach its true BOLD peak.
    cvr_window = (
        (time >= t_bold_start)
        & (time <= min(float(time[-1]), t_bold_start + max(6.0 * roi_t, 150.0)))
    )
    if np.any(cvr_window):
        cvr_indices = np.where(cvr_window)[0]
        amp_idx = int(cvr_indices[np.nanargmax(bold_roi[cvr_window])])
    else:
        amp_idx = int(np.argmin(np.abs(time - t_bold_late)))
    t_amp = float(time[amp_idx])
    y_amp = float(bold_roi[amp_idx])
    y_arrow_low, y_arrow_high = sorted([y_pre, y_amp])
    if abs(y_arrow_high - y_arrow_low) < 0.2:
        y_arrow_low = y_pre
        y_arrow_high = y_pre + max(0.25, 0.08 * float(np.nanmax(bold_roi) - np.nanmin(bold_roi)))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(6.7, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.2], "hspace": 0.13},
    )

    et_line, = ax_top.plot(time, etco2, color="#1f77b4", lw=2.2, label="CO2 stimulus, u(t): ETCO2", zorder=3)
    base_line, = ax_top.plot(time, baseline, color="0.35", lw=1.8, ls="--", label="Baseline ETCO2")
    ax_top.set_ylabel("ETCO2 (mmHg)")
    ax_top.grid(True, color="0.90", linewidth=0.8)

    ax_delta = ax_top.twinx()
    delta_line, = ax_delta.plot(time, delta, color="#2ca02c", lw=1.8, ls=":", alpha=0.70, label="Delta ETCO2")
    ax_delta.axhline(0.0, color="#2ca02c", lw=1.0, alpha=0.35)
    ax_delta.set_ylabel("Delta ETCO2 (mmHg)")

    lines = [et_line, base_line, delta_line]
    labels = [line.get_label() for line in lines]
    ax_top.legend(lines, labels, loc="upper right", frameon=False, fontsize=7)

    ax_bottom.plot(time, bold_roi, color="#d62728", lw=2.0, label="BOLD response, y(t)")
    ax_bottom.axhline(
        y_pre,
        color="0.25",
        lw=1.8,
        ls="--",
        alpha=0.75,
        label="Pre-response BOLD baseline",
    )
    ax_bottom.set_ylabel("BOLD PSC (%)")
    ax_bottom.set_xlabel("Time (s)")
    ax_bottom.grid(True, color="0.90", linewidth=0.8)
    ax_bottom.legend(loc="upper right", frameon=False, fontsize=7)

    for ax in (ax_top, ax_bottom):
        ax.axvline(t_co2, color="0.25", lw=1.2, ls=":", alpha=0.75)
        ax.axvline(t_bold_start, color="0.25", lw=1.2, ls=":", alpha=0.75)

    yrange = float(np.nanmax(bold_roi) - np.nanmin(bold_roi))
    y_delay = float(np.nanpercentile(bold_roi, 10) + 0.08 * yrange)
    ax_bottom.hlines(y_delay, t_co2, t_bold_start, color="black", linewidth=4.0)
    ax_bottom.text(
        (t_co2 + t_bold_start) / 2.0,
        y_delay - 0.28,
        "Delay, τ",
        ha="center",
        va="top",
        fontweight="bold",
    )

    arrowprops = dict(arrowstyle="<->", lw=2.8, color="black", shrinkA=0, shrinkB=0)
    y_t_bracket = max(y_start, y_late) + 0.60
    ax_bottom.vlines(
        [t_bold_start, t_bold_late],
        [y_start, y_late],
        [y_t_bracket, y_t_bracket],
        color="black",
        linewidth=1.3,
        linestyles=":",
    )
    ax_bottom.annotate(
        "",
        xy=(t_bold_late, y_t_bracket),
        xytext=(t_bold_start, y_t_bracket),
        arrowprops=dict(arrowstyle="<->", lw=2.6, color="black", shrinkA=0, shrinkB=0),
    )
    ax_bottom.scatter(
        [t_bold_start, t_bold_late],
        [y_start, y_late],
        s=18,
        color="black",
        zorder=5,
    )
    ax_bottom.text(
        (t_bold_start + t_bold_late) / 2.0,
        y_t_bracket + 0.35,
        "Response time constant, T",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

    ax_bottom.annotate(
        "",
        xy=(t_amp, y_amp),
        xytext=(t_amp, y_pre),
        arrowprops=arrowprops,
    )
    ax_bottom.text(
        t_amp + 8.0,
        (y_arrow_low + y_arrow_high) / 2.0,
        "CVR = ΔBOLD / ΔETCO2",
        ha="left",
        va="center",
        fontweight="bold",
    )

    etco2_at_amp = float(np.interp(t_amp, time, etco2))
    baseline_at_amp = float(np.interp(t_amp, time, baseline))
    ax_top.annotate(
        "",
        xy=(t_amp, etco2_at_amp),
        xytext=(t_amp, baseline_at_amp),
        arrowprops=dict(arrowstyle="<->", lw=2.2, color="black", shrinkA=0, shrinkB=0),
    )
    ax_top.text(
        t_amp + 8.0,
        (etco2_at_amp + baseline_at_amp) / 2.0,
        "ΔETCO2",
        ha="left",
        va="center",
        fontweight="bold",
    )

    ax_top.set_title("sub-01_ses-01: full ETCO2 stimulus and cortical-GM BOLD response", fontsize=11)
    display_start = 100.0
    ax_bottom.set_xlim(display_start, float(time[-1]))
    tick_step = 100.0
    tick_max = float(np.ceil(time[-1] / tick_step) * tick_step)
    ax_bottom.set_xticks(np.arange(display_start, tick_max + 0.5 * tick_step, tick_step))
    ax_bottom.margins(x=0)
    fig.text(
        0.12,
        0.015,
        (
            f"cortical-GM ROI, slice {args.slice_index}, n={int(selected.sum())}; "
            f"median CVR={roi_cvr:.3f} %/mmHg, delay={roi_delay:.1f} s, T={roi_t:.1f} s"
        ),
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.25",
    )
    fig.subplots_adjust(left=0.16, right=0.88, top=0.93, bottom=0.10, hspace=0.16)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png = args.out_dir / "sub-01_ses-01_co2_bold_parameter_concepts.png"
    pdf = args.out_dir / "sub-01_ses-01_co2_bold_parameter_concepts.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png}")
    print(f"Saved {pdf}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_out = root / "data/figures/presentation"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--etco2", type=Path, default=root / "data/processed/sub-01/ses-01/etco2_resampled.tsv")
    ap.add_argument("--bold-psc", type=Path, default=root / "data/processed/sub-01/ses-01/bold_psc.nii.gz")
    ap.add_argument("--cortical-gm-mask", type=Path, default=root / "data/derivatives/segmentation/sub-01/ses-01/bold/cortical_gm_mask.nii.gz")
    ap.add_argument("--hrf-cvr", type=Path, default=root / "data/derivatives/real_cvr/sub-01/ses-01/hrf_cvr.nii.gz")
    ap.add_argument("--hrf-delay", type=Path, default=root / "data/derivatives/real_cvr/sub-01/ses-01/hrf_delay.nii.gz")
    ap.add_argument("--hrf-t", type=Path, default=root / "data/derivatives/real_cvr/sub-01/ses-01/hrf_T.nii.gz")
    ap.add_argument("--hrf-r2", type=Path, default=root / "data/derivatives/real_cvr/sub-01/ses-01/hrf_r2.nii.gz")
    ap.add_argument("--out-dir", type=Path, default=default_out)
    ap.add_argument("--slice-index", type=int, default=25)
    ap.add_argument("--roi-voxels", type=int, default=80)
    ap.add_argument("--bold-smooth-sigma", type=float, default=2.0)
    ap.add_argument(
        "--transition-time",
        type=float,
        default=220.1,
        help="CO2 transition time to annotate; defaults to the first sustained hypercapnia rise.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    make_plot(parse_args())
