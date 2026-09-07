from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency


@dataclass(frozen=True)
class DistributionPoolConfig:
    cvr_min: float = -0.4
    cvr_max: float = 1.8
    delay_min: float = 0.0
    delay_max: float = 80.0
    T_min: float = 2.0
    T_max: float = 100.0
    r2_min: float = 0.05
    max_voxels_per_region_session: int | None = None
    include_whole_brain: bool = True
    include_boundary_region: bool = True
    random_seed: int = 17


REGION_MASK_FILES = {
    "cortical_gm": "cortical_gm_mask_exclusive_no_vessel.nii.gz",
    "subcortical_gm": "subcortical_gm_mask_exclusive_no_vessel.nii.gz",
    "wm": "wm_mask_exclusive_no_vessel.nii.gz",
    "vcsf": "vcsf_mask_exclusive_no_vessel.nii.gz",
    "vessel_like_high_confidence": "vessel_mask_high_confidence.nii.gz",
}


def pool_region_rows(parameter_maps: dict[str, Any], region_masks: dict[str, Any]) -> Any:
    pd = require_dependency("pandas", "pip install pandas")
    np = require_dependency("numpy", "pip install numpy")
    rows = []
    for region, mask in region_masks.items():
        keep = np.asarray(mask) > 0
        for idx in zip(*np.where(keep), strict=False):
            row = {"region": region}
            for name, values in parameter_maps.items():
                row[name] = float(np.asarray(values)[idx])
            rows.append(row)
    return pd.DataFrame(rows)


def pool_real_parameter_distributions(
    records: list[Any],
    real_fit_dir: str | Path,
    mask_dir: str | Path,
    segmentation_dir: str | Path,
    out_path: str | Path,
    config: DistributionPoolConfig | None = None,
) -> dict[str, Path | str | int]:
    pd = require_dependency("pandas", "pip install pandas")
    np = require_dependency("numpy", "pip install numpy")
    config = config or DistributionPoolConfig()
    real_fit_dir = Path(real_fit_dir)
    mask_dir = Path(mask_dir)
    segmentation_dir = Path(segmentation_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.random_seed)
    frames = []
    session_rows = []
    for record in records:
        subject = record.subject
        session = record.session
        try:
            frame = pool_session_parameter_rows(
                subject,
                session,
                real_fit_dir / subject / session,
                mask_dir / subject / session,
                segmentation_dir / subject / session,
                config,
                rng,
            )
            frames.append(frame)
            session_rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "status": "ok",
                    "error": "",
                    "n_rows": len(frame),
                }
            )
        except Exception as exc:
            session_rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "status": "failed",
                    "error": str(exc),
                    "n_rows": 0,
                }
            )

    if frames:
        table = pd.concat(frames, ignore_index=True)
    else:
        table = pd.DataFrame(
            columns=[
                "subject",
                "session",
                "region",
                "i",
                "j",
                "k",
                "CVR",
                "delay",
                "T",
                "tCNR",
                "R2",
                "vessel_likelihood",
                "boundary_uncertainty",
                "effective_delay",
                "glm_CVR",
                "glm_delay",
                "glm_R2",
            ]
        )

    samples_path = write_samples_table(table, out_path)
    summary_path = out_path.parent / "tissue_parameter_summary.csv"
    write_summary_table(table, summary_path)
    session_summary_path = out_path.parent / "tissue_parameter_session_summary.csv"
    pd.DataFrame(session_rows).to_csv(session_summary_path, index=False)
    qc_dir = out_path.parent / "qc_plots"
    write_distribution_qc_plots(table, qc_dir)
    return {
        "samples": samples_path,
        "summary": summary_path,
        "session_summary": session_summary_path,
        "qc_dir": qc_dir,
        "n_rows": int(len(table)),
        "n_sessions": int(len(records)),
    }


def pool_session_parameter_rows(
    subject: str,
    session: str,
    real_fit_dir: Path,
    mask_dir: Path,
    segmentation_dir: Path,
    config: DistributionPoolConfig,
    rng: Any,
) -> Any:
    pd = require_dependency("pandas", "pip install pandas")
    np = require_dependency("numpy", "pip install numpy")
    nib = require_dependency("nibabel", "pip install nibabel")

    ref_img = nib.load(str(real_fit_dir / "hrf_cvr.nii.gz"))
    maps = {
        "CVR": _load_like(real_fit_dir / "hrf_cvr.nii.gz", ref_img),
        "delay": _load_like(real_fit_dir / "hrf_delay.nii.gz", ref_img),
        "T": _load_like(real_fit_dir / "hrf_T.nii.gz", ref_img),
        "tCNR": _load_like(real_fit_dir / "tCNR.nii.gz", ref_img),
        "R2": _load_like(real_fit_dir / "hrf_r2.nii.gz", ref_img),
        "vessel_likelihood": _load_like(mask_dir / "vessel_likelihood.nii.gz", ref_img),
        "effective_delay": _load_like(real_fit_dir / "hrf_effective_delay.nii.gz", ref_img),
        "glm_CVR": _load_like(real_fit_dir / "glm_cvr.nii.gz", ref_img),
        "glm_delay": _load_like(real_fit_dir / "glm_delay.nii.gz", ref_img),
        "glm_R2": _load_like(real_fit_dir / "glm_r2.nii.gz", ref_img),
    }
    boundary_path = _first_existing(
        [
            segmentation_dir / "tissue_boundary_uncertainty.nii.gz",
            segmentation_dir / "bold" / "tissue_boundary_uncertainty.nii.gz",
        ]
    )
    if boundary_path is None:
        maps["boundary_uncertainty"] = np.zeros(ref_img.shape[:3], dtype=np.float32)
    else:
        maps["boundary_uncertainty"] = _load_like(boundary_path, ref_img)

    valid_fit = _load_like(real_fit_dir / "valid_fit_mask.nii.gz", ref_img) > 0
    region_masks = load_region_masks(mask_dir, ref_img)
    if config.include_whole_brain:
        vessel = region_masks["vessel_like_high_confidence"] > 0
        tissue_union = np.zeros(ref_img.shape[:3], dtype=bool)
        for name, mask in region_masks.items():
            if name != "vessel_like_high_confidence":
                tissue_union |= mask > 0
        region_masks["whole_brain_no_high_conf_vessel"] = tissue_union & ~vessel
    if config.include_boundary_region:
        boundary = maps["boundary_uncertainty"]
        boundary_valid = np.isfinite(boundary) & valid_fit
        if np.any(boundary_valid):
            threshold = np.nanpercentile(boundary[boundary_valid], 90.0)
            vessel = region_masks["vessel_like_high_confidence"] > 0
            region_masks["boundary_partial_volume"] = boundary_valid & (boundary >= threshold) & ~vessel

    rows = []
    for region, region_mask in region_masks.items():
        keep = np.asarray(region_mask) > 0
        keep &= valid_fit
        keep &= qc_filter(maps, config)
        indices = np.flatnonzero(keep)
        if (
            config.max_voxels_per_region_session is not None
            and indices.size > config.max_voxels_per_region_session
        ):
            indices = rng.choice(indices, size=config.max_voxels_per_region_session, replace=False)
        if indices.size == 0:
            continue
        coords = np.column_stack(np.unravel_index(indices, ref_img.shape[:3]))
        data = {
            "subject": np.repeat(subject, indices.size),
            "session": np.repeat(session, indices.size),
            "region": np.repeat(region, indices.size),
            "i": coords[:, 0].astype(np.int16),
            "j": coords[:, 1].astype(np.int16),
            "k": coords[:, 2].astype(np.int16),
        }
        for name, values in maps.items():
            data[name] = values.reshape(-1)[indices].astype(np.float32)
        rows.append(pd.DataFrame(data))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_region_masks(mask_dir: Path, reference_img: Any) -> dict[str, Any]:
    return {
        region: _load_like(mask_dir / filename, reference_img) > 0
        for region, filename in REGION_MASK_FILES.items()
    }


def qc_filter(maps: dict[str, Any], config: DistributionPoolConfig) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    cvr = np.asarray(maps["CVR"])
    delay = np.asarray(maps["delay"])
    T = np.asarray(maps["T"])
    r2 = np.asarray(maps["R2"])
    keep = np.isfinite(cvr) & np.isfinite(delay) & np.isfinite(T) & np.isfinite(r2)
    keep &= (cvr >= config.cvr_min) & (cvr <= config.cvr_max)
    keep &= (delay >= config.delay_min) & (delay <= config.delay_max)
    keep &= (T >= config.T_min) & (T <= config.T_max)
    keep &= r2 >= config.r2_min
    return keep


def write_samples_table(table: Any, out_path: Path) -> Path:
    if out_path.suffix == ".parquet":
        try:
            table.to_parquet(out_path, index=False)
            return out_path
        except Exception as exc:
            note = out_path.with_suffix(out_path.suffix + ".unavailable.txt")
            note.write_text(
                "Could not write parquet because no parquet engine is available.\n"
                f"Original error: {exc}\n"
                "Wrote compressed CSV instead.\n",
                encoding="utf-8",
            )
            csv_path = out_path.with_suffix(".csv.gz")
            table.to_csv(csv_path, index=False, compression="gzip")
            return csv_path
    table.to_csv(out_path, index=False)
    return out_path


def write_summary_table(table: Any, out_path: Path) -> Path:
    if len(table) == 0:
        table.to_csv(out_path, index=False)
        return out_path
    summary = (
        table.groupby("region", dropna=False)
        .agg(
            n_voxels=("CVR", "size"),
            cvr_mean=("CVR", "mean"),
            cvr_median=("CVR", "median"),
            cvr_std=("CVR", "std"),
            cvr_p01=("CVR", lambda s: s.quantile(0.01)),
            cvr_p99=("CVR", lambda s: s.quantile(0.99)),
            delay_mean=("delay", "mean"),
            delay_median=("delay", "median"),
            delay_std=("delay", "std"),
            delay_p01=("delay", lambda s: s.quantile(0.01)),
            delay_p99=("delay", lambda s: s.quantile(0.99)),
            T_mean=("T", "mean"),
            T_median=("T", "median"),
            T_std=("T", "std"),
            T_p01=("T", lambda s: s.quantile(0.01)),
            T_p99=("T", lambda s: s.quantile(0.99)),
            tcnr_median=("tCNR", "median"),
            r2_median=("R2", "median"),
            vessel_likelihood_median=("vessel_likelihood", "median"),
        )
        .reset_index()
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    return out_path


def write_distribution_qc_plots(table: Any, qc_dir: Path) -> list[Path]:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    np = require_dependency("numpy", "pip install numpy")
    if len(table) == 0:
        return []
    qc_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for column, limits in {
        "CVR": (-0.4, 1.8),
        "delay": (0.0, 80.0),
        "T": (2.0, 100.0),
        "tCNR": None,
        "R2": (0.0, 1.0),
    }.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for region, group in table.groupby("region"):
            values = group[column].dropna()
            if len(values) == 0:
                continue
            ax.hist(values, bins=80, density=True, histtype="step", linewidth=1.2, label=region)
        ax.set_xlabel(column)
        ax.set_ylabel("density")
        ax.set_title(f"{column} pooled distribution by region")
        if limits is not None:
            ax.set_xlim(*limits)
        ax.legend(fontsize=7)
        fig.tight_layout()
        out = qc_dir / f"{column}_distribution_by_region.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)
    outputs.extend(write_region_distribution_plots(table, qc_dir / "region_distributions"))
    rng = np.random.default_rng(17)
    for x_col, y_col, x_label, y_label, filename in [
        ("CVR", "delay", "CVR", "delay", "CVR_vs_delay_by_region.png"),
        ("CVR", "T", "CVR", "T", "CVR_vs_T_by_region.png"),
        ("delay", "T", "delay", "T", "delay_vs_T_by_region.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for region, group in table.groupby("region"):
            group = group[[x_col, y_col]].dropna()
            if len(group) == 0:
                continue
            if len(group) > 8000:
                group = group.iloc[rng.choice(len(group), size=8000, replace=False)]
            ax.scatter(group[x_col], group[y_col], s=2, alpha=0.18, label=region)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{x_label} vs {y_label} pooled by region")
        ax.legend(fontsize=7, markerscale=4)
        fig.tight_layout()
        out = qc_dir / filename
        fig.savefig(out, dpi=160)
        plt.close(fig)
        outputs.append(out)
    return outputs


def write_region_distribution_plots(table: Any, out_dir: Path) -> list[Path]:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    np = require_dependency("numpy", "pip install numpy")
    if len(table) == 0:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    specs = [
        ("CVR", "CVR magnitude (%BOLD/mmHg)", (-0.4, 1.8), "cvr_magnitude", "kde", None),
        ("delay", "Delay tau (s)", (0.0, 80.0), "delay", "grid_smooth", None),
        ("T", "Vascular time constant T (s)", (2.0, 100.0), "T", "bounded_kde", 0.035),
        ("tCNR", "tCNR", _robust_plot_limits(table["tCNR"], floor=0.0), "tcnr", "kde", None),
        ("R2", "R2", (0.0, 1.0), "r2", "bounded_kde", 0.035),
        (
            "effective_delay",
            "Effective delay (s)",
            _robust_plot_limits(table["effective_delay"], floor=0.0),
            "effective_delay",
            "kde",
            0.035,
        ),
    ]
    for column, label, limits, stem, density_mode, bw_method in specs:
        if column not in table.columns:
            continue
        outputs.append(
            _save_all_region_kde_plot(
                table,
                column,
                label,
                limits,
                out_dir / f"{stem}_all_regions.png",
                density_mode=density_mode,
                bw_method=bw_method,
            )
        )
        for region, group in table.groupby("region"):
            values = group[column].dropna().to_numpy(dtype=float)
            safe_region = _safe_filename(str(region))
            outputs.append(
                _save_single_region_distribution_plot(
                    values,
                    label,
                    limits,
                    str(region),
                    out_dir / f"{stem}_{safe_region}.png",
                    density_mode=density_mode,
                    bw_method=bw_method,
                )
            )
    return outputs


def _save_all_region_kde_plot(
    table: Any,
    column: str,
    label: str,
    limits: tuple[float, float],
    out: Path,
    density_mode: str = "kde",
    bw_method: float | None = None,
) -> Path:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    np = require_dependency("numpy", "pip install numpy")
    fig, ax = plt.subplots(figsize=(9, 5))
    x_grid = np.linspace(limits[0], limits[1], 512)
    for region, group in table.groupby("region"):
        values = group[column].dropna().to_numpy(dtype=float)
        if density_mode == "grid":
            centers, density = _histogram_density_curve(values, limits, bins=80)
            ax.step(centers, density, where="mid", linewidth=1.5, label=region)
        elif density_mode == "grid_smooth":
            x_smooth, density = _grid_smoothed_density(values, limits)
            ax.plot(x_smooth, density, linewidth=1.5, label=region)
        else:
            density = _kde_density(
                values,
                x_grid,
                bounds=limits if density_mode == "bounded_kde" else None,
                bw_method=bw_method,
            )
            if density is None:
                ax.hist(
                    values,
                    bins=80,
                    range=limits,
                    density=True,
                    histtype="step",
                    linewidth=1.0,
                    label=region,
                )
            else:
                ax.plot(x_grid, density, linewidth=1.5, label=region)
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.set_xlim(*limits)
    ax.set_title(f"{label} distribution, all regions")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _save_single_region_distribution_plot(
    values: Any,
    label: str,
    limits: tuple[float, float],
    region: str,
    out: Path,
    density_mode: str = "kde",
    bw_method: float | None = None,
) -> Path:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    np = require_dependency("numpy", "pip install numpy")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(8, 5))
    if values.size:
        ax.hist(
            values,
            bins=80,
            range=limits,
            density=True,
            alpha=0.38 if density_mode == "grid" else 0.22,
            color="#5B84B1",
            label="histogram",
        )
        if density_mode == "grid":
            centers, density = _histogram_density_curve(values, limits, bins=80)
            ax.step(centers, density, where="mid", color="#C33C54", linewidth=2.0, label="grid density")
            ax.legend(fontsize=8)
        elif density_mode == "grid_smooth":
            x_smooth, density = _grid_smoothed_density(values, limits)
            ax.plot(x_smooth, density, color="#C33C54", linewidth=2.0, label="grid-smoothed density")
            ax.legend(fontsize=8)
        else:
            x_grid = np.linspace(limits[0], limits[1], 512)
            density = _kde_density(
                values,
                x_grid,
                bounds=limits if density_mode == "bounded_kde" else None,
                bw_method=bw_method,
            )
            if density is not None:
                label_text = "boundary-corrected KDE" if density_mode == "bounded_kde" else "KDE"
                ax.plot(x_grid, density, color="#C33C54", linewidth=2.0, label=label_text)
                ax.legend(fontsize=8)
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.set_xlim(*limits)
    ax.set_title(f"{region}: {label} distribution")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _kde_density(
    values: Any,
    x_grid: Any,
    max_points: int = 50000,
    bounds: tuple[float, float] | None = None,
    bw_method: float | None = None,
) -> Any | None:
    np = require_dependency("numpy", "pip install numpy")
    stats = require_dependency("scipy.stats", "pip install scipy")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or np.nanstd(values) <= 1e-8:
        return None
    if values.size > max_points:
        rng = np.random.default_rng(17)
        values = values[rng.choice(values.size, size=max_points, replace=False)]
    try:
        if bounds is not None:
            lo, hi = bounds
            bounded_values = values[(values >= lo) & (values <= hi)]
            if bounded_values.size < 3 or np.nanstd(bounded_values) <= 1e-8:
                return None
            reflected = np.concatenate(
                [
                    bounded_values,
                    2.0 * lo - bounded_values,
                    2.0 * hi - bounded_values,
                ]
            )
            kde = stats.gaussian_kde(reflected, bw_method=bw_method)
            return 3.0 * kde(np.asarray(x_grid, dtype=float))
        kde = stats.gaussian_kde(values, bw_method=bw_method)
        return kde(np.asarray(x_grid, dtype=float))
    except Exception:
        return None


def _histogram_density_curve(values: Any, limits: tuple[float, float], bins: int = 80) -> tuple[Any, Any]:
    np = require_dependency("numpy", "pip install numpy")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins, range=limits, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def _grid_smoothed_density(
    values: Any,
    limits: tuple[float, float],
    bins: int = 640,
    sigma_bins: float = 0.35,
) -> tuple[Any, Any]:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins, range=limits, density=False)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if values.size == 0 or not np.any(hist):
        return centers, np.zeros_like(centers)
    padded = np.pad(hist.astype(float), 8, mode="edge")
    smooth = ndi.gaussian_filter1d(padded, sigma=sigma_bins, mode="nearest")[8:-8]
    area = np.sum(smooth * widths)
    if area > 0:
        smooth = smooth / area
    return centers, smooth


def _robust_plot_limits(values: Any, floor: float | None = None) -> tuple[float, float]:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0 if floor is None else floor, 1.0)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if floor is not None:
        lo = min(max(float(lo), float(floor)), float(np.nanmedian(arr)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if floor is not None:
        lo = max(lo, floor)
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.03 * (hi - lo)
    return float(lo), float(hi + pad)


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.lower())
    return safe.strip("_") or "region"


def _load_like(path: str | Path, reference_img: Any) -> Any:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    img = nib.load(str(path))
    if img.shape[:3] != reference_img.shape[:3] or not np.allclose(img.affine, reference_img.affine):
        raise ValueError(f"Image grid mismatch for {path}")
    return img.get_fdata(dtype=np.float32)


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
