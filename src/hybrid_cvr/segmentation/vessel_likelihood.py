from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency


@dataclass(frozen=True)
class VesselLikelihoodConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "abs_glm_cvr": 1.0,
            "abs_hrf_cvr": 1.0,
            "abs_tcnr": 1.0,
            "temporal_std": 0.75,
            "low_temporal_snr": 0.50,
            "etco2_corr": 0.75,
            "fit_quality": 0.50,
            "bold_darkness": 0.75,
            "mean_bold_frangi": 0.75,
            "temporal_std_frangi": 0.50,
            "fast_response": 0.25,
        }
    )
    high_percentile: float = 98.5
    medium_percentile: float = 96.0
    min_component_size: int = 5
    closing_iterations: int = 1
    vcsf_buffer_iterations: int = 1
    tissue_boundary_buffer_iterations: int = 0
    frangi_sigmas: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    repeatability_weight: float = 0.35
    use_repeatability: bool = True
    repeatability_resample_if_needed: bool = True
    generate_qc_plots: bool = True


def compute_vessel_likelihood(
    feature_maps: dict[str, Any],
    brain_mask: Any,
    csf_mask: Any | None = None,
    config: VesselLikelihoodConfig | None = None,
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    config = config or VesselLikelihoodConfig()
    mask = np.asarray(brain_mask) > 0
    if csf_mask is not None:
        mask &= np.asarray(csf_mask) <= 0

    score = np.zeros_like(np.asarray(brain_mask, dtype=float), dtype=float)
    weight_sum = 0.0
    feature_scores: dict[str, Any] = {}
    for name, weight in config.weights.items():
        if name not in feature_maps or float(weight) == 0.0:
            continue
        signed = name not in {"abs_glm_cvr", "abs_hrf_cvr", "abs_tcnr"}
        values = np.asarray(feature_maps[name], dtype=float)
        if not signed:
            values = np.abs(values)
        feature_score = _robust_upper_tail_score(values, mask)
        feature_scores[f"{name}_score"] = feature_score.astype("float32")
        score += float(weight) * feature_score
        weight_sum += abs(float(weight))

    if weight_sum > 0:
        score /= weight_sum
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    score *= mask
    score = _normalise_inside(score, mask)

    high, medium = threshold_likelihood_masks(score, mask, config)
    return {
        "vessel_likelihood": score.astype("float32"),
        "vessel_mask_high_confidence": high,
        "vessel_mask_medium_confidence": medium,
        "vessel_like_high_confidence_mask": high,
        "vessel_like_medium_confidence_mask": medium,
        **feature_scores,
    }


def segment_vessels_session(
    processed_dir: str | Path,
    real_cvr_dir: str | Path,
    segmentation_dir: str | Path,
    out_dir: str | Path,
    config: VesselLikelihoodConfig | None = None,
) -> dict[str, Path | str | int | float]:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    config = config or VesselLikelihoodConfig()
    processed_dir = Path(processed_dir)
    real_cvr_dir = Path(real_cvr_dir)
    segmentation_dir = Path(segmentation_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_img = nib.load(str(processed_dir / "mean_bold.nii.gz"))
    mean_bold = mean_img.get_fdata(dtype=np.float32)
    brain = _load_like(processed_dir / "valid_signal_mask.nii.gz", mean_img) > 0
    vcsf = _first_existing(
        [
            segmentation_dir / "bold" / "vcsf_mask.nii.gz",
            segmentation_dir / "vcsf_mask.nii.gz",
        ]
    )
    if vcsf is None:
        vcsf_mask = np.zeros(brain.shape, dtype=bool)
    else:
        vcsf_mask = _load_like(vcsf, mean_img) > 0
    exclusion = _buffer_mask(vcsf_mask, config.vcsf_buffer_iterations)

    boundary = _first_existing(
        [
            segmentation_dir / "bold" / "tissue_boundary_uncertainty.nii.gz",
            segmentation_dir / "tissue_boundary_uncertainty.nii.gz",
        ]
    )
    if boundary is not None and config.tissue_boundary_buffer_iterations > 0:
        boundary_data = _load_like(boundary, mean_img)
        boundary_mask = boundary_data > np.nanpercentile(boundary_data[brain], 90)
        exclusion |= _buffer_mask(boundary_mask, config.tissue_boundary_buffer_iterations)

    candidate_mask = brain & ~exclusion
    feature_maps = {
        "abs_glm_cvr": _load_like(real_cvr_dir / "glm_cvr.nii.gz", mean_img),
        "abs_hrf_cvr": _load_like(real_cvr_dir / "hrf_cvr.nii.gz", mean_img),
        "abs_tcnr": _load_like(real_cvr_dir / "tCNR.nii.gz", mean_img),
        "temporal_std": _load_like(processed_dir / "temporal_std.nii.gz", mean_img),
        "low_temporal_snr": -_load_like(processed_dir / "temporal_snr.nii.gz", mean_img),
        "etco2_corr": _load_like(real_cvr_dir / "etco2_correlation_peak.nii.gz", mean_img),
        "fit_quality": _load_like(real_cvr_dir / "fit_quality.nii.gz", mean_img),
        "bold_darkness": -_robust_intensity(mean_bold, brain),
        "mean_bold_frangi": _frangi_vesselness(mean_bold, brain, config.frangi_sigmas, black_ridges=True),
        "temporal_std_frangi": _frangi_vesselness(
            _load_like(processed_dir / "temporal_std.nii.gz", mean_img),
            brain,
            config.frangi_sigmas,
            black_ridges=False,
        ),
        "fast_response": -_load_like(real_cvr_dir / "hrf_effective_delay.nii.gz", mean_img),
    }
    outputs = compute_vessel_likelihood(feature_maps, candidate_mask, None, config)

    raw_high = outputs["vessel_mask_high_confidence"]
    raw_medium = outputs["vessel_mask_medium_confidence"]
    high = clean_binary_mask(raw_high, candidate_mask, config)
    medium = clean_binary_mask(raw_medium, candidate_mask, config)
    likelihood = np.asarray(outputs["vessel_likelihood"], dtype=np.float32)

    written: dict[str, Path | str | int | float] = {
        "vessel_likelihood": _save_map(likelihood, mean_img, out_dir / "vessel_likelihood.nii.gz"),
        "vessel_mask_high_confidence": _save_mask(
            high, mean_img, out_dir / "vessel_mask_high_confidence.nii.gz"
        ),
        "vessel_mask_medium_confidence": _save_mask(
            medium, mean_img, out_dir / "vessel_mask_medium_confidence.nii.gz"
        ),
        "vcsf_exclusion_mask": _save_mask(
            exclusion, mean_img, out_dir / "vcsf_exclusion_mask.nii.gz"
        ),
        "candidate_mask": _save_mask(candidate_mask, mean_img, out_dir / "candidate_mask.nii.gz"),
        "n_candidate_voxels": int(np.sum(candidate_mask)),
        "n_high_voxels": int(np.sum(high)),
        "n_medium_voxels": int(np.sum(medium)),
        "high_percentile": float(config.high_percentile),
        "medium_percentile": float(config.medium_percentile),
        "repeatability_boosted": "false",
    }
    if config.generate_qc_plots:
        written["vessel_qc"] = save_vessel_qc_plot(
            mean_bold,
            likelihood,
            high,
            medium,
            out_dir / "vessel_qc.png",
        )
    write_vessel_metadata(out_dir / "vessel_metadata.csv", written)
    return written


def apply_repeatability_boost(
    session_outputs: dict[tuple[str, str], dict[str, Any]],
    config: VesselLikelihoodConfig | None = None,
) -> None:
    nib = require_dependency("nibabel", "pip install nibabel")
    processing = require_dependency("nibabel.processing", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    config = config or VesselLikelihoodConfig()
    if not config.use_repeatability or config.repeatability_weight <= 0:
        return

    by_subject: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for (subject, session), outputs in session_outputs.items():
        by_subject.setdefault(subject, []).append((session, outputs))

    for sessions in by_subject.values():
        if len(sessions) < 2:
            continue
        for session, outputs in sessions:
            path = Path(outputs["vessel_likelihood"])
            img = nib.load(str(path))
            likelihood = img.get_fdata(dtype=np.float32)
            candidate = nib.load(str(outputs["candidate_mask"])).get_fdata() > 0
            peer_maps = []
            for other_session, other_outputs in sessions:
                if other_session == session:
                    continue
                other_img = nib.load(str(other_outputs["vessel_likelihood"]))
                if other_img.shape == img.shape and np.allclose(other_img.affine, img.affine):
                    peer_maps.append(other_img.get_fdata(dtype=np.float32))
                elif config.repeatability_resample_if_needed:
                    peer = processing.resample_from_to(other_img, img, order=1)
                    peer_maps.append(peer.get_fdata(dtype=np.float32))
            if not peer_maps:
                continue
            peer = np.mean(np.stack(peer_maps, axis=0), axis=0)
            boosted = _normalise_inside(
                (1.0 - config.repeatability_weight) * likelihood
                + config.repeatability_weight * peer,
                candidate,
            )
            high, medium = threshold_likelihood_masks(boosted, candidate, config)
            high = clean_binary_mask(high, candidate, config)
            medium = clean_binary_mask(medium, candidate, config)
            _save_map(boosted.astype(np.float32), img, path)
            _save_mask(high, img, Path(outputs["vessel_mask_high_confidence"]))
            _save_mask(medium, img, Path(outputs["vessel_mask_medium_confidence"]))
            outputs["repeatability_boosted"] = "true"
            outputs["n_high_voxels"] = int(np.sum(high))
            outputs["n_medium_voxels"] = int(np.sum(medium))
            write_vessel_metadata(Path(path).parent / "vessel_metadata.csv", outputs)


def threshold_likelihood_masks(
    score: Any, candidate_mask: Any, config: VesselLikelihoodConfig
) -> tuple[Any, Any]:
    np = require_dependency("numpy", "pip install numpy")
    score = np.asarray(score, dtype=float)
    candidate = np.asarray(candidate_mask) > 0
    valid = candidate & np.isfinite(score)
    values = score[valid]
    if values.size == 0:
        empty = np.zeros(score.shape, dtype="uint8")
        return empty, empty
    positive = values[values > 0]
    threshold_values = positive if positive.size else values
    high_thr = float(np.percentile(threshold_values, config.high_percentile))
    med_thr = float(np.percentile(threshold_values, config.medium_percentile))
    high = (score >= high_thr) & valid
    medium = (score >= med_thr) & valid
    return high.astype("uint8"), medium.astype("uint8")


def clean_binary_mask(mask: Any, candidate_mask: Any, config: VesselLikelihoodConfig) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    cleaned = (np.asarray(mask) > 0) & (np.asarray(candidate_mask) > 0)
    structure = ndi.generate_binary_structure(cleaned.ndim, 1)
    if config.closing_iterations > 0 and np.any(cleaned):
        cleaned = ndi.binary_closing(cleaned, structure=structure, iterations=config.closing_iterations)
        cleaned &= np.asarray(candidate_mask) > 0
    if config.min_component_size > 1 and np.any(cleaned):
        labels, n_labels = ndi.label(cleaned, structure=structure)
        counts = np.bincount(labels.ravel())
        keep = np.zeros(n_labels + 1, dtype=bool)
        keep[np.flatnonzero(counts >= int(config.min_component_size))] = True
        keep[0] = False
        cleaned = keep[labels]
    return cleaned.astype("uint8")


def save_vessel_qc_plot(
    mean_bold: Any,
    likelihood: Any,
    high_mask: Any,
    medium_mask: Any,
    out: str | Path,
) -> Path:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    np = require_dependency("numpy", "pip install numpy")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mean_bold = np.asarray(mean_bold)
    likelihood = np.asarray(likelihood)
    high = np.asarray(high_mask) > 0
    medium = np.asarray(medium_mask) > 0
    z = _representative_slice(high | medium | (likelihood > 0), axis=2)
    y = _representative_slice(high | medium | (likelihood > 0), axis=1)
    x = _representative_slice(high | medium | (likelihood > 0), axis=0)
    slices = [
        ("axial", mean_bold[..., z], likelihood[..., z], high[..., z], medium[..., z]),
        ("coronal", mean_bold[:, y, :], likelihood[:, y, :], high[:, y, :], medium[:, y, :]),
        ("sagittal", mean_bold[x, :, :], likelihood[x, :, :], high[x, :, :], medium[x, :, :]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row, (name, base, score, hi, med) in enumerate(slices):
        for col, data in enumerate([base, score, base]):
            ax = axes[row, col]
            ax.axis("off")
            if col == 0:
                ax.imshow(np.rot90(data), cmap="gray")
                ax.set_title(f"{name} mean BOLD")
            elif col == 1:
                ax.imshow(np.rot90(score), cmap="magma", vmin=0, vmax=1)
                ax.set_title(f"{name} likelihood")
            else:
                ax.imshow(np.rot90(data), cmap="gray")
                med_overlay = np.ma.masked_where(~np.rot90(med), np.rot90(med))
                hi_overlay = np.ma.masked_where(~np.rot90(hi), np.rot90(hi))
                ax.imshow(med_overlay, cmap="autumn", alpha=0.35, vmin=0, vmax=1)
                ax.imshow(hi_overlay, cmap="cool", alpha=0.65, vmin=0, vmax=1)
                ax.set_title(f"{name} masks")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def write_vessel_metadata(path: Path, outputs: dict[str, Path | str | int | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in outputs.items():
            writer.writerow([key, value])


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


def _buffer_mask(mask: Any, iterations: int) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    if iterations <= 0:
        return np.asarray(mask) > 0
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    structure = ndi.generate_binary_structure(np.asarray(mask).ndim, 1)
    return ndi.binary_dilation(np.asarray(mask) > 0, structure=structure, iterations=iterations)


def _robust_upper_tail_score(data: Any, mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(data, dtype=float)
    valid = (np.asarray(mask) > 0) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.float32)
    values = arr[valid]
    if values.size == 0:
        return out
    lo, hi = np.percentile(values, [50.0, 99.5])
    if hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        return out
    out[valid] = np.clip((arr[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _normalise_inside(data: Any, mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.asarray(mask) > 0
    out = np.zeros(arr.shape, dtype=np.float32)
    if not np.any(valid):
        return out
    values = arr[valid]
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi <= lo:
        return out
    out[valid] = ((arr[valid] - lo) / (hi - lo)).astype(np.float32)
    return out


def _robust_intensity(data: Any, mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(data, dtype=float)
    valid = (np.asarray(mask) > 0) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.float32)
    values = arr[valid]
    if values.size == 0:
        return out
    lo, hi = np.percentile(values, [2.0, 98.0])
    if hi <= lo:
        return out
    out[valid] = np.clip((arr[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _frangi_vesselness(data: Any, mask: Any, sigmas: tuple[float, ...], black_ridges: bool) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = _robust_intensity(data, mask)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        filters = require_dependency("skimage.filters", "pip install scikit-image")
        vesselness = filters.frangi(
            arr,
            sigmas=sigmas,
            alpha=0.5,
            beta=0.5,
            gamma=None,
            black_ridges=black_ridges,
        )
    except Exception:
        vesselness = np.zeros_like(arr, dtype=np.float32)
    return _normalise_inside(vesselness, mask)


def _save_map(data: Any, reference_img: Any, out_path: str | Path) -> Path:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference_img.affine, reference_img.header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(out_path))
    return out_path


def _save_mask(data: Any, reference_img: Any, out_path: str | Path) -> Path:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image((np.asarray(data) > 0).astype(np.uint8), reference_img.affine, reference_img.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))
    return out_path


def _representative_slice(mask: Any, axis: int) -> int:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(mask) > 0
    if arr.ndim != 3:
        return 0
    counts = np.sum(arr, axis=tuple(i for i in range(3) if i != axis))
    if np.any(counts):
        return int(np.argmax(counts))
    return arr.shape[axis] // 2
