from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency


MIDA_REGION_LABELS = {
    "cortical_gm": 1,
    "subcortical_gm": 2,
    "wm": 3,
    "vcsf": 4,
    "vessel_like": 5,
}

MIDA_DISTRIBUTION_REGIONS = {
    "cortical_gm": "cortical_gm",
    "subcortical_gm": "subcortical_gm",
    "wm": "wm",
    "vcsf": "vcsf",
    "vessel_like": "vessel_like_high_confidence",
}


@dataclass(frozen=True)
class MidaSegmentationConfig:
    label_map_path: Path = Path("data/raw/MIDA_v1.nii")
    output_dir: Path = Path("data/derivatives/mida")
    target_shape: tuple[int, int, int] = (94, 94, 50)
    downsample_factor: tuple[int, int, int] = (5, 5, 5)
    target_voxel_size_mm: tuple[float, float, float] = (2.5, 2.5, 2.5)
    wm_labels: tuple[int, ...] = (9, 12)
    cortical_gm_labels: tuple[int, ...] = (2, 10)
    subcortical_gm_labels: tuple[int, ...] = (4, 5, 7, 8, 16, 17, 20, 21, 99, 116)
    vcsf_labels: tuple[int, ...] = (6,)
    vessel_like_labels: tuple[int, ...] = (24, 25)
    vessel_brain_margin_mm: float = 3.0
    fill_sigma_mm: float = 1.0
    brain_support_fraction_threshold: float = 0.05
    save_fsleyes_canonical: bool = True


@dataclass(frozen=True)
class MidaParameterConfig:
    distribution_path: Path = Path("data/distributions/tissue_parameter_samples.parquet")
    mida_dir: Path = Path("data/derivatives/mida/2p5mm")
    highres_dir: Path = Path("data/derivatives/mida/highres")
    output_dir: Path = Path("data/simulated/mida_parameters")
    n_cases: int = 1
    seed: int = 17
    component_fraction_min: float = 0.005
    parameter_sampling_mode: str = "subvoxel_monte_carlo"
    sample_quantile_min: float = 0.0
    sample_quantile_max: float = 1.0
    subvoxel_samples_per_lowres_voxel: int = 125
    parameter_patch_size_vox: tuple[int, int, int] = (1, 1, 1)
    delay_sampling_mode: str = "same_as_parameter"
    delay_sample_quantile_min: float = 0.0
    delay_sample_quantile_max: float = 1.0
    T_sampling_mode: str = "same_as_parameter"
    T_sample_quantile_min: float = 0.0
    T_sample_quantile_max: float = 1.0
    parameter_post_smooth_sigma_vox: float = 0.0
    parameter_post_smooth_blend: float = 0.0
    spatial_smoothing_sigma_vox: float = 5.0
    rank_jitter: float = 0.0
    within_tissue_variation_scale: float = 1.0
    save_highres_case_indices: tuple[int, ...] = field(default_factory=tuple)
    save_fsleyes_canonical: bool = True


def mida_missing_message() -> str:
    return "MIDA label map is unavailable; using phantom fallback only for tests/debug simulation."


def segment_mida(config: MidaSegmentationConfig | None = None) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    nib = require_dependency("nibabel", "pip install nibabel")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")

    config = config or MidaSegmentationConfig()
    label_path = Path(config.label_map_path).expanduser()
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    img = nib.load(str(label_path))
    labels = np.rint(img.get_fdata(dtype=np.float32)).astype(np.int32)
    voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[:3])

    raw_masks = {
        "cortical_gm": np.isin(labels, config.cortical_gm_labels),
        "subcortical_gm": np.isin(labels, config.subcortical_gm_labels),
        "wm": np.isin(labels, config.wm_labels),
        "vcsf": np.isin(labels, config.vcsf_labels),
        "vessel_like": np.isin(labels, config.vessel_like_labels),
    }
    raw_counts = {name: int(mask.sum()) for name, mask in raw_masks.items()}
    brain_base = raw_masks["cortical_gm"] | raw_masks["subcortical_gm"] | raw_masks["wm"] | raw_masks["vcsf"]
    if not np.any(brain_base):
        raise ValueError("MIDA tissue labels produced an empty brain mask")

    brain_support = largest_component(ndi.binary_fill_holes(brain_base))
    margin_vox = max(0, int(round(config.vessel_brain_margin_mm / min(voxel_sizes))))
    vessel_neighbourhood = ndi.binary_dilation(brain_support, iterations=margin_vox) if margin_vox else brain_support
    raw_masks["vessel_like"] &= vessel_neighbourhood

    target_highres_shape = tuple(
        int(shape * factor) for shape, factor in zip(config.target_shape, config.downsample_factor, strict=True)
    )
    crop_slices = centered_crop_slices(brain_base, target_highres_shape, labels.shape)
    crop_start = tuple(int(s.start or 0) for s in crop_slices)
    crop_stop = tuple(int(s.stop or dim) for s, dim in zip(crop_slices, labels.shape, strict=True))

    cropped_masks = {name: mask[crop_slices] for name, mask in raw_masks.items()}
    cropped_brain = brain_support[crop_slices]
    filled_highres = softly_fill_internal_holes(
        cropped_masks,
        cropped_brain,
        sigma_vox=tuple(config.fill_sigma_mm / v for v in voxel_sizes),
    )

    crop_affine = translated_affine(img.affine, crop_start)
    lowres_affine = downsampled_affine(crop_affine, config.downsample_factor)
    highres_dir = config.output_dir / "highres"
    lowres_dir = config.output_dir / "2p5mm"
    highres_dir.mkdir(parents=True, exist_ok=True)
    lowres_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Any] = {
        "highres_dir": highres_dir,
        "lowres_dir": lowres_dir,
        "crop_start": crop_start,
        "crop_stop": crop_stop,
        "voxel_sizes": voxel_sizes,
    }
    highres_ref = nib.Nifti1Image(np.zeros(target_highres_shape, dtype=np.uint8), crop_affine)
    for name, mask in cropped_masks.items():
        out = highres_dir / f"mida_mask_{name}.nii.gz"
        save_nifti(mask.astype(np.uint8), highres_ref, out, dtype=np.uint8)
        outputs[f"highres_mask_{name}"] = out
    brain_out = highres_dir / "mida_brain_support.nii.gz"
    save_nifti(cropped_brain.astype(np.uint8), highres_ref, brain_out, dtype=np.uint8)
    outputs["highres_brain_support"] = brain_out

    fraction_maps = {
        name: block_average(prob.astype(np.float32), config.downsample_factor)
        for name, prob in filled_highres.items()
    }
    brain_fraction = block_average(cropped_brain.astype(np.float32), config.downsample_factor)
    lowres_brain_mask = fill_lowres_brain_support(brain_fraction > 0)
    fraction_maps = normalize_fraction_maps(
        fraction_maps,
        support_mask=lowres_brain_mask,
    )
    fraction_sum = sum(fraction_maps.values())
    dominant = dominant_region_label(fraction_maps, fraction_sum > 0)

    lowres_ref = nib.Nifti1Image(np.zeros(config.target_shape, dtype=np.float32), lowres_affine)
    for name, fraction in fraction_maps.items():
        out = lowres_dir / f"mida_fraction_{name}.nii.gz"
        save_nifti(fraction.astype(np.float32), lowres_ref, out, dtype=np.float32)
        outputs[f"fraction_{name}"] = out
    fraction_sum_path = lowres_dir / "mida_fraction_sum.nii.gz"
    brain_fraction_path = lowres_dir / "mida_brain_support_fraction.nii.gz"
    brain_mask_path = lowres_dir / "mida_brain_mask.nii.gz"
    dominant_path = lowres_dir / "mida_dominant_region_label.nii.gz"
    save_nifti(fraction_sum.astype(np.float32), lowres_ref, fraction_sum_path, dtype=np.float32)
    save_nifti(brain_fraction.astype(np.float32), lowres_ref, brain_fraction_path, dtype=np.float32)
    save_nifti(lowres_brain_mask.astype(np.uint8), lowres_ref, brain_mask_path, dtype=np.uint8)
    save_nifti(dominant.astype(np.uint8), lowres_ref, dominant_path, dtype=np.uint8)
    outputs["fraction_sum"] = fraction_sum_path
    outputs["brain_support_fraction"] = brain_fraction_path
    outputs["brain_mask"] = brain_mask_path
    outputs["dominant_region_label"] = dominant_path
    outputs["fraction_qc"] = save_fraction_qc_plot(fraction_maps, dominant, lowres_dir / "mida_fraction_qc.png")
    if config.save_fsleyes_canonical:
        outputs["fsleyes_canonical_dir"] = save_canonical_nifti_copies(
            [
                *[outputs[f"fraction_{name}"] for name in MIDA_REGION_LABELS],
                fraction_sum_path,
                brain_fraction_path,
                brain_mask_path,
                dominant_path,
            ],
            lowres_dir / "fsleyes_RAS",
        )

    summary_path = write_mida_segmentation_summary(
        config.output_dir / "mida_segmentation_summary.csv",
        raw_masks,
        cropped_masks,
        filled_highres,
        fraction_maps,
        raw_counts,
        crop_start,
        crop_stop,
        labels.shape,
        config,
    )
    outputs["summary"] = summary_path
    return outputs


def centered_crop_slices(
    support_mask: Any,
    target_shape: tuple[int, int, int],
    full_shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    np = require_dependency("numpy", "pip install numpy")
    coords = np.column_stack(np.where(support_mask))
    if coords.size == 0:
        raise ValueError("Cannot crop around an empty support mask")
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = np.rint((mins + maxs) / 2.0).astype(int)
    starts = []
    for dim, target, c in zip(full_shape, target_shape, center, strict=True):
        if target > dim:
            raise ValueError(f"Target crop size {target} exceeds source dimension {dim}")
        start = int(c - target // 2)
        start = max(0, min(start, dim - target))
        starts.append(start)
    return tuple(slice(start, start + target) for start, target in zip(starts, target_shape, strict=True))


def largest_component(mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    labels, n_labels = ndi.label(mask)
    if n_labels <= 1:
        return np.asarray(mask, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def fill_lowres_brain_support(support_mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    support = np.asarray(support_mask, dtype=bool)
    filled = ndi.binary_fill_holes(support)
    for axis in range(3):
        moved = np.moveaxis(support, axis, 0)
        moved_filled = np.zeros_like(moved, dtype=bool)
        for idx in range(moved.shape[0]):
            moved_filled[idx] = ndi.binary_fill_holes(moved[idx])
        filled |= np.moveaxis(moved_filled, 0, axis)
    return largest_component(filled)


def softly_fill_internal_holes(
    masks: dict[str, Any],
    brain_support: Any,
    sigma_vox: tuple[float, float, float],
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")

    ordered = ["cortical_gm", "subcortical_gm", "wm", "vcsf", "vessel_like"]
    filled = {name: np.asarray(masks[name], dtype=np.float32).copy() for name in ordered}
    total = sum(filled.values())
    missing = np.asarray(brain_support, dtype=bool) & (total <= 0)
    if not np.any(missing):
        return filled

    tissue_names = ["cortical_gm", "subcortical_gm", "wm", "vcsf"]
    smoothed = {
        name: ndi.gaussian_filter(np.asarray(masks[name], dtype=np.float32), sigma=sigma_vox)
        for name in tissue_names
    }
    smooth_total = sum(smoothed.values())
    fallback_label = nearest_tissue_labels({name: masks[name] for name in tissue_names})
    for label_value, name in enumerate(tissue_names, start=1):
        probs = np.divide(
            smoothed[name],
            smooth_total,
            out=np.zeros_like(smoothed[name], dtype=np.float32),
            where=smooth_total > 0,
        )
        fallback = (fallback_label == label_value).astype(np.float32)
        probs = np.where(smooth_total > 0, probs, fallback)
        filled[name][missing] = probs[missing]
    filled["vessel_like"][missing] = 0.0
    return filled


def nearest_tissue_labels(masks: dict[str, Any]) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    stacked = np.stack([np.asarray(mask, dtype=bool) for mask in masks.values()], axis=0)
    tissue_union = np.any(stacked, axis=0)
    if not np.any(tissue_union):
        return np.ones(next(iter(masks.values())).shape, dtype=np.uint8)
    _, nearest = ndi.distance_transform_edt(~tissue_union, return_indices=True)
    hard = (np.argmax(stacked, axis=0) + 1).astype(np.uint8)
    nearest_hard = hard[tuple(nearest)]
    return nearest_hard


def block_average(data: Any, factor: tuple[int, int, int]) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if any(dim % f != 0 for dim, f in zip(arr.shape, factor, strict=True)):
        raise ValueError(f"Array shape {arr.shape} is not divisible by block factor {factor}")
    nx, ny, nz = (dim // f for dim, f in zip(arr.shape, factor, strict=True))
    fx, fy, fz = factor
    return arr.reshape(nx, fx, ny, fy, nz, fz).mean(axis=(1, 3, 5), dtype=np.float32)


def normalize_fraction_maps(
    fraction_maps: dict[str, Any],
    min_support: float = 0.05,
    support_mask: Any | None = None,
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    ordered = ["cortical_gm", "subcortical_gm", "wm", "vcsf", "vessel_like"]
    fractions = {name: np.clip(np.asarray(fraction_maps[name], dtype=np.float32), 0.0, 1.0) for name in ordered}
    total = sum(fractions.values())
    support = np.asarray(support_mask, dtype=bool) if support_mask is not None else total >= float(min_support)
    missing = support & (total <= 0)
    if np.any(missing):
        fractions = nearest_fill_fraction_vectors(fractions, missing)
        total = sum(fractions.values())
    normalized = {}
    for name in ordered:
        normalized[name] = np.divide(
            fractions[name],
            total,
            out=np.zeros_like(fractions[name], dtype=np.float32),
            where=support & (total > 0),
        )
    return normalized


def nearest_fill_fraction_vectors(fractions: dict[str, Any], missing_mask: Any) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    filled = {name: np.asarray(value, dtype=np.float32).copy() for name, value in fractions.items()}
    total = sum(filled.values())
    source = total > 0
    missing = np.asarray(missing_mask, dtype=bool) & ~source
    if not np.any(missing) or not np.any(source):
        return filled
    _, nearest = ndi.distance_transform_edt(~source, return_indices=True)
    for name in filled:
        values = filled[name]
        values[missing] = values[tuple(nearest)][missing]
    return filled


def dominant_region_label(fraction_maps: dict[str, Any], support_mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ordered = list(MIDA_REGION_LABELS)
    stack = np.stack([np.asarray(fraction_maps[name], dtype=np.float32) for name in ordered], axis=0)
    dominant_idx = np.argmax(stack, axis=0)
    labels = np.zeros(stack.shape[1:], dtype=np.uint8)
    region_values = np.array([MIDA_REGION_LABELS[name] for name in ordered], dtype=np.uint8)
    labels[np.asarray(support_mask, dtype=bool)] = region_values[dominant_idx[np.asarray(support_mask, dtype=bool)]]
    return labels


def generate_mida_parameter_maps(config: MidaParameterConfig | None = None) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    pd = require_dependency("pandas", "pip install pandas")
    nib = require_dependency("nibabel", "pip install nibabel")
    config = config or MidaParameterConfig()

    table = pd.read_parquet(config.distribution_path)
    validate_distribution_table(table)
    fraction_imgs = load_fraction_images(config.mida_dir)
    fractions = {name: img.get_fdata(dtype=np.float32) for name, img in fraction_imgs.items()}
    ref_img = next(iter(fraction_imgs.values()))
    brain_mask = load_lowres_brain_mask(config.mida_dir, sum(fractions.values()))
    fractions = normalize_fraction_maps(fractions, support_mask=brain_mask)
    fraction_sum = sum(fractions.values())
    support = brain_mask
    dominant = dominant_region_label(fractions, support)

    grouped = build_region_sample_pools(
        table,
        q_min=config.sample_quantile_min,
        q_max=config.sample_quantile_max,
    )
    delay_grouped = build_region_sample_pools(
        table,
        q_min=config.delay_sample_quantile_min,
        q_max=config.delay_sample_quantile_max,
    )
    T_grouped = build_region_sample_pools(
        table,
        q_min=config.T_sample_quantile_min,
        q_max=config.T_sample_quantile_max,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_idx in range(config.n_cases):
        rng = np.random.default_rng(config.seed + case_idx)
        case_id = f"case_{case_idx:03d}"
        case_dir = output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        maps = sample_partial_volume_maps(
            fractions,
            grouped,
            rng,
            component_fraction_min=config.component_fraction_min,
            parameter_sampling_mode=config.parameter_sampling_mode,
            subvoxel_samples_per_lowres_voxel=config.subvoxel_samples_per_lowres_voxel,
            parameter_patch_size_vox=config.parameter_patch_size_vox,
            delay_region_samples=delay_grouped,
            delay_sampling_mode=config.delay_sampling_mode,
            T_region_samples=T_grouped,
            T_sampling_mode=config.T_sampling_mode,
            spatial_smoothing_sigma_vox=config.spatial_smoothing_sigma_vox,
            rank_jitter=config.rank_jitter,
            within_tissue_variation_scale=config.within_tissue_variation_scale,
        )
        maps = nearest_fill_parameter_maps(maps, support)
        maps = smooth_parameter_maps_in_support(
            maps,
            support,
            sigma_vox=config.parameter_post_smooth_sigma_vox,
            blend=config.parameter_post_smooth_blend,
        )
        maps = nearest_fill_parameter_maps(maps, support)
        maps["CVR"][~support] = 0.0
        maps["delay"][~support] = 0.0
        maps["T"][~support] = 0.0

        case_paths = [
            save_nifti(maps["CVR"], ref_img, case_dir / "GT_CVR.nii.gz", dtype=np.float32),
            save_nifti(maps["delay"], ref_img, case_dir / "GT_delay.nii.gz", dtype=np.float32),
            save_nifti(maps["T"], ref_img, case_dir / "GT_T.nii.gz", dtype=np.float32),
            save_nifti(dominant, ref_img, case_dir / "GT_region_labels.nii.gz", dtype=np.uint8),
            save_nifti(
                fractions["vessel_like"],
                ref_img,
                case_dir / "GT_vessel_likelihood.nii.gz",
                dtype=np.float32,
            ),
        ]
        for name, fraction in fractions.items():
            case_paths.append(
                save_nifti(fraction, ref_img, case_dir / f"GT_fraction_{name}.nii.gz", dtype=np.float32)
            )
        qc_png = save_parameter_qc_plot(
            maps,
            fractions,
            dominant,
            case_dir / "mida_parameter_qc.png",
            title=f"MIDA partial-volume GT {case_id}",
        )
        canonical_dir = ""
        if config.save_fsleyes_canonical:
            canonical_dir = str(save_canonical_nifti_copies(case_paths, case_dir / "fsleyes_RAS"))
        highres_outputs = {}
        if case_idx in set(config.save_highres_case_indices):
            highres_outputs = save_highres_parameter_qc_case(config, table, rng, case_dir / "highres_qc")
        rows.append(
            {
                "case": case_id,
                "status": "ok",
                "GT_CVR": str(case_dir / "GT_CVR.nii.gz"),
                "GT_delay": str(case_dir / "GT_delay.nii.gz"),
                "GT_T": str(case_dir / "GT_T.nii.gz"),
                "GT_region_labels": str(case_dir / "GT_region_labels.nii.gz"),
                "GT_vessel_likelihood": str(case_dir / "GT_vessel_likelihood.nii.gz"),
                "mida_parameter_qc": str(qc_png),
                "fsleyes_canonical_dir": canonical_dir,
                "highres_qc_dir": str(highres_outputs.get("highres_qc_dir", "")),
            }
        )

    summary = output_dir / "mida_parameter_maps_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    return {"output_dir": output_dir, "summary": summary, "n_cases": config.n_cases}


def load_fraction_images(mida_dir: Path) -> dict[str, Any]:
    nib = require_dependency("nibabel", "pip install nibabel")
    mida_dir = Path(mida_dir)
    images = {}
    for name in MIDA_REGION_LABELS:
        path = mida_dir / f"mida_fraction_{name}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        images[name] = nib.load(str(path))
    return images


def load_lowres_brain_mask(mida_dir: Path, fraction_sum: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    nib = require_dependency("nibabel", "pip install nibabel")
    mask_path = Path(mida_dir) / "mida_brain_mask.nii.gz"
    if mask_path.exists():
        return nib.load(str(mask_path)).get_fdata(dtype=np.float32) > 0
    return np.asarray(fraction_sum, dtype=np.float32) > 0


def build_region_sample_pools(table: Any, q_min: float = 0.25, q_max: float = 0.75) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    if not 0.0 <= q_min < q_max <= 1.0:
        raise ValueError("sample quantile bounds must satisfy 0 <= q_min < q_max <= 1")
    fallback = table[["CVR", "delay", "T"]].to_numpy(dtype=np.float32)
    pools = {}
    for name, dist_region in MIDA_DISTRIBUTION_REGIONS.items():
        frame = table.loc[table["region"] == dist_region, ["CVR", "delay", "T"]].dropna()
        if len(frame) == 0:
            pools[name] = fallback
            continue
        lo = frame.quantile(q_min)
        hi = frame.quantile(q_max)
        central = frame[
            (frame["CVR"] >= lo["CVR"])
            & (frame["CVR"] <= hi["CVR"])
            & (frame["delay"] >= lo["delay"])
            & (frame["delay"] <= hi["delay"])
            & (frame["T"] >= lo["T"])
            & (frame["T"] <= hi["T"])
        ]
        if len(central) < 10:
            central = frame
        pools[name] = central.to_numpy(dtype=np.float32)
    return pools


def sample_partial_volume_maps(
    fractions: dict[str, Any],
    region_samples: dict[str, Any],
    rng: Any,
    component_fraction_min: float = 0.005,
    parameter_sampling_mode: str = "tissue_constant",
    subvoxel_samples_per_lowres_voxel: int = 125,
    parameter_patch_size_vox: tuple[int, int, int] = (3, 3, 2),
    delay_region_samples: dict[str, Any] | None = None,
    delay_sampling_mode: str = "tissue_constant",
    T_region_samples: dict[str, Any] | None = None,
    T_sampling_mode: str = "tissue_constant",
    spatial_smoothing_sigma_vox: float = 2.0,
    rank_jitter: float = 0.03,
    within_tissue_variation_scale: float = 1.0,
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    shape = next(iter(fractions.values())).shape
    accum = {name: np.zeros(shape, dtype=np.float32) for name in ("CVR", "delay", "T")}
    weight_sum = np.zeros(shape, dtype=np.float32)
    for region, fraction in fractions.items():
        frac = np.asarray(fraction, dtype=np.float32)
        keep = frac >= component_fraction_min
        n = int(np.count_nonzero(keep))
        if n == 0:
            continue
        if parameter_sampling_mode in {"tissue_constant", "region_constant"}:
            samples = np.repeat(sample_rows(region_samples[region], 1, rng), n, axis=0)
        elif parameter_sampling_mode in {
            "subvoxel_monte_carlo",
            "subvoxel_average",
            "fraction_average",
        }:
            samples = subvoxel_average_rows(
                region_samples[region],
                frac[keep],
                rng,
                n_subvoxels=subvoxel_samples_per_lowres_voxel,
            )
        elif parameter_sampling_mode in {"tissue_patch_distribution", "patch_distribution", "patchwise"}:
            samples = patchwise_rows(region_samples[region], keep, rng, parameter_patch_size_vox)
        elif parameter_sampling_mode in {"spatial", "spatial_quantile", "coherent"}:
            samples = spatially_coherent_rows(
                region_samples[region],
                keep,
                rng,
                smoothing_sigma_vox=spatial_smoothing_sigma_vox,
                rank_jitter=rank_jitter,
            )
            samples = shrink_samples_to_region_center(
                region_samples[region],
                samples,
                scale=within_tissue_variation_scale,
            )
        else:
            samples = sample_rows(region_samples[region], n, rng)
        if delay_sampling_mode in {"tissue_stratified_full", "stratified_full", "full_distribution"}:
            delay_pool = region_samples[region] if delay_region_samples is None else delay_region_samples[region]
            samples[:, 1] = stratified_parameter_values(
                delay_pool[:, 1],
                keep,
                frac,
                rng,
            )
        if T_sampling_mode in {"tissue_stratified_full", "stratified_full", "full_distribution"}:
            T_pool = region_samples[region] if T_region_samples is None else T_region_samples[region]
            samples[:, 2] = stratified_parameter_values(
                T_pool[:, 2],
                keep,
                frac,
                rng,
            )
        for col_idx, name in enumerate(("CVR", "delay", "T")):
            accum[name][keep] += frac[keep] * samples[:, col_idx]
        weight_sum[keep] += frac[keep]
    for name in accum:
        accum[name] = np.divide(
            accum[name],
            weight_sum,
            out=np.zeros_like(accum[name], dtype=np.float32),
            where=weight_sum > 0,
        )
    return accum


def patchwise_rows(rows: Any, keep_mask: Any, rng: Any, patch_size: tuple[int, int, int]) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    keep = np.asarray(keep_mask, dtype=bool)
    samples = np.zeros((int(np.count_nonzero(keep)), 3), dtype=np.float32)
    if samples.shape[0] == 0:
        return samples
    linear_positions = np.full(keep.shape, -1, dtype=np.int32)
    linear_positions[keep] = np.arange(samples.shape[0], dtype=np.int32)
    patch_size = tuple(max(1, int(v)) for v in patch_size)
    for x0 in range(0, keep.shape[0], patch_size[0]):
        for y0 in range(0, keep.shape[1], patch_size[1]):
            for z0 in range(0, keep.shape[2], patch_size[2]):
                patch = (
                    slice(x0, min(x0 + patch_size[0], keep.shape[0])),
                    slice(y0, min(y0 + patch_size[1], keep.shape[1])),
                    slice(z0, min(z0 + patch_size[2], keep.shape[2])),
                )
                patch_positions = linear_positions[patch]
                patch_positions = patch_positions[patch_positions >= 0]
                if patch_positions.size == 0:
                    continue
                samples[patch_positions] = sample_rows(rows, 1, rng)[0]
    return samples


def subvoxel_average_rows(rows: Any, fractions: Any, rng: Any, n_subvoxels: int = 125) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(rows, dtype=np.float32)
    frac = np.asarray(fractions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("Region distribution rows must have shape (n, 3)")
    if frac.ndim != 1:
        raise ValueError("fractions must be a 1D array for kept voxels")
    n_subvoxels = max(1, int(n_subvoxels))
    draw_counts = np.rint(np.clip(frac, 0.0, 1.0) * n_subvoxels).astype(np.int16)
    draw_counts = np.clip(draw_counts, 1, n_subvoxels)
    out = np.zeros((frac.size, 3), dtype=np.float32)
    for count in np.unique(draw_counts):
        positions = np.where(draw_counts == count)[0]
        if positions.size == 0:
            continue
        if int(count) == 1:
            out[positions] = sample_rows(arr, positions.size, rng)
            continue
        idx = rng.integers(0, arr.shape[0], size=(positions.size, int(count)))
        out[positions] = arr[idx].mean(axis=1, dtype=np.float32)
    return out


def stratified_parameter_values(values: Any, keep_mask: Any, fraction: Any, rng: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("Cannot stratify an empty delay distribution")
    keep = np.asarray(keep_mask, dtype=bool)
    n = int(np.count_nonzero(keep))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    q_offset = float(rng.random()) / max(n, 1)
    quantiles = np.clip((np.arange(n, dtype=np.float64) + q_offset) / n, 0.0, 1.0)
    sampled = np.quantile(vals, quantiles, method="nearest").astype(np.float32)
    score = tissue_rank_score(keep, fraction)
    order = np.argsort(score[keep], kind="mergesort")
    out = np.empty(n, dtype=np.float32)
    out[order] = np.sort(sampled, kind="mergesort")
    return out


def tissue_rank_score(keep_mask: Any, fraction: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    keep = np.asarray(keep_mask, dtype=bool)
    frac = np.asarray(fraction, dtype=np.float32)
    score = np.zeros_like(frac, dtype=np.float32)
    if not np.any(keep):
        return score
    depth = ndi.distance_transform_edt(keep).astype(np.float32)
    if float(depth.max()) > 0:
        depth /= float(depth.max())
    coords = np.column_stack(np.where(keep)).astype(np.float32)
    center = coords.mean(axis=0)
    scale = np.maximum(np.asarray(keep.shape, dtype=np.float32), 1.0)
    radius = np.linalg.norm((coords - center) / scale, axis=1)
    radial = np.zeros_like(frac, dtype=np.float32)
    if float(radius.max()) > 0:
        radial_values = 1.0 - radius / float(radius.max())
    else:
        radial_values = np.ones(coords.shape[0], dtype=np.float32)
    radial[tuple(coords.astype(int).T)] = radial_values.astype(np.float32)
    score[keep] = 0.70 * frac[keep] + 0.25 * depth[keep] + 0.05 * radial[keep]
    return score


def spatially_coherent_rows(
    rows: Any,
    keep_mask: Any,
    rng: Any,
    smoothing_sigma_vox: float = 2.0,
    rank_jitter: float = 0.03,
) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    keep = np.asarray(keep_mask, dtype=bool)
    n = int(np.count_nonzero(keep))
    sampled = sample_rows(rows, n, rng)
    if n <= 1 or smoothing_sigma_vox <= 0:
        return sampled

    field = rng.normal(size=keep.shape).astype(np.float32)
    field = ndi.gaussian_filter(field, sigma=float(smoothing_sigma_vox))
    field_values = field[keep]
    if rank_jitter > 0:
        field_values = field_values + float(rank_jitter) * rng.normal(size=n)

    latent = distribution_latent_score(sampled)
    spatial_order = np.argsort(field_values, kind="mergesort")
    sample_order = np.argsort(latent, kind="mergesort")
    coherent = np.empty_like(sampled)
    coherent[spatial_order] = sampled[sample_order]
    return coherent


def distribution_latent_score(samples: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(samples, dtype=np.float32)
    score = np.zeros(arr.shape[0], dtype=np.float32)
    for col in range(arr.shape[1]):
        values = arr[:, col]
        scale = float(np.nanstd(values))
        if scale > 0:
            score += (values - float(np.nanmean(values))) / scale
    return score


def shrink_samples_to_region_center(pool_rows: Any, samples: Any, scale: float) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    if scale >= 0.999:
        return np.asarray(samples, dtype=np.float32)
    pool = np.asarray(pool_rows, dtype=np.float32)
    sampled = np.asarray(samples, dtype=np.float32)
    center = np.nanmedian(pool, axis=0).astype(np.float32)
    q01 = np.nanquantile(pool, 0.01, axis=0).astype(np.float32)
    q99 = np.nanquantile(pool, 0.99, axis=0).astype(np.float32)
    shrunk = center + float(scale) * (sampled - center)
    return np.clip(shrunk, q01, q99).astype(np.float32)


def nearest_fill_parameter_maps(maps: dict[str, Any], support_mask: Any) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    support = np.asarray(support_mask, dtype=bool)
    finite = support.copy()
    nonzero_any = np.zeros_like(support, dtype=bool)
    for values in maps.values():
        arr = np.asarray(values)
        finite &= np.isfinite(arr)
        nonzero_any |= np.abs(arr) > 0
    valid = support & finite & nonzero_any
    missing = support & ~valid
    if not np.any(missing) or not np.any(valid):
        return maps
    _, nearest = ndi.distance_transform_edt(~valid, return_indices=True)
    filled = {}
    for name, values in maps.items():
        arr = np.asarray(values, dtype=np.float32).copy()
        arr[missing] = arr[tuple(nearest)][missing]
        filled[name] = arr
    return filled


def smooth_parameter_maps_in_support(
    maps: dict[str, Any],
    support_mask: Any,
    sigma_vox: float = 0.75,
    blend: float = 0.75,
) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    ndi = require_dependency("scipy.ndimage", "pip install scipy")
    sigma = float(sigma_vox)
    if sigma <= 0:
        return maps
    support = np.asarray(support_mask, dtype=bool)
    weight = support.astype(np.float32)
    smooth_weight = ndi.gaussian_filter(weight, sigma=sigma)
    alpha = min(max(float(blend), 0.0), 1.0)
    smoothed = {}
    for name, values in maps.items():
        arr = np.asarray(values, dtype=np.float32)
        smooth_num = ndi.gaussian_filter(np.where(support, arr, 0.0).astype(np.float32), sigma=sigma)
        local_mean = np.divide(
            smooth_num,
            smooth_weight,
            out=arr.copy(),
            where=smooth_weight > 1e-6,
        )
        rank_field = (1.0 - alpha) * arr + alpha * local_mean
        out = arr.copy()
        inside = support & np.isfinite(arr) & np.isfinite(rank_field)
        if np.count_nonzero(inside) > 1:
            order = np.argsort(rank_field[inside], kind="mergesort")
            sorted_values = np.sort(arr[inside], kind="mergesort")
            ranked = np.empty_like(sorted_values, dtype=np.float32)
            ranked[order] = sorted_values
            out[inside] = ranked
        out[~support] = 0.0
        smoothed[name] = out.astype(np.float32)
    return smoothed


def sample_rows(rows: Any, n: int, rng: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        raise ValueError("Region distribution rows must have shape (n, 3)")
    idx = rng.integers(0, arr.shape[0], size=n)
    return arr[idx]


def save_highres_parameter_qc_case(
    config: MidaParameterConfig,
    table: Any,
    rng: Any,
    out_dir: Path,
) -> dict[str, Path]:
    np = require_dependency("numpy", "pip install numpy")
    nib = require_dependency("nibabel", "pip install nibabel")
    out_dir.mkdir(parents=True, exist_ok=True)
    fraction_imgs = load_highres_masks(config.highres_dir)
    ref_img = next(iter(fraction_imgs.values()))
    masks = {name: img.get_fdata(dtype=np.float32) > 0 for name, img in fraction_imgs.items()}
    support = np.zeros(ref_img.shape[:3], dtype=bool)
    for mask in masks.values():
        support |= mask
    grouped = build_region_sample_pools(
        table,
        q_min=config.sample_quantile_min,
        q_max=config.sample_quantile_max,
    )

    outputs = {"highres_qc_dir": out_dir}
    region_vectors = {
        region: sample_rows(rows, 1, rng)[0]
        for region, rows in grouped.items()
    }
    for param_idx, param_name in enumerate(("CVR", "delay", "T")):
        out = np.zeros(ref_img.shape[:3], dtype=np.float32)
        for region, mask in masks.items():
            if not np.any(mask):
                continue
            out[mask] = region_vectors[region][param_idx]
        out[~support] = 0.0
        path = out_dir / f"highres_GT_{param_name}.nii.gz"
        save_nifti(out, ref_img, path, dtype=np.float32)
        outputs[f"highres_GT_{param_name}"] = path
    return outputs


def load_highres_masks(highres_dir: Path) -> dict[str, Any]:
    nib = require_dependency("nibabel", "pip install nibabel")
    highres_dir = Path(highres_dir)
    images = {}
    for name in MIDA_REGION_LABELS:
        path = highres_dir / f"mida_mask_{name}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        images[name] = nib.load(str(path))
    return images


def validate_distribution_table(table: Any) -> None:
    required = {"region", "CVR", "delay", "T"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Distribution table is missing required columns: {missing}")
    if len(table) == 0:
        raise ValueError("Distribution table is empty")


def save_nifti(data: Any, ref_img: Any, out_path: Path, dtype: Any) -> Path:
    np = require_dependency("numpy", "pip install numpy")
    nib = require_dependency("nibabel", "pip install nibabel")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data).astype(dtype), ref_img.affine, ref_img.header)
    img.set_data_dtype(dtype)
    nib.save(img, str(out_path))
    return out_path


def save_canonical_nifti_copies(paths: list[Path], out_dir: Path) -> Path:
    nib = require_dependency("nibabel", "pip install nibabel")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        img = nib.load(str(path))
        canonical = nib.as_closest_canonical(img)
        canonical.set_data_dtype(img.get_data_dtype())
        nib.save(canonical, str(out_dir / path.name))
    return out_dir


def translated_affine(affine: Any, start: tuple[int, int, int]) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    out = np.array(affine, dtype=float, copy=True)
    out[:3, 3] = out[:3, 3] + out[:3, :3] @ np.asarray(start, dtype=float)
    return out


def downsampled_affine(affine: Any, factor: tuple[int, int, int]) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    out = np.array(affine, dtype=float, copy=True)
    for axis, value in enumerate(factor):
        out[:3, axis] *= float(value)
    return out


def save_fraction_qc_plot(fractions: dict[str, Any], dominant: Any, out_path: Path) -> Path:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    np = require_dependency("numpy", "pip install numpy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = dominant.shape[2] // 2
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, name in zip(axes.flat[:5], MIDA_REGION_LABELS, strict=False):
        im = ax.imshow(np.rot90(fractions[name][..., z]), cmap="magma", vmin=0, vmax=1)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes.flat[5].imshow(np.rot90(dominant[..., z]), cmap="tab10", vmin=0, vmax=5)
    axes.flat[5].set_title("dominant label")
    axes.flat[5].axis("off")
    fig.suptitle("MIDA 2.5 mm tissue fractions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_parameter_qc_plot(
    maps: dict[str, Any],
    fractions: dict[str, Any],
    dominant: Any,
    out_path: Path,
    title: str,
) -> Path:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    np = require_dependency("numpy", "pip install numpy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = dominant.shape[2] // 2
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, name, cmap in [
        (axes[0, 0], "CVR", "viridis"),
        (axes[0, 1], "delay", "plasma"),
        (axes[0, 2], "T", "magma"),
    ]:
        values = np.asarray(maps[name])
        vmax = float(np.nanpercentile(values[values != 0], 98)) if np.any(values != 0) else 1.0
        im = ax.imshow(np.rot90(values[..., z]), cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(f"GT_{name}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[1, 0].imshow(np.rot90(dominant[..., z]), cmap="tab10", vmin=0, vmax=5)
    axes[1, 0].set_title("dominant region")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(np.rot90(fractions["vessel_like"][..., z]), cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title("vessel likelihood")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(np.rot90(sum(fractions.values())[..., z]), cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("fraction sum")
    axes[1, 2].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_mida_segmentation_summary(
    out_path: Path,
    raw_masks: dict[str, Any],
    cropped_masks: dict[str, Any],
    filled_highres: dict[str, Any],
    fraction_maps: dict[str, Any],
    raw_counts: dict[str, int],
    crop_start: tuple[int, int, int],
    crop_stop: tuple[int, int, int],
    full_shape: tuple[int, int, int],
    config: MidaSegmentationConfig,
) -> Path:
    pd = require_dependency("pandas", "pip install pandas")
    np = require_dependency("numpy", "pip install numpy")
    rows = []
    for name in MIDA_REGION_LABELS:
        raw = int(raw_counts[name])
        cropped = int(np.asarray(cropped_masks[name], dtype=bool).sum())
        filled_sum = float(np.asarray(filled_highres[name], dtype=np.float32).sum())
        fraction_sum = float(np.asarray(fraction_maps[name], dtype=np.float32).sum())
        cropped_percent = 100.0 * cropped / raw if raw else 0.0
        rows.append(
            {
                "region": name,
                "raw_voxels": raw,
                "cropped_voxels": cropped,
                "cropped_percent": cropped_percent,
                "cropped_tissue_percent_loss": max(0.0, 100.0 - cropped_percent),
                "filled_highres_sum": filled_sum,
                "lowres_fraction_sum": fraction_sum,
                "crop_start": str(crop_start),
                "crop_stop": str(crop_stop),
                "full_shape": str(full_shape),
                "target_shape": str(config.target_shape),
                "downsample_factor": str(config.downsample_factor),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
