from __future__ import annotations

from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency
from hybrid_cvr.segmentation.boundary_uncertainty import tissue_boundary_uncertainty
from hybrid_cvr.segmentation.tissue_masks import (
    binary_erosion_numpy,
    high_confidence_core,
    make_brain_from_tissues,
)


def build_test_tissue_outputs(gm: Any, wm: Any, csf: Any) -> dict[str, Any]:
    brain = make_brain_from_tissues(gm, wm, csf)
    gm_core = high_confidence_core(gm)
    wm_core = high_confidence_core(wm)
    csf_core = high_confidence_core(csf, erosion_iters=0)
    boundary = tissue_boundary_uncertainty({"gm": gm, "wm": wm, "csf": csf}, brain)
    return {
        "gm_prob_or_mask": gm,
        "wm_prob_or_mask": wm,
        "csf_prob_or_mask": csf,
        "gm_core": gm_core,
        "wm_core": wm_core,
        "csf_core": csf_core,
        "brain_mask": brain,
        "tissue_boundary_uncertainty": boundary,
    }


def simple_t1_brain_mask(t1_data: Any, quantile: float = 0.15) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    data = np.nan_to_num(np.asarray(t1_data, dtype=float))
    positive = data[data > 0]
    if positive.size == 0:
        return np.zeros(data.shape, dtype="uint8")
    threshold = np.quantile(positive, quantile)
    return (data > threshold).astype("uint8")


def fallback_bold_tissue_segmentation(mean_bold: Any, brain_mask: Any) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    mean_bold = np.nan_to_num(np.asarray(mean_bold, dtype=float))
    brain = np.asarray(brain_mask) > 0
    if not np.any(brain):
        raise ValueError("Brain mask is empty; cannot create fallback tissue masks")
    values = mean_bold[brain]
    lo, hi = np.percentile(values, [5, 98])
    norm = np.clip((mean_bold - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    inner_brain = binary_erosion_numpy(brain)
    outer_shell = brain & ~inner_brain
    q_csf, q_wm = np.quantile(norm[brain], [0.18, 0.52])
    hard_csf = (norm <= q_csf) & inner_brain
    hard_wm = (norm > q_csf) & (norm <= q_wm) & brain
    hard_gm = (norm > q_wm) & brain

    # BOLD-space fallback: intensity-derived tissue-like probabilities, not anatomical labels.
    centers = np.asarray(
        [
            float(np.median(norm[hard_gm])) if np.any(hard_gm) else 0.75,
            float(np.median(norm[hard_wm])) if np.any(hard_wm) else 0.45,
            float(np.median(norm[hard_csf])) if np.any(hard_csf) else 0.15,
        ]
    )
    width = 0.07
    gm, wm, csf = [np.exp(-0.5 * ((norm - center) / width) ** 2) for center in centers]
    csf *= inner_brain
    gm *= brain
    wm *= brain
    csf[outer_shell] = 0.0
    stack = np.stack([gm, wm, csf], axis=0) * brain
    stack_sum = stack.sum(axis=0)
    stack_sum = np.where(stack_sum > 0, stack_sum, 1.0)
    gm_prob, wm_prob, csf_prob = (stack / stack_sum).astype("float32")

    gm_mask = hard_gm.astype("uint8")
    wm_mask = hard_wm.astype("uint8")
    csf_mask = hard_csf.astype("uint8")
    gm_core = high_confidence_core(gm_prob, threshold=0.60, erosion_iters=0)
    wm_core = high_confidence_core(wm_prob, threshold=0.50, erosion_iters=0)
    csf_core = high_confidence_core(csf_prob, threshold=0.50, erosion_iters=0)
    gm_split = _split_cortical_subcortical_gm(gm_prob, gm_mask, brain)
    ventricle = _central_component_proxy(csf_prob, brain)
    boundary = tissue_boundary_uncertainty({"gm": gm_prob, "wm": wm_prob, "csf": csf_prob}, brain)
    return {
        "gm_prob_or_mask": gm_prob,
        "wm_prob_or_mask": wm_prob,
        "csf_prob_or_mask": csf_prob,
        "cortical_gm_prob_or_mask": gm_split["cortical_gm_prob_or_mask"],
        "subcortical_gm_prob_or_mask": gm_split["subcortical_gm_prob_or_mask"],
        "gm_mask": gm_mask,
        "wm_mask": wm_mask,
        "csf_mask": csf_mask,
        "cortical_gm_mask": gm_split["cortical_gm_mask"],
        "subcortical_gm_mask": gm_split["subcortical_gm_mask"],
        "gm_core": gm_core,
        "wm_core": wm_core,
        "csf_core": csf_core,
        "cortical_gm_core": gm_split["cortical_gm_core"],
        "subcortical_gm_core": gm_split["subcortical_gm_core"],
        "ventricle_mask": ventricle,
        "brain_mask_bold": brain.astype("uint8"),
        "tissue_boundary_uncertainty": boundary,
    }


def _brain_distance_transform(brain_mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    brain = np.asarray(brain_mask, dtype=bool)
    try:
        ndi = require_dependency("scipy.ndimage", "pip install scipy")
        return ndi.distance_transform_edt(brain)
    except RuntimeError:
        distance = np.zeros(brain.shape, dtype=float)
        shell = brain.copy()
        depth = 1.0
        while np.any(shell):
            eroded = binary_erosion_numpy(shell)
            distance[shell & ~eroded] = depth
            shell = eroded
            depth += 1.0
        return distance


def _centrality_weight(brain_mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    brain = np.asarray(brain_mask, dtype=bool)
    coords = np.indices(brain.shape, dtype=float)
    brain_coords = [axis[brain] for axis in coords]
    mins = np.asarray([axis.min() for axis in brain_coords], dtype=float)
    maxs = np.asarray([axis.max() for axis in brain_coords], dtype=float)
    center = (mins + maxs) / 2.0
    half_width = np.maximum((maxs - mins) / 2.0, 1.0)
    radius2 = np.zeros(brain.shape, dtype=float)
    for axis, midpoint, width in zip(coords, center, half_width, strict=True):
        radius2 += ((axis - midpoint) / width) ** 2
    return np.exp(-0.5 * radius2 / (0.55**2)) * brain


def _split_cortical_subcortical_gm(gm_prob: Any, gm_mask: Any, brain_mask: Any) -> dict[str, Any]:
    np = require_dependency("numpy", "pip install numpy")
    gm_prob = np.asarray(gm_prob, dtype=float)
    gm = np.asarray(gm_mask, dtype=bool)
    brain = np.asarray(brain_mask, dtype=bool)
    if not np.any(gm):
        empty_prob = np.zeros_like(gm_prob, dtype="float32")
        empty_mask = np.zeros_like(gm, dtype="uint8")
        return {
            "cortical_gm_prob_or_mask": empty_prob,
            "subcortical_gm_prob_or_mask": empty_prob,
            "cortical_gm_mask": empty_mask,
            "subcortical_gm_mask": empty_mask,
            "cortical_gm_core": empty_mask,
            "subcortical_gm_core": empty_mask,
        }

    distance = _brain_distance_transform(brain)
    gm_distances = distance[gm]
    inner_threshold = max(2.0, float(np.quantile(gm_distances, 0.45)))
    inner_weight = np.clip(distance / max(float(np.quantile(distance[brain], 0.80)), 1.0), 0.0, 1.0)
    score = gm_prob * _centrality_weight(brain) * inner_weight
    gm_count = int(np.sum(gm))
    target_voxels = min(max(1, int(round(0.18 * gm_count))), max(1, gm_count - 1))
    candidates = gm & (distance >= inner_threshold)
    candidate_indices = np.flatnonzero(candidates)
    if candidate_indices.size < target_voxels:
        candidate_indices = np.flatnonzero(gm)
    ranked = candidate_indices[np.argsort(score.ravel()[candidate_indices])[::-1]]
    subcortical = np.zeros_like(gm, dtype=bool)
    np.put(subcortical, ranked[:target_voxels], True)
    cortical = gm & ~subcortical

    cortical_prob = (gm_prob * cortical).astype("float32")
    subcortical_prob = (gm_prob * subcortical).astype("float32")
    return {
        "cortical_gm_prob_or_mask": cortical_prob,
        "subcortical_gm_prob_or_mask": subcortical_prob,
        "cortical_gm_mask": cortical.astype("uint8"),
        "subcortical_gm_mask": subcortical.astype("uint8"),
        "cortical_gm_core": high_confidence_core(cortical_prob, threshold=0.60, erosion_iters=0),
        "subcortical_gm_core": high_confidence_core(subcortical_prob, threshold=0.60, erosion_iters=0),
    }


def _central_component_proxy(csf_core: Any, brain_mask: Any) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    csf_values = np.asarray(csf_core, dtype=float)
    brain = np.asarray(brain_mask) > 0
    if csf_values.ndim != 3 or not np.any(brain):
        return np.zeros_like(csf_values, dtype="uint8")
    coords = np.indices(csf_values.shape)
    brain_coords = [axis[brain] for axis in coords]
    mins = np.asarray([axis.min() for axis in brain_coords])
    maxs = np.asarray([axis.max() for axis in brain_coords])
    lower = mins + 0.25 * (maxs - mins)
    upper = mins + 0.75 * (maxs - mins)
    central = np.ones(csf_values.shape, dtype=bool)
    for axis, lo, hi in zip(coords, lower, upper, strict=True):
        central &= (axis >= lo) & (axis <= hi)
    candidate_values = csf_values[central & brain]
    if candidate_values.size == 0:
        return np.zeros_like(csf_values, dtype="uint8")
    threshold = max(float(np.quantile(candidate_values, 0.98)), 0.60)
    ventricle = (csf_values >= threshold) & central & brain
    if not np.any(ventricle):
        threshold = max(float(np.quantile(candidate_values, 0.95)), 0.50)
        ventricle = (csf_values >= threshold) & central & brain
    return ventricle.astype("uint8")


def save_fallback_segmentation(
    mean_bold_path: str | Path,
    brain_mask_path: str | Path,
    t1w_path: str | Path,
    out_dir: str | Path,
    generate_qc_plot: bool = False,
) -> dict[str, Path | str]:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_img = nib.load(str(mean_bold_path))
    mean_bold = mean_img.get_fdata(dtype=np.float32)
    brain_img = nib.load(str(brain_mask_path))
    brain_mask = brain_img.get_fdata(dtype=np.float32) > 0
    maps = fallback_bold_tissue_segmentation(mean_bold, brain_mask)

    outputs: dict[str, Path | str] = {"method": "fallback_bold_intensity"}
    for name, data in maps.items():
        path = out_dir / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, mean_img.affine, mean_img.header), str(path))
        outputs[name] = path

    t1_img = nib.load(str(t1w_path))
    t1_mask = simple_t1_brain_mask(t1_img.get_fdata(dtype=np.float32))
    t1_mask_path = out_dir / "brain_mask_t1.nii.gz"
    nib.save(nib.Nifti1Image(t1_mask, t1_img.affine, t1_img.header), str(t1_mask_path))
    outputs["brain_mask_t1"] = t1_mask_path

    if generate_qc_plot:
        outputs["segmentation_qc"] = save_segmentation_qc(mean_bold, maps, out_dir / "segmentation_qc.png")
    return outputs


def save_segmentation_qc(mean_bold: Any, maps: dict[str, Any], out: str | Path) -> Path:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    np = require_dependency("numpy", "pip install numpy")
    out = Path(out)
    z = mean_bold.shape[2] // 2 if np.asarray(mean_bold).ndim == 3 else 0
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    panels = [
        ("mean BOLD", mean_bold[..., z], "gray"),
        ("GM-like", maps["gm_prob_or_mask"][..., z], "Reds"),
        ("WM-like", maps["wm_prob_or_mask"][..., z], "Blues"),
        ("CSF-like", maps["csf_prob_or_mask"][..., z], "Greens"),
        ("boundary", maps["tissue_boundary_uncertainty"][..., z], "magma"),
        ("brain", maps["brain_mask_bold"][..., z], "gray"),
    ]
    for ax, (title, image, cmap) in zip(axes.ravel(), panels, strict=True):
        ax.imshow(np.rot90(image), cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
