from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from hybrid_cvr.inference.predict_sim import _load_model, _save_float_nifti
from hybrid_cvr.simulation.mida_bold import TISSUE_FRACTION_NAMES, tissue_boundary_uncertainty
from hybrid_cvr.training.on_the_fly_dataset import ON_THE_FLY_FEATURE_NAMES


def real_inference_outputs() -> list[str]:
    return [
        "predicted_CVR.nii.gz",
        "predicted_delay.nii.gz",
        "predicted_T.nii.gz",
        "predicted_uncertainty_sigma.nii.gz",
        "reconstructed_BOLD_PSC_mean.nii.gz",
        "residual_rms.nii.gz",
        "real_inference_qc_report.png",
    ]


def infer_real_session(
    checkpoint: str | Path,
    config: dict[str, Any],
    processed_dir: str | Path,
    segmentation_dir: str | Path,
    out_dir: str | Path,
    *,
    vessel_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run trained self-supervised U-Net/PINN on one processed real BOLD session.

    The network was trained on BOLD PSC slices plus tissue maps. Real inference
    therefore uses the MCFLIRT-derived `bold_psc.nii.gz` and the BOLD-space masks
    already created by preprocessing/segmentation.
    """

    torch = __import__("torch")
    np = __import__("numpy")
    nib = __import__("nibabel")
    plt = __import__("matplotlib.pyplot", fromlist=["pyplot"])

    processed = Path(processed_dir)
    segmentation = Path(segmentation_dir)
    vessels = Path(vessel_dir) if vessel_dir is not None else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    input_channels = list(payload.get("input_channel_names") or ON_THE_FLY_FEATURE_NAMES)
    if input_channels != list(ON_THE_FLY_FEATURE_NAMES):
        raise ValueError(
            "Real inference currently builds the standard on-the-fly feature stack; "
            f"checkpoint expects {input_channels}"
        )

    psc_img = nib.load(str(processed / "bold_psc.nii.gz"))
    psc = np.asarray(psc_img.get_fdata(dtype=np.float32), dtype=np.float32)
    if psc.ndim != 4:
        raise ValueError(f"Expected 4D BOLD PSC image, got {psc.shape}: {processed}")
    spatial_shape = psc.shape[:3]
    n_time = psc.shape[3]

    mask = _load_mask(processed / "valid_signal_mask.nii.gz", spatial_shape, nib)
    if not np.any(mask):
        mask = _load_mask(segmentation / "brain_mask_bold.nii.gz", spatial_shape, nib)
    if not np.any(mask):
        raise ValueError(f"No valid brain mask found for {processed}")

    etco2 = _load_delta_etco2(processed / "etco2_resampled.tsv", n_time, np)
    time_grid = _load_time_grid(processed / "etco2_resampled.tsv", n_time, config, np)

    baseline = _load_baseline(processed, spatial_shape, mask, nib, np)
    fractions = _load_tissue_fractions(segmentation, vessels, spatial_shape, mask, nib, np)
    vessel_likelihood = _load_optional_like(
        vessels / "vessel_likelihood.nii.gz" if vessels is not None else None,
        spatial_shape,
        nib,
        np,
        default=fractions[4],
    )
    boundary = tissue_boundary_uncertainty(fractions, mask)
    tcnr = _estimate_real_tcnr(psc, etco2, mask, np)
    feature_volume = _build_real_feature_volume(
        psc, baseline, tcnr, mask, fractions, vessel_likelihood, boundary, etco2, np
    )

    model = _load_model(payload, config, in_channels=len(input_channels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    pred_cvr = np.zeros(spatial_shape, dtype=np.float32)
    pred_delay = np.zeros(spatial_shape, dtype=np.float32)
    pred_T = np.zeros(spatial_shape, dtype=np.float32)
    pred_sigma = np.zeros(spatial_shape, dtype=np.float32)
    recon_mean = np.zeros(spatial_shape, dtype=np.float32)
    residual_rms = np.zeros(spatial_shape, dtype=np.float32)
    eval_mask = np.zeros(spatial_shape, dtype=bool)
    slice_rows: list[dict[str, Any]] = []

    slice_axis = _slice_axis(psc_img, str(config.get("dataset", {}).get("slice_axis", "axial")), np, nib)
    valid_slices = _valid_slice_indices(mask, slice_axis, np)
    slice_range = config.get("dataset", {}).get("slice_index_range")
    use_training_slice_range = bool(
        config.get("real_inference", {}).get("use_training_slice_range", False)
    )
    if use_training_slice_range and slice_range is not None:
        lo, hi = int(slice_range[0]), int(slice_range[1])
        if hi < lo:
            lo, hi = hi, lo
        ranged = [idx for idx in valid_slices if lo <= idx <= hi]
        if ranged:
            valid_slices = ranged

    time_t = torch.as_tensor(time_grid, device=device, dtype=torch.float32)
    etco2_t = torch.as_tensor(etco2, device=device, dtype=torch.float32)
    with torch.no_grad():
        for idx in valid_slices:
            feat = np.take(feature_volume, idx, axis=slice_axis + 1)
            tissue = np.take(fractions, idx, axis=slice_axis + 1)
            sl_mask = np.take(mask, idx, axis=slice_axis)
            if not np.any(sl_mask):
                continue
            features_t = torch.as_tensor(feat[None], device=device, dtype=torch.float32)
            tissue_t = torch.as_tensor(tissue[None], device=device, dtype=torch.float32)
            mask_t = torch.as_tensor(sl_mask[None], device=device, dtype=torch.float32)
            pred = model(features_t, etco2=etco2_t, time_grid=time_t, mask=mask_t, tissue_maps=tissue_t)
            cvr = pred["cvr"][0].detach().cpu().numpy().astype(np.float32)
            delay = pred["delay"][0].detach().cpu().numpy().astype(np.float32)
            T = pred["T"][0].detach().cpu().numpy().astype(np.float32)
            sigma = pred["sigma"][0].detach().cpu().numpy().astype(np.float32)
            y_hat = pred["bold_psc_hat"][0].detach().cpu().numpy().astype(np.float32)
            observed = np.moveaxis(np.take(psc, idx, axis=slice_axis), -1, 0).astype(np.float32)
            rms = np.sqrt(np.mean((observed - y_hat) ** 2, axis=0)).astype(np.float32)
            _put_slice(pred_cvr, idx, slice_axis, cvr, sl_mask)
            _put_slice(pred_delay, idx, slice_axis, delay, sl_mask)
            _put_slice(pred_T, idx, slice_axis, T, sl_mask)
            _put_slice(pred_sigma, idx, slice_axis, sigma, sl_mask)
            _put_slice(recon_mean, idx, slice_axis, np.mean(y_hat, axis=0), sl_mask)
            _put_slice(residual_rms, idx, slice_axis, rms, sl_mask)
            _put_slice(eval_mask, idx, slice_axis, sl_mask.astype(np.float32), sl_mask)
            slice_rows.append(
                {
                    "slice_index": int(idx),
                    "brain_voxels": int(sl_mask.sum()),
                    "cvr_median": float(np.median(cvr[sl_mask])),
                    "delay_median": float(np.median(delay[sl_mask])),
                    "T_median": float(np.median(T[sl_mask])),
                    "sigma_median": float(np.median(sigma[sl_mask])),
                    "residual_rms_median": float(np.median(rms[sl_mask])),
                }
            )

    outputs = {
        "predicted_CVR": _save_float_nifti(pred_cvr, psc_img, out / "predicted_CVR.nii.gz", nib),
        "predicted_delay": _save_float_nifti(
            pred_delay, psc_img, out / "predicted_delay.nii.gz", nib
        ),
        "predicted_T": _save_float_nifti(pred_T, psc_img, out / "predicted_T.nii.gz", nib),
        "predicted_uncertainty_sigma": _save_float_nifti(
            pred_sigma, psc_img, out / "predicted_uncertainty_sigma.nii.gz", nib
        ),
        "reconstructed_BOLD_PSC_mean": _save_float_nifti(
            recon_mean, psc_img, out / "reconstructed_BOLD_PSC_mean.nii.gz", nib
        ),
        "residual_rms": _save_float_nifti(residual_rms, psc_img, out / "residual_rms.nii.gz", nib),
    }
    _write_slice_summary(out / "slice_summary.csv", slice_rows)
    _write_qc_png(out / "real_inference_qc_report.png", pred_cvr, pred_delay, pred_T, pred_sigma, residual_rms, eval_mask, plt, np)
    metadata = {
        "checkpoint": str(checkpoint),
        "processed_dir": str(processed),
        "segmentation_dir": str(segmentation),
        "vessel_dir": str(vessels) if vessels is not None else None,
        "input_channel_names": input_channels,
        "slice_axis": int(slice_axis),
        "slice_axis_name": str(config.get("dataset", {}).get("slice_axis", "axial")),
        "slice_index_range": list(slice_range) if slice_range is not None else None,
        "used_training_slice_range": use_training_slice_range,
        "processed_slices": len(slice_rows),
        "source_bold_psc": str(processed / "bold_psc.nii.gz"),
        "source_motion_corrected_bold": str(processed / "bold_mcflirt.nii.gz"),
    }
    (out / "prediction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**outputs, "metadata": out / "prediction_metadata.json", "processed_slices": len(slice_rows), "device": str(device)}


def _load_mask(path: Path, shape: tuple[int, int, int], nib: Any) -> Any:
    np = __import__("numpy")
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    data = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    if data.shape != shape:
        raise ValueError(f"Mask shape mismatch for {path}: {data.shape} != {shape}")
    return data > 0


def _load_optional_like(path: Path | None, shape: tuple[int, int, int], nib: Any, np: Any, *, default: Any | None = None) -> Any:
    if path is not None and path.exists():
        data = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        if data.shape != shape:
            raise ValueError(f"Shape mismatch for {path}: {data.shape} != {shape}")
        return data
    if default is not None:
        return np.asarray(default, dtype=np.float32)
    return np.zeros(shape, dtype=np.float32)


def _load_baseline(processed: Path, shape: tuple[int, int, int], mask: Any, nib: Any, np: Any) -> Any:
    path = processed / "baseline_bold.nii.gz"
    if not path.exists():
        path = processed / "mean_bold.nii.gz"
    baseline = _load_optional_like(path, shape, nib, np)
    inside = baseline[np.asarray(mask, dtype=bool)]
    scale = 320.0 / max(float(np.nanmedian(inside)), 1e-6) if inside.size else 1.0
    baseline = np.clip(baseline * scale, 0.0, 800.0).astype(np.float32)
    baseline[~np.asarray(mask, dtype=bool)] = 0.0
    return baseline


def _load_tissue_fractions(segmentation: Path, vessels: Path | None, shape: tuple[int, int, int], mask: Any, nib: Any, np: Any) -> Any:
    paths = {
        "cortical_gm": segmentation / "cortical_gm_mask.nii.gz",
        "subcortical_gm": segmentation / "subcortical_gm_mask.nii.gz",
        "wm": segmentation / "wm_mask.nii.gz",
        "vcsf": segmentation / "ventricle_mask.nii.gz",
        "vessel_like": vessels / "vessel_mask_high_confidence.nii.gz" if vessels is not None else None,
    }
    fractions = []
    for name in TISSUE_FRACTION_NAMES:
        arr = _load_optional_like(paths[name], shape, nib, np)
        fractions.append(np.clip(arr, 0.0, 1.0))
    frac = np.stack(fractions, axis=0).astype(np.float32)
    frac[:, ~np.asarray(mask, dtype=bool)] = 0.0
    total = np.sum(frac, axis=0, keepdims=True)
    frac = np.where(total > 1.0, frac / np.maximum(total, 1e-6), frac)
    return frac.astype(np.float32)


def _load_delta_etco2(path: Path, n_time: int, np: Any) -> Any:
    rows = _read_tsv(path)
    if not rows:
        raise FileNotFoundError(f"Missing or empty ETCO2 file: {path}")
    columns = rows[0].keys()
    if "delta_etco2_mmhg" in columns:
        values = [float(row["delta_etco2_mmhg"]) for row in rows]
    elif "etco2_mmhg" in columns:
        values = [float(row["etco2_mmhg"]) for row in rows]
        baseline = np.nanmedian(values[: max(4, len(values) // 5)])
        values = [float(v - baseline) for v in values]
    else:
        raise ValueError(f"{path} must contain delta_etco2_mmhg or etco2_mmhg")
    return _fit_length(np.asarray(values, dtype=np.float32), n_time, np)


def _load_time_grid(path: Path, n_time: int, config: dict[str, Any], np: Any) -> Any:
    rows = _read_tsv(path)
    if rows and "time" in rows[0]:
        return _fit_length(np.asarray([float(row["time"]) for row in rows], dtype=np.float32), n_time, np)
    tr = float(config.get("dataset", {}).get("tr_seconds", 1.55))
    return np.arange(n_time, dtype=np.float32) * tr


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _fit_length(arr: Any, n_time: int, np: Any) -> Any:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == n_time:
        return arr
    if arr.size > n_time:
        return arr[:n_time]
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, n_time)
    return np.interp(x_new, x_old, arr).astype(np.float32)


def _estimate_real_tcnr(psc: Any, etco2: Any, mask: Any, np: Any) -> Any:
    baseline_n = max(4, psc.shape[3] // 5)
    positive = etco2 > max(1.0, 0.25 * float(np.nanmax(etco2)))
    if not np.any(positive):
        positive = np.arange(psc.shape[3]) >= psc.shape[3] // 2
    delta = np.mean(psc[..., positive], axis=3) - np.mean(psc[..., :baseline_n], axis=3)
    noise = np.std(np.diff(psc, axis=3), axis=3) / np.sqrt(2.0)
    tcnr = np.abs(delta) / np.maximum(noise, 1e-3)
    tcnr = np.nan_to_num(tcnr, nan=0.0, posinf=0.0, neginf=0.0)
    tcnr = np.clip(tcnr, 0.0, 10.0).astype(np.float32)
    tcnr[~np.asarray(mask, dtype=bool)] = 0.0
    return tcnr


def _build_real_feature_volume(
    psc: Any,
    baseline: Any,
    tcnr: Any,
    mask: Any,
    fractions: Any,
    vessel_likelihood: Any,
    boundary_uncertainty: Any,
    etco2: Any,
    np: Any,
) -> Any:
    baseline_n = max(4, psc.shape[3] // 5)
    positive = etco2 > max(1.0, 0.25 * float(np.nanmax(etco2)))
    if not np.any(positive):
        positive = np.arange(psc.shape[3]) >= psc.shape[3] // 2
    psc_mean_baseline = np.mean(psc[..., :baseline_n], axis=3)
    psc_mean_hypercapnia = np.mean(psc[..., positive], axis=3)
    maps = {
        "baseline_bold": baseline,
        "psc_mean_baseline": psc_mean_baseline,
        "psc_mean_hypercapnia": psc_mean_hypercapnia,
        "psc_mean": np.mean(psc, axis=3),
        "psc_std": np.std(psc, axis=3),
        "psc_delta": psc_mean_hypercapnia - psc_mean_baseline,
        "tcnr": tcnr,
        "mask": mask.astype(np.float32),
        "vessel_likelihood": np.clip(vessel_likelihood, 0.0, 1.0).astype(np.float32),
        "boundary_uncertainty": boundary_uncertainty.astype(np.float32),
    }
    for frac, name in zip(fractions, TISSUE_FRACTION_NAMES, strict=True):
        maps[f"fraction_{name}"] = frac.astype(np.float32)
    features = np.stack([maps[name] for name in ON_THE_FLY_FEATURE_NAMES], axis=0).astype(np.float32)
    features[:, ~np.asarray(mask, dtype=bool)] = 0.0
    return features


def _slice_axis(img: Any, axis_name: str, np: Any, nib: Any) -> int:
    axis_name = str(axis_name).lower()
    if axis_name in {"0", "1", "2"}:
        return int(axis_name)
    if axis_name in {"array_z", "z"}:
        return 2
    anatomical = {"axial": {"s", "i"}, "coronal": {"a", "p"}, "sagittal": {"l", "r"}}
    axcodes = tuple(str(code).lower() for code in nib.aff2axcodes(img.affine))
    for axis, code in enumerate(axcodes[:3]):
        if code in anatomical.get(axis_name, set()):
            return axis
    return 2


def _valid_slice_indices(mask: Any, axis: int, np: Any) -> list[int]:
    counts = np.asarray(mask, dtype=bool).sum(axis=tuple(i for i in range(3) if i != axis))
    return [int(v) for v in np.flatnonzero(counts > 0)]


def _put_slice(volume: Any, index: int, axis: int, values: Any, mask: Any) -> None:
    np = __import__("numpy")
    view = [slice(None)] * 3
    view[axis] = int(index)
    target = volume[tuple(view)]
    target[np.asarray(mask, dtype=bool)] = np.asarray(values, dtype=target.dtype)[np.asarray(mask, dtype=bool)]


def _write_slice_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_qc_png(path: Path, cvr: Any, delay: Any, T: Any, sigma: Any, residual: Any, mask: Any, plt: Any, np: Any) -> None:
    if not np.any(mask):
        return
    counts = np.asarray(mask, dtype=bool).sum(axis=(0, 1))
    z = int(np.argmax(counts))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), constrained_layout=True)
    panels = [
        ("CVR", cvr[:, :, z], "viridis"),
        ("delay", delay[:, :, z], "magma"),
        ("T", T[:, :, z], "plasma"),
        ("sigma", sigma[:, :, z], "cividis"),
        ("residual RMS", residual[:, :, z], "inferno"),
    ]
    for ax, (title, data, cmap) in zip(axes, panels, strict=True):
        im = ax.imshow(np.rot90(data), cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(path, dpi=180)
    plt.close(fig)
