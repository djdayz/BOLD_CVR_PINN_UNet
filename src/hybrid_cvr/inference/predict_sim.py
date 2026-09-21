from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from hybrid_cvr.models.constraints import ParameterRanges
from hybrid_cvr.simulation.mida_bold import load_mida_parameter_case
from hybrid_cvr.training.on_the_fly_dataset import OnTheFlyCVRDataset, OnTheFlyDatasetConfig


def predict_sim_parameter_maps(
    checkpoint: str | Path,
    config: dict[str, Any],
    sim_root: str | Path,
    out_dir: str | Path,
    *,
    split: str = "test",
    case_id: str | None = None,
    paradigm: str = "block",
    tcnr: float = 2.0,
) -> dict[str, Path | int | str]:
    """Export predicted CVR/delay/T maps from a self-supervised checkpoint.

    Ground-truth maps are loaded only to define the simulation source geometry and
    tissue fractions used by the on-the-fly simulator. They are not passed as model
    inputs and no GT metrics are computed here.
    """

    torch = __import__("torch")
    np = __import__("numpy")
    nib = __import__("nibabel")

    checkpoint_path = Path(checkpoint)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config") or {}
    run_config = _merged_config(checkpoint_config, config)
    dataset_cfg = dict(run_config.get("dataset", {}))
    architecture = str(
        run_config.get("model", {}).get("architecture", "cnn1d_unet3d_physiology")
    ).lower()
    temporal_architecture = architecture in {
        "cnn1d_unet3d_physiology",
        "cnn1d_hybrid_3d",  # Compatibility with checkpoints from the active VM run.
    }
    dataset_cfg.update(
        {
            "sim_root": Path(sim_root),
            "split": split,
            "samples_per_epoch": 1,
            "slice_mode": "full_volume" if temporal_architecture else "2d",
            "n_timepoints": int(dataset_cfg.get("n_timepoints", 480)),
            "paradigms": (str(paradigm),),
            "tcnr_levels": (float(tcnr),),
            "temporal_mode": "full",
            "etco2_input": str(dataset_cfg.get("etco2_input", "measured")),
            "sampling_strategy": "balanced_grid",
            "randomize_artifacts_for_training": False,
        }
    )
    dataset = OnTheFlyCVRDataset(OnTheFlyDatasetConfig(**dataset_cfg))
    if case_id is not None:
        case_dir = Path(sim_root) / "mida_parameters" / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Requested parameter case does not exist: {case_dir}")
        dataset.case_dirs = [case_dir]

    case_dir = dataset.case_dirs[0]
    maps, ref_img = load_mida_parameter_case(case_dir)
    support = np.asarray(maps["region_labels"] > 0, dtype=bool)
    input_channels = payload.get("input_channel_names") or list(dataset.feature_names)
    if list(input_channels) != list(dataset.feature_names):
        raise ValueError(
            "Checkpoint input channels do not match current on-the-fly feature order: "
            f"{input_channels} != {list(dataset.feature_names)}"
        )

    model = _load_model(payload, run_config, in_channels=len(input_channels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    shape = support.shape
    pred_cvr = np.zeros(shape, dtype=np.float32)
    pred_delay = np.zeros(shape, dtype=np.float32)
    pred_T = np.zeros(shape, dtype=np.float32)
    pred_sigma = np.zeros(shape, dtype=np.float32)
    pred_sigma_cvr = np.zeros(shape, dtype=np.float32)
    pred_sigma_delay = np.zeros(shape, dtype=np.float32)
    pred_sigma_T = np.zeros(shape, dtype=np.float32)
    eval_mask = np.zeros(shape, dtype=bool)
    processed_slices = 0
    slice_rows: list[dict[str, Any]] = []
    simulation_generation_seconds = 0.0
    model_inference_seconds = 0.0
    etco2_clean_trace = None
    etco2_model_trace = None
    time_trace = None

    slice_axis = dataset._slice_axis(ref_img, support.ndim)
    if temporal_architecture:
        from hybrid_cvr.inference.sliding_window import sliding_window_parameter_inference
        from hybrid_cvr.training.train import _batch_to_device, _materialize_gpu_simulated_batch

        dataset.case_dirs = [case_dir]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        simulation_start = time.perf_counter()
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))
        batch = _batch_to_device(batch, device)
        batch = _materialize_gpu_simulated_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        simulation_generation_seconds = time.perf_counter() - simulation_start
        etco2_clean_trace = batch["etco2_clean"][0].detach().cpu().numpy()
        etco2_model_trace = batch["etco2_for_model"][0].detach().cpu().numpy()
        time_trace = batch["time_grid"].detach().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        with torch.no_grad():
            if bool(run_config.get("model", {}).get("full_volume_inference", False)):
                direct = model(
                    batch["features"],
                    etco2=batch["etco2_for_model"],
                    time_grid=batch["time_grid"],
                    mask=batch["mask"],
                    tissue_maps=None,
                    bold_psc=batch["bold_psc"],
                    valid_time_mask=batch.get("valid_time_mask"),
                )
                prediction = {key: direct[key][0] for key in (
                    "cvr", "delay", "T", "sigma", "sigma_cvr", "sigma_delay", "sigma_T"
                )}
                prediction["coverage"] = batch["mask"][0] > 0.5
            else:
                prediction = sliding_window_parameter_inference(
                    model,
                    batch["features"],
                    batch["bold_psc"],
                    batch["etco2_for_model"],
                    batch["time_grid"],
                    batch["mask"],
                    None,
                    patch_size=tuple(dataset_cfg.get("patch_size", (32, 32, 24))),
                    overlap=0.5,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        model_inference_seconds = time.perf_counter() - inference_start
        pred_cvr = prediction["cvr"].cpu().numpy().astype(np.float32)
        pred_delay = prediction["delay"].cpu().numpy().astype(np.float32)
        pred_T = prediction["T"].cpu().numpy().astype(np.float32)
        pred_sigma = prediction["sigma"].cpu().numpy().astype(np.float32)
        pred_sigma_cvr = prediction["sigma_cvr"].cpu().numpy().astype(np.float32)
        pred_sigma_delay = prediction["sigma_delay"].cpu().numpy().astype(np.float32)
        pred_sigma_T = prediction["sigma_T"].cpu().numpy().astype(np.float32)
        eval_mask = prediction["coverage"].cpu().numpy().astype(bool)
        processed_slices = len(dataset._valid_slice_indices(eval_mask, slice_axis))
        slice_rows.append(
            {
                "z": -1,
                "brain_voxels": int(eval_mask.sum()),
                "cvr_median": float(np.median(pred_cvr[eval_mask])),
                "delay_median": float(np.median(pred_delay[eval_mask])),
                "T_median": float(np.median(pred_T[eval_mask])),
                "sigma_median": float(np.median(pred_sigma[eval_mask])),
            }
        )
    else:
        valid_z = sorted(dataset._valid_slice_indices(support, slice_axis))
        if dataset.config.slice_index_range is not None:
            lo, hi = int(dataset.config.slice_index_range[0]), int(dataset.config.slice_index_range[1])
            if hi < lo:
                lo, hi = hi, lo
            valid_z = [idx for idx in valid_z if lo <= idx <= hi]
        if dataset.config.slice_indices:
            requested = {int(v) for v in dataset.config.slice_indices}
            valid_z = [idx for idx in valid_z if idx in requested]
        with torch.no_grad():
            for z in valid_z:
                dataset.config = OnTheFlyDatasetConfig(**{**dataset.config.__dict__, "slice_indices": (int(z),)})
                sample = dataset[0]
                features = sample["features"].unsqueeze(0).to(device=device, dtype=torch.float32)
                tissue_maps = sample["tissue_maps"].unsqueeze(0).to(device=device, dtype=torch.float32)
                mask = sample["mask"].numpy().astype(bool)
                pred = model(features, tissue_maps=tissue_maps)
                cvr = pred["cvr"][0].detach().cpu().numpy().astype(np.float32)
                delay = pred["delay"][0].detach().cpu().numpy().astype(np.float32)
                T = pred["T"][0].detach().cpu().numpy().astype(np.float32)
                sigma = pred["sigma"][0].detach().cpu().numpy().astype(np.float32)
                _put_slice(pred_cvr, z, slice_axis, cvr, mask)
                _put_slice(pred_delay, z, slice_axis, delay, mask)
                _put_slice(pred_T, z, slice_axis, T, mask)
                _put_slice(pred_sigma, z, slice_axis, sigma, mask)
                _put_slice(eval_mask, z, slice_axis, mask, mask)
                processed_slices += 1
                slice_rows.append(
                    {
                        "z": int(z),
                        "brain_voxels": int(mask.sum()),
                        "cvr_median": float(np.median(cvr[mask])),
                        "delay_median": float(np.median(delay[mask])),
                        "T_median": float(np.median(T[mask])),
                        "sigma_median": float(np.median(sigma[mask])),
                    }
                )

    outputs = {
        "GT_CVR": _save_float_nifti(
            maps["CVR"], ref_img, out / "GT_CVR.nii.gz", nib
        ),
        "GT_delay": _save_float_nifti(
            maps["delay"], ref_img, out / "GT_delay.nii.gz", nib
        ),
        "GT_T": _save_float_nifti(maps["T"], ref_img, out / "GT_T.nii.gz", nib),
        "predicted_CVR": _save_float_nifti(pred_cvr, ref_img, out / "predicted_CVR.nii.gz", nib),
        "predicted_delay": _save_float_nifti(
            pred_delay, ref_img, out / "predicted_delay.nii.gz", nib
        ),
        "predicted_T": _save_float_nifti(pred_T, ref_img, out / "predicted_T.nii.gz", nib),
        "predicted_uncertainty_sigma": _save_float_nifti(
            pred_sigma, ref_img, out / "predicted_uncertainty_sigma.nii.gz", nib
        ),
        "predicted_CVR_uncertainty": _save_float_nifti(
            pred_sigma_cvr, ref_img, out / "predicted_CVR_uncertainty.nii.gz", nib
        ),
        "predicted_delay_uncertainty": _save_float_nifti(
            pred_sigma_delay, ref_img, out / "predicted_delay_uncertainty.nii.gz", nib
        ),
        "predicted_T_uncertainty": _save_float_nifti(
            pred_sigma_T, ref_img, out / "predicted_T_uncertainty.nii.gz", nib
        ),
    }
    gt_error_outputs = _save_gt_error_uncertainty_maps(
        out, ref_img, maps, pred_cvr, pred_delay, pred_T, pred_sigma, eval_mask, nib
    )
    outputs.update(gt_error_outputs)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "case_dir": str(case_dir),
        "split": split,
        "paradigm": str(paradigm),
        "target_tcnr": float(tcnr),
        "processed_slices": int(processed_slices),
        "slice_axis": int(slice_axis),
        "slice_axis_name": str(dataset.config.slice_axis),
        "slice_index_range": list(dataset.config.slice_index_range)
        if dataset.config.slice_index_range is not None
        else None,
        "input_channel_names": list(input_channels),
        "T_mode": payload.get("T_mode"),
        "timing_seconds": {
            "simulation_generation": float(simulation_generation_seconds),
            "model_inference_only": float(model_inference_seconds),
        },
        "notes": (
            "Prediction export uses GT parameter maps only as the simulator source. "
            "GT CVR/delay/T values are not model inputs and are not used for checkpoint selection."
        ),
    }
    metadata_path = out / "prediction_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if time_trace is not None:
        np.savetxt(
            out / "etco2_trace.csv",
            np.column_stack([time_trace, etco2_clean_trace, etco2_model_trace]),
            delimiter=",",
            header="time_seconds,etco2_clean_mmhg,etco2_model_input_mmhg",
            comments="",
        )
    _write_slice_summary(out / "slice_summary.csv", slice_rows)
    _write_qc_png(out / "predicted_maps_qc.png", pred_cvr, pred_delay, pred_T, eval_mask)
    _write_error_qc_png(
        out / "prediction_error_uncertainty_qc.png",
        maps,
        pred_cvr,
        pred_delay,
        pred_T,
        pred_sigma_cvr,
        pred_sigma_delay,
        pred_sigma_T,
        eval_mask,
        slice_axis,
    )
    comparison_qc = out / "gt_vs_predicted_three_plane_qc.png"
    _write_gt_prediction_three_plane_qc(
        comparison_qc, maps, pred_cvr, pred_delay, pred_T, eval_mask, ref_img.affine
    )
    return {
        **outputs,
        "metadata": metadata_path,
        "slice_summary": out / "slice_summary.csv",
        "qc_png": out / "predicted_maps_qc.png",
        "error_uncertainty_qc_png": out / "prediction_error_uncertainty_qc.png",
        "gt_prediction_three_plane_qc_png": comparison_qc,
        "processed_slices": processed_slices,
        "device": str(device),
    }


def _write_gt_prediction_three_plane_qc(
    path: Path,
    maps: dict[str, Any],
    pred_cvr: Any,
    pred_delay: Any,
    pred_T: Any,
    support: Any,
    affine: Any,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import nibabel as nib
        import numpy as np
    except Exception:
        return

    def canonical(volume: Any) -> Any:
        image = nib.Nifti1Image(np.asarray(volume), np.asarray(affine))
        return np.asanyarray(nib.as_closest_canonical(image).dataobj)

    # Canonical RAS guarantees axes 0/1/2 are sagittal/coronal/axial.
    mask = canonical(np.asarray(support, dtype=np.uint8)).astype(bool)
    if not np.any(mask):
        return
    coordinates = np.argwhere(mask)
    centers = tuple(int(np.median(coordinates[:, axis])) for axis in range(3))
    rows = (
        ("CVR", canonical(maps["CVR"]), canonical(pred_cvr)),
        ("Delay", canonical(maps["delay"]), canonical(pred_delay)),
        ("T", canonical(maps["T"]), canonical(pred_T)),
    )
    planes = (
        ("Sagittal", lambda volume: volume[centers[0], :, :]),
        ("Coronal", lambda volume: volume[:, centers[1], :]),
        ("Axial", lambda volume: volume[:, :, centers[2]]),
    )
    fig, axes = plt.subplots(3, 6, figsize=(15, 7.2), constrained_layout=True)
    for row_index, (parameter, gt, prediction) in enumerate(rows):
        values = np.concatenate([gt[mask], prediction[mask]])
        finite = values[np.isfinite(values)]
        vmin, vmax = (0.0, 1.0) if finite.size == 0 else tuple(np.percentile(finite, [1, 99]))
        if parameter == "CVR":
            vmin, vmax = 0.0, 1.0
        elif parameter in {"Delay", "T"}:
            vmin = max(0.0, float(vmin))
        vmax = max(float(vmax), float(vmin) + 1e-6)
        last_image = None
        for plane_index, (plane_name, extract) in enumerate(planes):
            for version_index, (version, volume) in enumerate((("GT", gt), ("Predicted", prediction))):
                column = 2 * plane_index + version_index
                axis = axes[row_index, column]
                image = np.flipud(np.rot90(extract(volume), k=1))
                last_image = axis.imshow(image, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
                axis.set_title(f"{parameter} | {plane_name} {version}", fontsize=10)
                axis.axis("off")
        fig.colorbar(last_image, ax=axes[row_index, :], fraction=0.012, pad=0.006)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _merged_config(checkpoint_config: dict[str, Any], cli_config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cli_config)
    for section in ("dataset", "model", "parameter_ranges"):
        merged[section] = {**checkpoint_config.get(section, {}), **cli_config.get(section, {})}
    return merged


def _load_model(payload: dict[str, Any], config: dict[str, Any], in_channels: int):
    from hybrid_cvr.training.train import _build_model

    effective = dict(config)
    effective["parameter_ranges"] = payload.get("parameter_ranges") or config.get(
        "parameter_ranges", {}
    )
    effective["model"] = dict(config.get("model", {}))
    if payload.get("T_mode"):
        effective["model"].setdefault("T_mode", payload["T_mode"])
    model = _build_model(effective, in_channels)
    model.load_state_dict(payload.get("model_state_dict") or payload["model"])
    return model


def _save_float_nifti(data: Any, ref_img: Any, out_path: Path, nib: Any) -> Path:
    import numpy as np

    arr = np.asarray(data, dtype=np.float32)
    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)
    cal_min, cal_max = _display_calibration(out_path.name, arr)
    header["cal_min"] = cal_min
    header["cal_max"] = cal_max
    img = nib.Nifti1Image(arr, ref_img.affine, header)
    nib.save(img, str(out_path))
    return out_path


def _display_calibration(name: str, data: Any) -> tuple[float, float]:
    import numpy as np

    arr = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(arr)
    nonzero = finite & (arr != 0)
    values = arr[nonzero] if np.any(nonzero) else arr[finite]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [1.0, 99.0])
    if "CVR" in name:
        lo = min(float(lo), 0.0)
        hi = max(float(hi), 0.1)
    elif "delay" in name:
        lo = max(0.0, float(lo))
        hi = min(80.0, max(float(hi), lo + 1.0))
    elif name == "predicted_T.nii.gz" or name.endswith("_T.nii.gz"):
        lo = max(0.0, float(lo))
        hi = min(100.0, max(float(hi), lo + 1.0))
    elif "uncertainty" in name or "residual" in name or "error" in name:
        lo = 0.0
        hi = max(float(np.percentile(values, 99.0)), 1e-3)
    else:
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def _put_slice(volume: Any, index: int, axis: int, values: Any, mask: Any) -> None:
    import numpy as np

    view = [slice(None)] * np.asarray(volume).ndim
    view[axis] = int(index)
    target = volume[tuple(view)]
    target[np.asarray(mask, dtype=bool)] = np.asarray(values)[np.asarray(mask, dtype=bool)]


def _save_gt_error_uncertainty_maps(
    out: Path,
    ref_img: Any,
    maps: dict[str, Any],
    pred_cvr: Any,
    pred_delay: Any,
    pred_T: Any,
    pred_sigma: Any,
    eval_mask: Any,
    nib: Any,
) -> dict[str, Path]:
    import numpy as np

    mask = np.asarray(eval_mask, dtype=bool)
    cvr_error = np.zeros_like(pred_cvr, dtype=np.float32)
    delay_error = np.zeros_like(pred_delay, dtype=np.float32)
    T_error = np.zeros_like(pred_T, dtype=np.float32)
    cvr_error[mask] = np.abs(np.asarray(pred_cvr)[mask] - np.asarray(maps["CVR"])[mask])
    delay_error[mask] = np.abs(np.asarray(pred_delay)[mask] - np.asarray(maps["delay"])[mask])
    T_error[mask] = np.abs(np.asarray(pred_T)[mask] - np.asarray(maps["T"])[mask])
    cvr_percent_error = np.zeros_like(pred_cvr, dtype=np.float32)
    delay_percent_error = np.zeros_like(pred_delay, dtype=np.float32)
    T_percent_error = np.zeros_like(pred_T, dtype=np.float32)
    cvr_percent_error[mask] = 100.0 * cvr_error[mask] / np.maximum(
        np.abs(np.asarray(maps["CVR"])[mask]), 1e-6
    )
    delay_percent_error[mask] = 100.0 * delay_error[mask] / np.maximum(
        np.abs(np.asarray(maps["delay"])[mask]), 1e-6
    )
    T_percent_error[mask] = 100.0 * T_error[mask] / np.maximum(
        np.abs(np.asarray(maps["T"])[mask]), 1e-6
    )
    normalized_error = np.zeros_like(pred_cvr, dtype=np.float32)
    normalized_error[mask] = (
        cvr_error[mask] / 2.2 + delay_error[mask] / 80.0 + T_error[mask] / 100.0
    ) / 3.0
    sigma_norm = np.zeros_like(pred_sigma, dtype=np.float32)
    if np.any(mask):
        vals = np.asarray(pred_sigma, dtype=np.float32)[mask]
        lo, hi = float(np.nanpercentile(vals, 1)), float(np.nanpercentile(vals, 99))
        sigma_norm[mask] = np.clip((vals - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    combined = np.zeros_like(pred_cvr, dtype=np.float32)
    combined[mask] = 0.5 * normalized_error[mask] + 0.5 * sigma_norm[mask]
    return {
        "CVR_abs_error_from_GT": _save_float_nifti(
            cvr_error, ref_img, out / "CVR_abs_error_from_GT.nii.gz", nib
        ),
        "delay_abs_error_from_GT": _save_float_nifti(
            delay_error, ref_img, out / "delay_abs_error_from_GT.nii.gz", nib
        ),
        "T_abs_error_from_GT": _save_float_nifti(
            T_error, ref_img, out / "T_abs_error_from_GT.nii.gz", nib
        ),
        "CVR_percent_error_from_GT": _save_float_nifti(
            cvr_percent_error, ref_img, out / "CVR_percent_error_from_GT.nii.gz", nib
        ),
        "delay_percent_error_from_GT": _save_float_nifti(
            delay_percent_error, ref_img, out / "delay_percent_error_from_GT.nii.gz", nib
        ),
        "T_percent_error_from_GT": _save_float_nifti(
            T_percent_error, ref_img, out / "T_percent_error_from_GT.nii.gz", nib
        ),
        "parameter_error_uncertainty_from_GT": _save_float_nifti(
            normalized_error, ref_img, out / "parameter_error_uncertainty_from_GT.nii.gz", nib
        ),
        "combined_predicted_uncertainty_and_GT_error": _save_float_nifti(
            combined, ref_img, out / "combined_predicted_uncertainty_and_GT_error.nii.gz", nib
        ),
    }


def _write_slice_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_qc_png(path: Path, cvr: Any, delay: Any, T: Any, support: Any) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    z_counts = np.asarray(support, dtype=bool).sum(axis=(0, 1))
    if not np.any(z_counts):
        return
    z = int(np.flatnonzero(z_counts > 0)[len(np.flatnonzero(z_counts > 0)) // 2])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    for ax, arr, title in zip(axes, (cvr, delay, T), ("CVR", "Delay", "T"), strict=True):
        shown = np.asarray(arr[:, :, z], dtype=np.float32)
        shown = np.ma.masked_where(~support[:, :, z], shown)
        im = ax.imshow(np.rot90(shown), cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_error_qc_png(
    path: Path,
    maps: dict[str, Any],
    pred_cvr: Any,
    pred_delay: Any,
    pred_T: Any,
    pred_sigma_cvr: Any,
    pred_sigma_delay: Any,
    pred_sigma_T: Any,
    eval_mask: Any,
    slice_axis: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    mask = np.asarray(eval_mask, dtype=bool)
    counts = mask.sum(axis=tuple(i for i in range(mask.ndim) if i != slice_axis))
    valid = np.flatnonzero(counts > 0)
    if valid.size == 0:
        return
    z = int(valid[len(valid) // 2])
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), constrained_layout=True)
    rows = [
        ("CVR", maps["CVR"], pred_cvr, pred_sigma_cvr),
        (
            "Delay",
            maps["delay"],
            pred_delay,
            pred_sigma_delay,
        ),
        ("T", maps["T"], pred_T, pred_sigma_T),
    ]
    for row_idx, (name, gt, pred, sigma) in enumerate(rows):
        gt_array = np.asarray(gt, dtype=np.float32)
        pred_array = np.asarray(pred, dtype=np.float32)
        percent_error = 100.0 * np.abs(pred_array - gt_array) / np.maximum(np.abs(gt_array), 1e-6)
        percent_uncertainty = 100.0 * np.asarray(sigma, dtype=np.float32) / np.maximum(
            np.abs(pred_array), 1e-6
        )
        for col_idx, (title, arr) in enumerate(
            [
                ("GT", gt_array),
                ("Pred", pred_array),
                ("Error (%)", percent_error),
                ("Uncertainty (%)", percent_uncertainty),
            ]
        ):
            ax = axes[row_idx, col_idx]
            shown = _take_slice(arr, z, slice_axis)
            shown_mask = _take_slice(mask, z, slice_axis).astype(bool)
            shown = np.ma.masked_where(~shown_mask, shown)
            cmap = "magma" if "%" in title else "viridis"
            vmax = None
            if "%" in title:
                finite = np.asarray(shown.compressed())
                vmax = max(float(np.percentile(finite, 99)), 1.0) if finite.size else 1.0
            im = ax.imshow(np.flipud(np.rot90(shown)), cmap=cmap, vmin=0.0 if "%" in title else None, vmax=vmax)
            ax.set_title(f"{name} {title}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _take_slice(volume: Any, index: int, axis: int) -> Any:
    import numpy as np

    return np.take(np.asarray(volume), int(index), axis=int(axis))
