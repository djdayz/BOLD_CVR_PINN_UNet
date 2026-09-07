from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hybrid_cvr.models.constraints import ParameterRanges
from hybrid_cvr.models.hybrid_unet_pinn import HybridUNetPINN
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
    dataset_cfg.update(
        {
            "sim_root": Path(sim_root),
            "split": split,
            "samples_per_epoch": 1,
            "slice_mode": "2d",
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
    eval_mask = np.zeros(shape, dtype=bool)
    processed_slices = 0
    slice_rows: list[dict[str, Any]] = []

    slice_axis = dataset._slice_axis(ref_img, support.ndim)
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
        "predicted_CVR": _save_float_nifti(pred_cvr, ref_img, out / "predicted_CVR.nii.gz", nib),
        "predicted_delay": _save_float_nifti(
            pred_delay, ref_img, out / "predicted_delay.nii.gz", nib
        ),
        "predicted_T": _save_float_nifti(pred_T, ref_img, out / "predicted_T.nii.gz", nib),
        "predicted_uncertainty_sigma": _save_float_nifti(
            pred_sigma, ref_img, out / "predicted_uncertainty_sigma.nii.gz", nib
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
        "notes": (
            "Prediction export uses GT parameter maps only as the simulator source. "
            "GT CVR/delay/T values are not model inputs and are not used for checkpoint selection."
        ),
    }
    metadata_path = out / "prediction_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_slice_summary(out / "slice_summary.csv", slice_rows)
    _write_qc_png(out / "predicted_maps_qc.png", pred_cvr, pred_delay, pred_T, eval_mask)
    _write_error_qc_png(
        out / "prediction_error_uncertainty_qc.png",
        maps,
        pred_cvr,
        pred_delay,
        pred_T,
        pred_sigma,
        eval_mask,
        slice_axis,
    )
    return {
        **outputs,
        "metadata": metadata_path,
        "slice_summary": out / "slice_summary.csv",
        "qc_png": out / "predicted_maps_qc.png",
        "error_uncertainty_qc_png": out / "prediction_error_uncertainty_qc.png",
        "processed_slices": processed_slices,
        "device": str(device),
    }


def _merged_config(checkpoint_config: dict[str, Any], cli_config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cli_config)
    for section in ("dataset", "model", "parameter_ranges"):
        merged[section] = {**checkpoint_config.get(section, {}), **cli_config.get(section, {})}
    return merged


def _load_model(payload: dict[str, Any], config: dict[str, Any], in_channels: int):
    model_cfg = config.get("model", {})
    ranges = ParameterRanges(**(payload.get("parameter_ranges") or config.get("parameter_ranges", {})))
    model = HybridUNetPINN(
        in_channels=in_channels,
        base_channels=int(model_cfg.get("base_channels", 32)),
        depth=int(model_cfg.get("depth", 3)),
        norm=str(model_cfg.get("norm", "instance")),
        dropout=float(model_cfg.get("dropout", 0.05)),
        parameter_ranges=ranges,
        T_mode=str(payload.get("T_mode") or model_cfg.get("T_mode", "tissuewise")),
    )
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
    pred_sigma: Any,
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
        ("CVR", maps["CVR"], pred_cvr, np.abs(np.asarray(pred_cvr) - np.asarray(maps["CVR"]))),
        (
            "Delay",
            maps["delay"],
            pred_delay,
            np.abs(np.asarray(pred_delay) - np.asarray(maps["delay"])),
        ),
        ("T", maps["T"], pred_T, np.abs(np.asarray(pred_T) - np.asarray(maps["T"]))),
    ]
    for row_idx, (name, gt, pred, err) in enumerate(rows):
        for col_idx, (title, arr) in enumerate(
            [("GT", gt), ("Pred", pred), ("Abs error", err), ("Sigma", pred_sigma)]
        ):
            ax = axes[row_idx, col_idx]
            shown = _take_slice(arr, z, slice_axis)
            shown_mask = _take_slice(mask, z, slice_axis).astype(bool)
            shown = np.ma.masked_where(~shown_mask, shown)
            cmap = "magma" if title in {"Abs error", "Sigma"} else "viridis"
            im = ax.imshow(np.rot90(shown), cmap=cmap)
            ax.set_title(f"{name} {title}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _take_slice(volume: Any, index: int, axis: int) -> Any:
    import numpy as np

    return np.take(np.asarray(volume), int(index), axis=int(axis))
