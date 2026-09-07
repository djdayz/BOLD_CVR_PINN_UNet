from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

from hybrid_cvr.config import require_dependency
from hybrid_cvr.segmentation.external_tools import require_tool


@dataclass(frozen=True)
class BoldPreprocessConfig:
    tr_seconds: float | None = None
    discard_volumes: int = 0
    baseline_volumes: int = 30
    min_baseline_quantile: float = 0.05
    psc_clip_min: float | None = -100.0
    psc_clip_max: float | None = 200.0
    motion_correction_enabled: bool = False
    motion_correction_tool: str = "fsl_mcflirt"


def percent_signal_change(data: Any, baseline_volumes: int = 30, eps: float = 1e-6) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D BOLD array X,Y,Z,T; got shape {arr.shape}")
    n_base = min(max(1, baseline_volumes), arr.shape[-1])
    baseline = arr[..., :n_base].mean(axis=-1, keepdims=True)
    valid = baseline > eps
    psc = np.zeros_like(arr, dtype=np.float32)
    np.divide(arr - baseline, baseline, out=psc, where=valid)
    psc *= 100.0
    return psc


def simple_epi_mask(mean_bold: Any, quantile: float = 0.2) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    threshold = np.quantile(mean_bold[mean_bold > 0], quantile) if np.any(mean_bold > 0) else 0
    return (mean_bold > threshold).astype("uint8")


def temporal_snr(data: Any, eps: float = 1e-6) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    mean = np.mean(data, axis=-1)
    std = np.std(data, axis=-1)
    return mean / np.maximum(std, eps)


def preprocess_bold_file(
    bold_path: str | Path,
    out_dir: str | Path,
    config: BoldPreprocessConfig | None = None,
) -> dict[str, Path]:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    config = config or BoldPreprocessConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(bold_path)
    motion_outputs: dict[str, Path] = {}
    if config.motion_correction_enabled:
        source_path = _run_mcflirt(source_path, out_dir, config.motion_correction_tool)
        motion_outputs = {
            "motion_corrected_bold": source_path,
            "motion_parameters": Path(str(source_path) + ".par"),
            "motion_matrices": Path(str(source_path) + ".mat"),
        }
    img = nib.load(str(source_path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD NIfTI, got {data.shape}")
    if config.discard_volumes:
        data = data[..., config.discard_volumes :]
    mean_bold = data.mean(axis=-1)
    baseline = data[..., : min(config.baseline_volumes, data.shape[-1])].mean(axis=-1)
    mask = simple_epi_mask(mean_bold)
    psc = percent_signal_change(data, config.baseline_volumes)
    baseline_values = baseline[mask > 0]
    if baseline_values.size:
        baseline_floor = float(np.quantile(baseline_values, config.min_baseline_quantile))
    else:
        baseline_floor = 0.0
    valid_signal_mask = (mask > 0) & (baseline > baseline_floor)
    psc *= valid_signal_mask[..., None].astype(np.float32)
    if config.psc_clip_min is not None or config.psc_clip_max is not None:
        psc = np.clip(
            psc,
            -np.inf if config.psc_clip_min is None else config.psc_clip_min,
            np.inf if config.psc_clip_max is None else config.psc_clip_max,
        ).astype(np.float32)
    std = data.std(axis=-1)
    tsnr = temporal_snr(data)
    outputs = {
        **motion_outputs,
        "bold_psc": out_dir / "bold_psc.nii.gz",
        "mean_bold": out_dir / "mean_bold.nii.gz",
        "baseline_bold": out_dir / "baseline_bold.nii.gz",
        "brain_mask": out_dir / "brain_mask.nii.gz",
        "valid_signal_mask": out_dir / "valid_signal_mask.nii.gz",
        "temporal_std": out_dir / "temporal_std.nii.gz",
        "temporal_snr": out_dir / "temporal_snr.nii.gz",
    }
    for key, value in [
        ("bold_psc", psc),
        ("mean_bold", mean_bold),
        ("baseline_bold", baseline),
        ("brain_mask", mask),
        ("valid_signal_mask", valid_signal_mask.astype("uint8")),
        ("temporal_std", std),
        ("temporal_snr", tsnr),
    ]:
        nib.save(nib.Nifti1Image(value, img.affine, img.header), str(outputs[key]))
    return outputs


def _run_mcflirt(bold_path: Path, out_dir: Path, tool: str) -> Path:
    if tool not in {"fsl_mcflirt", "mcflirt"}:
        raise ValueError(f"Unsupported motion correction tool: {tool}")
    mcflirt = require_tool("mcflirt")
    out_path = out_dir / "bold_mcflirt.nii.gz"
    cmd = [mcflirt, "-in", str(bold_path), "-out", str(out_path), "-plots", "-mats"]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"MCFLIRT failed with exit code {result.returncode}: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if not out_path.exists():
        raise FileNotFoundError(f"MCFLIRT finished but did not create expected output: {out_path}")
    return out_path
