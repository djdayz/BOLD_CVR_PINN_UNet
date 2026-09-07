from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any

from hybrid_cvr.config import require_dependency
from hybrid_cvr.segmentation.external_tools import require_tool

WM_LABELS = {2, 41}
SUBCORTICAL_GM_LABELS = {
    10,
    49,
    11,
    50,
    12,
    51,
    13,
    52,
    17,
    53,
    18,
    54,
    26,
    58,
    28,
    60,
}
VCSF_LABELS = {4, 43, 5, 44, 14, 15}


@dataclass(frozen=True)
class FastSurferConfig:
    fastsurfer_home: Path = Path("/Applications/FastSurfer-2.4.2")
    subjects_dir: Path = Path("data/derivatives/fastsurfer")
    fs_license: Path | None = None
    python_cmd: str | None = None
    device: str = "cpu"
    threads: int = 4
    run_surfaces: bool = False
    overwrite: bool = False
    allow_root: bool = False
    skip_cerebellum: bool = True
    skip_hypothalamus: bool = True
    skip_biasfield: bool = True
    dilate_masks: bool = False
    dilate_min_voxels: int = 20
    dilate_iterations: int = 1
    fallback_partial_volume_threshold: float = 0.05
    registration_cost: str = "normmi"
    use_brain_refweight: bool = False


def run_fastsurfer_subject(subject: str, t1w_path: str | Path, config: FastSurferConfig) -> dict[str, Any]:
    script = config.fastsurfer_home / "run_fastsurfer.sh"
    if not script.exists():
        raise FileNotFoundError(f"Missing FastSurfer runner: {script}")
    t1w_path = Path(t1w_path).expanduser().resolve()
    subjects_dir = config.subjects_dir.expanduser().resolve()
    subjects_dir.mkdir(parents=True, exist_ok=True)
    subject_dir = subjects_dir / subject
    aparc = subject_dir / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    if aparc.exists() and not config.overwrite:
        return {"subject": subject, "status": "skipped", "subject_dir": subject_dir, "aparc": aparc}

    cmd = [
        str(script),
        "--sid",
        subject,
        "--sd",
        str(subjects_dir),
        "--t1",
        str(t1w_path),
        "--device",
        config.device,
        "--threads",
        str(config.threads),
    ]
    if not config.run_surfaces:
        cmd.append("--seg_only")
    if config.allow_root:
        cmd.append("--allow_root")
    if config.skip_cerebellum:
        cmd.append("--no_cereb")
    if config.skip_hypothalamus:
        cmd.append("--no_hypothal")
    if config.skip_biasfield:
        cmd.append("--no_biasfield")
    if config.fs_license is not None:
        cmd.extend(["--fs_license", str(config.fs_license.expanduser().resolve())])
    if config.python_cmd:
        cmd.extend(["--py", config.python_cmd])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(config.fastsurfer_home) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "FastSurfer failed for "
            f"{subject} with exit code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if not aparc.exists():
        raise FileNotFoundError(f"FastSurfer finished but did not create expected labelmap: {aparc}")
    return {"subject": subject, "status": "ok", "subject_dir": subject_dir, "aparc": aparc}


def find_fastsurfer_mri_dir(subject: str, subjects_dir: str | Path) -> Path | None:
    mri_dir = Path(subjects_dir).expanduser() / subject / "mri"
    if (mri_dir / "aparc.DKTatlas+aseg.deep.mgz").exists():
        return mri_dir
    return None


def extract_fastsurfer_masks(mri_dir: str | Path, out_dir: str | Path) -> dict[str, Path]:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    mri_dir = Path(mri_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aparc_path = mri_dir / "aparc.DKTatlas+aseg.deep.mgz"
    aseg_path = mri_dir / "aseg.auto_noCCseg.mgz"
    if not aparc_path.exists():
        raise FileNotFoundError(f"Missing FastSurfer aparc/DKT/aseg labelmap: {aparc_path}")

    aparc_img = nib.load(str(aparc_path))
    aparc = np.rint(aparc_img.get_fdata()).astype(np.int32)
    if aseg_path.exists():
        aseg_img = nib.load(str(aseg_path))
        aseg = np.rint(aseg_img.get_fdata()).astype(np.int32)
    else:
        aseg_img = aparc_img
        aseg = aparc

    mask_defs = {
        "wm_mask": (np.isin(aseg, list(WM_LABELS)), aseg_img),
        "subcortical_gm_mask": (np.isin(aseg, list(SUBCORTICAL_GM_LABELS)), aseg_img),
        "vcsf_mask": (np.isin(aseg, list(VCSF_LABELS)), aseg_img),
        "cortical_gm_mask": (((aparc >= 1000) & (aparc < 3000)), aparc_img),
    }
    outputs: dict[str, Path] = {}
    for name, (mask, ref_img) in mask_defs.items():
        path = out_dir / f"{name}.nii.gz"
        _save_uint8_mask(mask, ref_img, path)
        outputs[name] = path
    return outputs


def register_fastsurfer_masks_to_bold(
    native_masks: dict[str, Path],
    t1w_path: str | Path,
    mean_bold_path: str | Path,
    out_dir: str | Path,
    brain_mask_path: str | Path | None = None,
    config: FastSurferConfig | None = None,
) -> dict[str, Path | str]:
    config = config or FastSurferConfig()
    flirt = require_tool("flirt")
    fslcpgeom = require_tool("fslcpgeom")
    fslmaths = require_tool("fslmaths")
    out_dir = Path(out_dir)
    t1_dir = out_dir / "t1w"
    bold_dir = out_dir / "bold"
    t1_dir.mkdir(parents=True, exist_ok=True)
    bold_dir.mkdir(parents=True, exist_ok=True)

    t1w_path = Path(t1w_path).expanduser().resolve()
    mean_bold_path = Path(mean_bold_path).expanduser().resolve()
    mat_path = out_dir / "t1w_to_bold.mat"
    t1w_to_bold = out_dir / "t1w_to_bold.nii.gz"

    reg_cmd = [
        flirt,
        "-in",
        str(t1w_path),
        "-ref",
        str(mean_bold_path),
        "-omat",
        str(mat_path),
        "-out",
        str(t1w_to_bold),
        "-dof",
        "6",
        "-cost",
        config.registration_cost,
    ]
    if config.use_brain_refweight and brain_mask_path is not None and Path(brain_mask_path).exists():
        reg_cmd.extend(["-refweight", str(Path(brain_mask_path).expanduser().resolve())])
    _run_cmd(reg_cmd)

    outputs: dict[str, Path | str] = {
        "t1w_to_bold_mat": mat_path,
        "t1w_to_bold": t1w_to_bold,
    }
    for name, native_path in native_masks.items():
        t1_mask = t1_dir / f"{name}.nii.gz"
        t1_tmp = t1_dir / f"{name}.tmp.nii.gz"
        _run_cmd(
            [
                flirt,
                "-in",
                str(native_path),
                "-ref",
                str(t1w_path),
                "-applyxfm",
                "-usesqform",
                "-interp",
                "nearestneighbour",
                "-out",
                str(t1_tmp),
            ]
        )
        _run_cmd([fslcpgeom, str(t1w_path), str(t1_tmp)])
        _run_cmd([fslmaths, str(t1_tmp), "-thr", "0.5", "-bin", str(t1_mask)])
        t1_tmp.unlink(missing_ok=True)

        bold_mask = bold_dir / f"{name}.nii.gz"
        bold_tmp = bold_dir / f"{name}.tmp.nii.gz"
        _run_cmd(
            [
                flirt,
                "-in",
                str(t1_mask),
                "-ref",
                str(mean_bold_path),
                "-applyxfm",
                "-init",
                str(mat_path),
                "-interp",
                "nearestneighbour",
                "-out",
                str(bold_tmp),
            ]
        )
        _run_cmd([fslmaths, str(bold_tmp), "-thr", "0.5", "-bin", str(bold_mask)])
        bold_tmp.unlink(missing_ok=True)
        if _mask_voxel_count(bold_mask) == 0 and _mask_voxel_count(t1_mask) > 0:
            _run_cmd(
                [
                    flirt,
                    "-in",
                    str(t1_mask),
                    "-ref",
                    str(mean_bold_path),
                    "-applyxfm",
                    "-init",
                    str(mat_path),
                    "-interp",
                    "trilinear",
                    "-out",
                    str(bold_tmp),
                ]
            )
            _run_cmd(
                [
                    fslmaths,
                    str(bold_tmp),
                    "-thr",
                    str(config.fallback_partial_volume_threshold),
                    "-bin",
                    str(bold_mask),
                ]
            )
            bold_tmp.unlink(missing_ok=True)
        if config.dilate_masks and _mask_voxel_count(bold_mask) >= config.dilate_min_voxels:
            _dilate_binary_mask(bold_mask, config.dilate_iterations)
        outputs[name] = bold_mask
        outputs[f"{name}_t1w"] = t1_mask
    return outputs


def save_fastsurfer_bold_segmentation(
    mri_dir: str | Path,
    t1w_path: str | Path,
    mean_bold_path: str | Path,
    brain_mask_path: str | Path,
    out_dir: str | Path,
    config: FastSurferConfig,
) -> dict[str, Path | str]:
    out_dir = Path(out_dir)
    native_masks = extract_fastsurfer_masks(mri_dir, out_dir / "native")
    outputs = register_fastsurfer_masks_to_bold(
        native_masks,
        t1w_path,
        mean_bold_path,
        out_dir,
        brain_mask_path=brain_mask_path,
        config=config,
    )
    outputs["method"] = "fastsurfer_labels_flirt"
    outputs["fastsurfer_mri_dir"] = str(Path(mri_dir))
    return outputs


def _save_uint8_mask(mask: Any, ref_img: Any, out_path: Path) -> None:
    nib = require_dependency("nibabel", "pip install nibabel")
    np = require_dependency("numpy", "pip install numpy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(mask, dtype=np.uint8)
    out = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(out_path))


def _mask_voxel_count(mask_path: Path) -> int:
    nib = require_dependency("nibabel", "pip install nibabel")
    data = nib.load(str(mask_path)).get_fdata()
    return int((data > 0).sum())


def _dilate_binary_mask(mask_path: Path, iterations: int) -> None:
    fslmaths = require_tool("fslmaths")
    cmd = [fslmaths, str(mask_path)]
    for _ in range(max(0, int(iterations))):
        cmd.append("-dilM")
    cmd.extend(["-bin", str(mask_path)])
    _run_cmd(cmd)


def _run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
