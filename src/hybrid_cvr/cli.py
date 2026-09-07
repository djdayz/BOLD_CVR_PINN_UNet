from __future__ import annotations

from pathlib import Path
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

from hybrid_cvr.config import load_yaml
from hybrid_cvr.io.bids import read_manifest, scan_bids_like, write_manifest

try:
    import typer
    from rich.console import Console
except ImportError:  # pragma: no cover - exercised before installation only
    typer = None
    Console = None


if typer is not None:
    app = typer.Typer(help="Hybrid CVR U-Net/PINN research pipeline.")
    console = Console()
else:  # pragma: no cover
    app = None
    console = None


def _write_status(out: str | Path, message: str) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    status = out / "STATUS.txt"
    status.write_text(message + "\n", encoding="utf-8")
    return status


def _source_time_offset_seconds(
    source: str,
    subject: str,
    session: str,
    bold_path: str | Path,
    gas_path: str | Path,
    scan_start_times: dict[tuple[str, str], float],
    fixed_offset: float,
    bold_acquisition_offset_seconds,
    scan_start_or_bold_acquisition_offset_seconds,
    scan_start_offset_seconds,
) -> float:
    if source in {"bold_acquisition_time", "bold_json", "acquisition_time"}:
        return float(bold_acquisition_offset_seconds(bold_path, gas_path))
    if source in {
        "subject_database_or_bold_acquisition_time",
        "database_or_bold",
        "spreadsheet_or_bold",
    }:
        return float(
            scan_start_or_bold_acquisition_offset_seconds(
                subject, session, bold_path, gas_path, scan_start_times
            )
        )
    if source in {"subject_database", "database", "spreadsheet"}:
        return float(scan_start_offset_seconds(subject, session, gas_path, scan_start_times))
    if source in {"fixed", "manual"}:
        return float(fixed_offset)
    if source in {"none", "off", "false"}:
        return 0.0
    raise ValueError(f"Unknown gas source_time_offset_source: {source}")


def _bold_paths_by_subject(records) -> dict[str, dict[str, str | Path]]:
    paths: dict[str, dict[str, str | Path]] = {}
    for record in records:
        paths.setdefault(record.subject, {})[record.session] = record.bold_path
    return paths


def _fastsurfer_config(cfg: dict) -> object:
    from hybrid_cvr.segmentation.fastsurfer import FastSurferConfig

    fs_cfg = cfg.get("fastsurfer", {})
    fs_license = fs_cfg.get("fs_license")
    python_cmd = fs_cfg.get("python_cmd")
    if python_cmd and ("/" in str(python_cmd)):
        python_path = Path(python_cmd).expanduser()
        if not python_path.is_absolute():
            python_path = Path.cwd() / python_path
        python_cmd = str(python_path)
    return FastSurferConfig(
        fastsurfer_home=Path(fs_cfg.get("fastsurfer_home", "/Applications/FastSurfer-2.4.2")),
        subjects_dir=Path(fs_cfg.get("subjects_dir", "data/derivatives/fastsurfer")),
        fs_license=Path(fs_license) if fs_license else None,
        python_cmd=python_cmd,
        device=str(fs_cfg.get("device", "cpu")),
        threads=int(fs_cfg.get("threads", 4)),
        run_surfaces=bool(fs_cfg.get("run_surfaces", False)),
        overwrite=bool(fs_cfg.get("overwrite", False)),
        allow_root=bool(fs_cfg.get("allow_root", False)),
        skip_cerebellum=bool(fs_cfg.get("skip_cerebellum", True)),
        skip_hypothalamus=bool(fs_cfg.get("skip_hypothalamus", True)),
        skip_biasfield=bool(fs_cfg.get("skip_biasfield", True)),
        dilate_masks=bool(fs_cfg.get("dilate_masks", False)),
        dilate_min_voxels=int(fs_cfg.get("dilate_min_voxels", 20)),
        dilate_iterations=int(fs_cfg.get("dilate_iterations", 1)),
        fallback_partial_volume_threshold=float(fs_cfg.get("fallback_partial_volume_threshold", 0.05)),
        registration_cost=str(fs_cfg.get("registration_cost", "normmi")),
        use_brain_refweight=bool(fs_cfg.get("use_brain_refweight", False)),
    )


def _mida_segmentation_config(cfg: dict) -> object:
    from hybrid_cvr.simulation.mida import MidaSegmentationConfig

    mida_cfg = cfg.get("mida", {})
    labels = mida_cfg.get("labels", mida_cfg.get("tissue_label_map", {}))
    return MidaSegmentationConfig(
        label_map_path=Path(mida_cfg.get("label_map_path", "data/raw/MIDA_v1.nii")),
        output_dir=Path(mida_cfg.get("output_dir", "data/derivatives/mida")),
        target_shape=tuple(int(v) for v in mida_cfg.get("target_shape", [94, 94, 50])),
        downsample_factor=tuple(int(v) for v in mida_cfg.get("downsample_factor", [5, 5, 5])),
        target_voxel_size_mm=tuple(float(v) for v in mida_cfg.get("target_voxel_size_mm", [2.5, 2.5, 2.5])),
        wm_labels=tuple(int(v) for v in labels.get("wm", [9, 12])),
        cortical_gm_labels=tuple(int(v) for v in labels.get("cortical_gm", [2, 10])),
        subcortical_gm_labels=tuple(
            int(v) for v in labels.get("subcortical_gm", [4, 5, 7, 8, 16, 17, 20, 21, 99, 116])
        ),
        vcsf_labels=tuple(int(v) for v in labels.get("vcsf", [6])),
        vessel_like_labels=tuple(int(v) for v in labels.get("vessel_like", labels.get("vessel", [24, 25]))),
        vessel_brain_margin_mm=float(mida_cfg.get("vessel_brain_margin_mm", 3.0)),
        fill_sigma_mm=float(mida_cfg.get("fill_sigma_mm", 1.0)),
        brain_support_fraction_threshold=float(mida_cfg.get("brain_support_fraction_threshold", 0.05)),
        save_fsleyes_canonical=bool(mida_cfg.get("save_fsleyes_canonical", True)),
    )


def _mida_parameter_config(cfg: dict) -> object:
    from hybrid_cvr.simulation.mida import MidaParameterConfig

    map_cfg = cfg.get("mida_parameter_maps", {})
    save_highres = map_cfg.get("save_highres_case_indices", [])
    return MidaParameterConfig(
        distribution_path=Path(
            map_cfg.get("distribution_path", "data/distributions/tissue_parameter_samples.parquet")
        ),
        mida_dir=Path(map_cfg.get("mida_dir", "data/derivatives/mida/2p5mm")),
        highres_dir=Path(map_cfg.get("highres_dir", "data/derivatives/mida/highres")),
        output_dir=Path(map_cfg.get("output_dir", "data/simulated/mida_parameters")),
        n_cases=int(map_cfg.get("n_cases", 1)),
        seed=int(map_cfg.get("seed", 17)),
        component_fraction_min=float(map_cfg.get("component_fraction_min", 0.005)),
        parameter_sampling_mode=str(map_cfg.get("parameter_sampling_mode", "subvoxel_monte_carlo")),
        sample_quantile_min=float(map_cfg.get("sample_quantile_min", 0.0)),
        sample_quantile_max=float(map_cfg.get("sample_quantile_max", 1.0)),
        subvoxel_samples_per_lowres_voxel=int(
            map_cfg.get("subvoxel_samples_per_lowres_voxel", 125)
        ),
        parameter_patch_size_vox=tuple(int(v) for v in map_cfg.get("parameter_patch_size_vox", [1, 1, 1])),
        delay_sampling_mode=str(map_cfg.get("delay_sampling_mode", "same_as_parameter")),
        delay_sample_quantile_min=float(map_cfg.get("delay_sample_quantile_min", 0.0)),
        delay_sample_quantile_max=float(map_cfg.get("delay_sample_quantile_max", 1.0)),
        T_sampling_mode=str(map_cfg.get("T_sampling_mode", "same_as_parameter")),
        T_sample_quantile_min=float(map_cfg.get("T_sample_quantile_min", 0.0)),
        T_sample_quantile_max=float(map_cfg.get("T_sample_quantile_max", 1.0)),
        parameter_post_smooth_sigma_vox=float(map_cfg.get("parameter_post_smooth_sigma_vox", 0.0)),
        parameter_post_smooth_blend=float(map_cfg.get("parameter_post_smooth_blend", 0.0)),
        spatial_smoothing_sigma_vox=float(map_cfg.get("spatial_smoothing_sigma_vox", 5.0)),
        rank_jitter=float(map_cfg.get("rank_jitter", 0.0)),
        within_tissue_variation_scale=float(map_cfg.get("within_tissue_variation_scale", 1.0)),
        save_highres_case_indices=tuple(int(v) for v in save_highres),
        save_fsleyes_canonical=bool(map_cfg.get("save_fsleyes_canonical", True)),
    )


def _mida_bold_simulation_config(cfg: dict, out: Path) -> object:
    from hybrid_cvr.simulation.mida_bold import MidaBoldSimulationConfig

    sim_cfg = cfg.get("mida_bold_simulation", {})
    max_cases = sim_cfg.get("max_cases")
    return MidaBoldSimulationConfig(
        parameter_case_dir=Path(
            sim_cfg.get("parameter_case_dir", "data/simulated/mida_parameters/case_000")
        ),
        output_dir=Path(out),
        output_mode=str(sim_cfg.get("output_mode", "volume4d")),
        tcnr_levels=tuple(float(v) for v in sim_cfg.get("tcnr_levels", cfg.get("noise_levels_tcnr", [0.5, 1.0, 2.0, 5.0]))),
        paradigms=tuple(str(v) for v in sim_cfg.get("paradigms", ["block", "multi_step"])),
        slice_indices=tuple(int(v) for v in sim_cfg.get("slice_indices", [])),
        n_timepoints=int(sim_cfg.get("n_timepoints", 480)),
        tr_seconds=float(sim_cfg.get("tr_seconds", 1.55)),
        seed=int(sim_cfg.get("seed", 17)),
        baseline_etco2_mmhg=float(sim_cfg.get("baseline_etco2_mmhg", 40.0)),
        etco2_noise_sd_mmhg=float(sim_cfg.get("etco2_noise_sd_mmhg", 0.25)),
        etco2_drift_sd_mmhg=float(sim_cfg.get("etco2_drift_sd_mmhg", 0.15)),
        save_bold_psc_nifti=bool(sim_cfg.get("save_bold_psc_nifti", True)),
        save_bold_intensity_nifti=bool(sim_cfg.get("save_bold_intensity_nifti", True)),
        save_slice_npz=bool(sim_cfg.get("save_slice_npz", False)),
        save_gt_niftis=bool(sim_cfg.get("save_gt_niftis", False)),
        volume_chunk_voxels=int(sim_cfg.get("volume_chunk_voxels", 50000)),
        max_cases=int(max_cases) if max_cases is not None else None,
        case_start_index=int(sim_cfg.get("case_start_index", 0)),
        append_summary=bool(sim_cfg.get("append_summary", False)),
        ar1_rho=float(sim_cfg.get("ar1_rho", 0.35)),
        drift_fraction_of_noise=float(sim_cfg.get("drift_fraction_of_noise", 0.15)),
        motion_spike_probability=float(sim_cfg.get("motion_spike_probability", 0.015)),
        motion_spike_scale=float(sim_cfg.get("motion_spike_scale", 3.0)),
        psc_spatial_smoothing_sigma_vox=float(sim_cfg.get("psc_spatial_smoothing_sigma_vox", 0.35)),
        baseline_intensity=float(sim_cfg.get("baseline_intensity", 1000.0)),
        baseline_bias_sd=float(sim_cfg.get("baseline_bias_sd", 60.0)),
    )


if typer is not None:

    @app.command("scan-data")
    def scan_data(
        bids_root: Path = typer.Option(..., "--bids-root"),
        out: Path = typer.Option(..., "--out"),
    ) -> None:
        records = scan_bids_like(bids_root)
        write_manifest(records, out)
        console.print(f"Wrote {len(records)} records to {out}")

    @app.command("preprocess-real")
    def preprocess_real(
        config: Path = typer.Option(..., "--config"),
        manifest: Path = typer.Option(..., "--manifest"),
    ) -> None:
        cfg = load_yaml(config)
        if not manifest.exists():
            raise typer.BadParameter(f"Manifest does not exist: {manifest}")
        np = __import__("numpy")
        nib = __import__("nibabel")
        from hybrid_cvr.preprocessing.bold import BoldPreprocessConfig, preprocess_bold_file
        from hybrid_cvr.preprocessing.gas import GasConfig, process_gas_trace, write_gas_outputs
        from hybrid_cvr.preprocessing.qc import save_trace_qc
        from hybrid_cvr.preprocessing.subject_database import (
            bold_acquisition_offset_seconds,
            load_scan_start_times,
            load_scan_start_times_by_participants,
            scan_start_or_bold_acquisition_offset_seconds,
            scan_start_offset_seconds,
        )

        bold_cfg_raw = cfg.get("bold", {})
        motion_cfg_raw = bold_cfg_raw.get("motion_correction", {})
        bold_cfg = BoldPreprocessConfig(
            tr_seconds=bold_cfg_raw.get("tr_seconds"),
            discard_volumes=int(bold_cfg_raw.get("discard_volumes", 0)),
            baseline_volumes=int(bold_cfg_raw.get("baseline_volumes", 30)),
            min_baseline_quantile=float(bold_cfg_raw.get("min_baseline_quantile", 0.05)),
            psc_clip_min=bold_cfg_raw.get("psc_clip_min", -100.0),
            psc_clip_max=bold_cfg_raw.get("psc_clip_max", 200.0),
            motion_correction_enabled=bool(motion_cfg_raw.get("enabled", False)),
            motion_correction_tool=str(motion_cfg_raw.get("tool", "fsl_mcflirt")),
        )
        gas_cfg_raw = cfg.get("gas", {})
        peak_cfg = gas_cfg_raw.get("peak_detection", {})
        gas_cfg = GasConfig(
            time_column=gas_cfg_raw.get("time_column"),
            co2_column=gas_cfg_raw.get("co2_column"),
            sampling_rate_hz=float(gas_cfg_raw.get("sampling_rate_hz", 20)),
            co2_units=str(gas_cfg_raw.get("co2_units", "mmHg")),
            min_distance_seconds=float(peak_cfg.get("min_distance_seconds", 1.0)),
            prominence=peak_cfg.get("prominence"),
            smoothing_seconds=float(gas_cfg_raw.get("smoothing_seconds", 2.0)),
            baseline_first_seconds=float(gas_cfg_raw.get("baseline_first_seconds", 45.0)),
            baseline_reference=str(gas_cfg_raw.get("baseline_reference", "first_peak")),
            baseline_statistic=str(gas_cfg_raw.get("baseline_statistic", "mean")),
        )
        records = list(read_manifest(manifest))
        bold_paths_by_subject = _bold_paths_by_subject(records)
        scan_start_times = {}
        if gas_cfg_raw.get("subject_database"):
            if gas_cfg_raw.get("participants_path"):
                scan_start_times = load_scan_start_times_by_participants(
                    gas_cfg_raw["subject_database"],
                    gas_cfg_raw["participants_path"],
                    bold_paths_by_subject,
                )
            else:
                scan_start_times = load_scan_start_times(gas_cfg_raw["subject_database"])
        apply_scan_start_offset = bool(gas_cfg_raw.get("apply_scan_start_offset", bool(scan_start_times)))
        source_time_offset_source = str(
            gas_cfg_raw.get("source_time_offset_source", "subject_database")
        ).lower()
        gas_rows = []
        bold_rows = []
        for record in records:
            console.print(f"Preprocessing {record.subject}/{record.session}")
            base_row = {"subject": record.subject, "session": record.session, "status": "ok", "error": ""}
            try:
                bold_img = nib.load(record.bold_path)
            except Exception as exc:
                gas_rows.append(
                    {
                        **base_row,
                        "status": "failed",
                        "error": f"Could not read BOLD NIfTI header: {exc}",
                        "n_timepoints": "",
                        "tr_seconds": "",
                        "n_peaks": "",
                        "source_time_offset_seconds": "",
                        "baseline_mmhg": "",
                        "delta_min_mmhg": "",
                        "delta_max_mmhg": "",
                        "etco2_resampled": "",
                    }
                )
                bold_rows.append(
                    {
                        **base_row,
                        "status": "failed",
                        "error": f"Could not read BOLD NIfTI header: {exc}",
                        "motion_corrected_bold": "",
                        "motion_parameters": "",
                        "motion_matrices": "",
                        "bold_psc": "",
                        "mean_bold": "",
                        "baseline_bold": "",
                        "brain_mask": "",
                        "valid_signal_mask": "",
                        "temporal_std": "",
                        "temporal_snr": "",
                    }
                )
                continue
            n_time = int(bold_img.shape[-1])
            bold_json = Path(record.bold_path).with_suffix("").with_suffix(".json")
            tr = cfg.get("bold", {}).get("tr_seconds")
            if tr is None and bold_json.exists():
                tr = json.loads(bold_json.read_text(encoding="utf-8")).get("RepetitionTime")
            if tr is None:
                raise typer.BadParameter(f"Missing TR for {record.bold_path}; set bold.tr_seconds")
            time_grid = np.arange(n_time, dtype=float) * float(tr)
            out_dir = Path(record.output_dir)
            source_time_offset = (
                _source_time_offset_seconds(
                    source_time_offset_source,
                    record.subject,
                    record.session,
                    record.bold_path,
                    record.gas_path,
                    scan_start_times,
                    float(gas_cfg_raw.get("source_time_offset_seconds", 0.0)),
                    bold_acquisition_offset_seconds,
                    scan_start_or_bold_acquisition_offset_seconds,
                    scan_start_offset_seconds,
                )
                if apply_scan_start_offset
                else float(gas_cfg_raw.get("source_time_offset_seconds", 0.0))
            )
            result = process_gas_trace(record.gas_path, time_grid, gas_cfg, source_time_offset)
            paths = write_gas_outputs(result, out_dir)
            if gas_cfg_raw.get("generate_qc_plots", False):
                try:
                    save_trace_qc(
                        result["time"],
                        result["delta_etco2_mmhg"],
                        out_dir / "gas_qc.png",
                        title=f"{record.subject} {record.session} delta ETCO2",
                    )
                except Exception as exc:  # pragma: no cover - plotting is noncritical
                    console.print(f"[yellow]Skipped gas QC plot for {record.subject}/{record.session}: {exc}[/yellow]")
            gas_rows.append(
                {
                    **base_row,
                    "n_timepoints": n_time,
                    "tr_seconds": float(tr),
                    "n_peaks": len(result["peak_time"]),
                    "source_time_offset_seconds": result["source_time_offset_seconds"],
                    "baseline_mmhg": result["baseline_mmhg"],
                    "delta_min_mmhg": float(np.min(result["delta_etco2_mmhg"])),
                    "delta_max_mmhg": float(np.max(result["delta_etco2_mmhg"])),
                    "etco2_resampled": str(paths["etco2_resampled"]),
                }
            )
            if bold_cfg_raw.get("generate_outputs", True):
                try:
                    bold_paths = preprocess_bold_file(record.bold_path, out_dir, bold_cfg)
                    bold_rows.append(
                        {
                            **base_row,
                            "motion_corrected_bold": str(bold_paths.get("motion_corrected_bold", "")),
                            "motion_parameters": str(bold_paths.get("motion_parameters", "")),
                            "motion_matrices": str(bold_paths.get("motion_matrices", "")),
                            "bold_psc": str(bold_paths["bold_psc"]),
                            "mean_bold": str(bold_paths["mean_bold"]),
                            "baseline_bold": str(bold_paths["baseline_bold"]),
                            "brain_mask": str(bold_paths["brain_mask"]),
                            "valid_signal_mask": str(bold_paths["valid_signal_mask"]),
                            "temporal_std": str(bold_paths["temporal_std"]),
                            "temporal_snr": str(bold_paths["temporal_snr"]),
                        }
                    )
                except Exception as exc:
                    bold_rows.append(
                        {
                            **base_row,
                            "status": "failed",
                            "error": f"Could not preprocess BOLD: {exc}",
                            "motion_corrected_bold": "",
                            "motion_parameters": "",
                            "motion_matrices": "",
                            "bold_psc": "",
                            "mean_bold": "",
                            "baseline_bold": "",
                            "brain_mask": "",
                            "valid_signal_mask": "",
                            "temporal_std": "",
                            "temporal_snr": "",
                        }
                    )
        summary = Path("data/processed/gas_summary.csv")
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(gas_rows[0].keys()) if gas_rows else [])
            writer.writeheader()
            writer.writerows(gas_rows)
        if bold_rows:
            bold_summary = Path("data/processed/bold_summary.csv")
            with bold_summary.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(bold_rows[0].keys()))
                writer.writeheader()
                writer.writerows(bold_rows)
            console.print(f"Processed BOLD for {len(bold_rows)} sessions; wrote {bold_summary}")
        console.print(f"Processed gas traces for {len(gas_rows)} sessions; wrote {summary}")

    @app.command("preprocess-gas")
    def preprocess_gas(
        config: Path = typer.Option(..., "--config"),
        manifest: Path = typer.Option(..., "--manifest"),
        processed_root: Path = typer.Option(Path("data/processed"), "--processed-root"),
    ) -> None:
        cfg = load_yaml(config)
        if not manifest.exists():
            raise typer.BadParameter(f"Manifest does not exist: {manifest}")
        np = __import__("numpy")
        nib = __import__("nibabel")
        from hybrid_cvr.preprocessing.gas import GasConfig, process_gas_trace, write_gas_outputs
        from hybrid_cvr.preprocessing.subject_database import (
            bold_acquisition_offset_seconds,
            load_scan_start_times,
            load_scan_start_times_by_participants,
            scan_start_or_bold_acquisition_offset_seconds,
            scan_start_offset_seconds,
        )

        gas_cfg_raw = cfg.get("gas", {})
        peak_cfg = gas_cfg_raw.get("peak_detection", {})
        gas_cfg = GasConfig(
            time_column=gas_cfg_raw.get("time_column"),
            co2_column=gas_cfg_raw.get("co2_column"),
            sampling_rate_hz=float(gas_cfg_raw.get("sampling_rate_hz", 20)),
            co2_units=str(gas_cfg_raw.get("co2_units", "mmHg")),
            min_distance_seconds=float(peak_cfg.get("min_distance_seconds", 1.0)),
            prominence=peak_cfg.get("prominence"),
            smoothing_seconds=float(gas_cfg_raw.get("smoothing_seconds", 2.0)),
            baseline_first_seconds=float(gas_cfg_raw.get("baseline_first_seconds", 45.0)),
            baseline_reference=str(gas_cfg_raw.get("baseline_reference", "first_peak")),
            baseline_statistic=str(gas_cfg_raw.get("baseline_statistic", "mean")),
        )
        records = list(read_manifest(manifest))
        bold_paths_by_subject = _bold_paths_by_subject(records)
        scan_start_times = {}
        if gas_cfg_raw.get("subject_database"):
            if gas_cfg_raw.get("participants_path"):
                scan_start_times = load_scan_start_times_by_participants(
                    gas_cfg_raw["subject_database"],
                    gas_cfg_raw["participants_path"],
                    bold_paths_by_subject,
                )
            else:
                scan_start_times = load_scan_start_times(gas_cfg_raw["subject_database"])
        apply_scan_start_offset = bool(gas_cfg_raw.get("apply_scan_start_offset", bool(scan_start_times)))
        source_time_offset_source = str(
            gas_cfg_raw.get("source_time_offset_source", "subject_database")
        ).lower()
        rows = []
        for record in records:
            console.print(f"Preprocessing gas {record.subject}/{record.session}")
            base_row = {"subject": record.subject, "session": record.session, "status": "ok", "error": ""}
            try:
                bold_img = nib.load(record.bold_path)
                n_time = int(bold_img.shape[-1])
                bold_json = Path(record.bold_path).with_suffix("").with_suffix(".json")
                tr = cfg.get("bold", {}).get("tr_seconds")
                if tr is None and bold_json.exists():
                    tr = json.loads(bold_json.read_text(encoding="utf-8")).get("RepetitionTime")
                if tr is None:
                    raise ValueError(f"Missing TR for {record.bold_path}; set bold.tr_seconds")
                time_grid = np.arange(n_time, dtype=float) * float(tr)
                source_time_offset = (
                    _source_time_offset_seconds(
                        source_time_offset_source,
                        record.subject,
                        record.session,
                        record.bold_path,
                        record.gas_path,
                        scan_start_times,
                        float(gas_cfg_raw.get("source_time_offset_seconds", 0.0)),
                        bold_acquisition_offset_seconds,
                        scan_start_or_bold_acquisition_offset_seconds,
                        scan_start_offset_seconds,
                    )
                    if apply_scan_start_offset
                    else float(gas_cfg_raw.get("source_time_offset_seconds", 0.0))
                )
                result = process_gas_trace(record.gas_path, time_grid, gas_cfg, source_time_offset)
                paths = write_gas_outputs(result, Path(record.output_dir))
                rows.append(
                    {
                        **base_row,
                        "n_timepoints": n_time,
                        "tr_seconds": float(tr),
                        "n_peaks": len(result["peak_time"]),
                        "source_time_offset_seconds": result["source_time_offset_seconds"],
                        "baseline_mmhg": result["baseline_mmhg"],
                        "delta_min_mmhg": float(np.min(result["delta_etco2_mmhg"])),
                        "delta_max_mmhg": float(np.max(result["delta_etco2_mmhg"])),
                        "etco2_resampled": str(paths["etco2_resampled"]),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        **base_row,
                        "status": "failed",
                        "error": str(exc),
                        "n_timepoints": "",
                        "tr_seconds": "",
                        "n_peaks": "",
                        "source_time_offset_seconds": "",
                        "baseline_mmhg": "",
                        "delta_min_mmhg": "",
                        "delta_max_mmhg": "",
                        "etco2_resampled": "",
                    }
                )
        summary = processed_root / "gas_summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"Processed gas traces for {len(rows)} sessions; wrote {summary}")

    @app.command("run-fastsurfer")
    def run_fastsurfer(
        config: Path = typer.Option(..., "--config"),
        manifest: Path = typer.Option(..., "--manifest"),
        out: Path = typer.Option(Path("data/derivatives/fastsurfer/fastsurfer_summary.csv"), "--out"),
    ) -> None:
        cfg = load_yaml(config)
        if not manifest.exists():
            raise typer.BadParameter(f"Manifest does not exist: {manifest}")
        from hybrid_cvr.segmentation.fastsurfer import run_fastsurfer_subject

        fs_config = _fastsurfer_config(cfg)
        records_by_subject = {}
        for record in read_manifest(manifest):
            records_by_subject.setdefault(record.subject, record)
        rows = []
        for subject, record in sorted(records_by_subject.items()):
            console.print(f"Running FastSurfer {subject}")
            base_row = {"subject": subject, "t1w_path": record.t1w_path, "status": "ok", "error": ""}
            try:
                result = run_fastsurfer_subject(subject, record.t1w_path, fs_config)
                rows.append(
                    {
                        **base_row,
                        "status": result["status"],
                        "subject_dir": str(result["subject_dir"]),
                        "aparc": str(result["aparc"]),
                    }
                )
            except Exception as exc:
                rows.append({**base_row, "status": "failed", "error": str(exc), "subject_dir": "", "aparc": ""})
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"Wrote FastSurfer summary for {len(rows)} subjects to {out}")

    @app.command("segment-real")
    def segment_real(
        config: Path = typer.Option(..., "--config"),
        manifest: Path = typer.Option(..., "--manifest"),
    ) -> None:
        cfg = load_yaml(config)
        if not manifest.exists():
            raise typer.BadParameter(f"Manifest does not exist: {manifest}")
        from hybrid_cvr.segmentation.anatomical import save_fallback_segmentation
        from hybrid_cvr.segmentation.fastsurfer import find_fastsurfer_mri_dir, save_fastsurfer_bold_segmentation

        seg_cfg = cfg.get("segmentation", {})
        fs_config = _fastsurfer_config(cfg)
        out_root = Path(seg_cfg.get("output_dir", "data/derivatives/segmentation"))
        generate_qc = bool(seg_cfg.get("generate_qc_plots", False))
        require_fastsurfer = bool(seg_cfg.get("require_fastsurfer", True))
        rows = []
        for record in read_manifest(manifest):
            console.print(f"Segmenting {record.subject}/{record.session}")
            out_dir = out_root / record.subject / record.session
            mean_bold = Path(record.output_dir) / "mean_bold.nii.gz"
            brain_mask = Path(record.output_dir) / "brain_mask.nii.gz"
            base_row = {
                "subject": record.subject,
                "session": record.session,
                "method": "",
                "status": "ok",
                "error": "",
            }
            try:
                mri_dir = find_fastsurfer_mri_dir(record.subject, fs_config.subjects_dir)
                if mri_dir is not None:
                    outputs = save_fastsurfer_bold_segmentation(
                        mri_dir,
                        record.t1w_path,
                        mean_bold,
                        brain_mask,
                        out_dir,
                        fs_config,
                    )
                elif require_fastsurfer:
                    raise FileNotFoundError(
                        f"Missing FastSurfer labelmaps for {record.subject}. "
                        f"Expected {fs_config.subjects_dir / record.subject / 'mri'}; "
                        "run `hybrid-cvr run-fastsurfer` first."
                    )
                else:
                    outputs = save_fallback_segmentation(
                        mean_bold,
                        brain_mask,
                        record.t1w_path,
                        out_dir,
                        generate_qc_plot=generate_qc,
                    )
                rows.append(
                    {
                        **base_row,
                        "method": str(outputs["method"]),
                        "fastsurfer_mri_dir": str(outputs.get("fastsurfer_mri_dir", "")),
                        "t1w_to_bold": str(outputs.get("t1w_to_bold", "")),
                        "t1w_to_bold_mat": str(outputs.get("t1w_to_bold_mat", "")),
                        "wm_mask": str(outputs.get("wm_mask", "")),
                        "subcortical_gm_mask": str(outputs.get("subcortical_gm_mask", "")),
                        "cortical_gm_mask": str(outputs.get("cortical_gm_mask", "")),
                        "vcsf_mask": str(outputs.get("vcsf_mask", "")),
                        "wm_mask_t1w": str(outputs.get("wm_mask_t1w", "")),
                        "subcortical_gm_mask_t1w": str(outputs.get("subcortical_gm_mask_t1w", "")),
                        "cortical_gm_mask_t1w": str(outputs.get("cortical_gm_mask_t1w", "")),
                        "vcsf_mask_t1w": str(outputs.get("vcsf_mask_t1w", "")),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        **base_row,
                        "status": "failed",
                        "error": str(exc),
                        "fastsurfer_mri_dir": "",
                        "t1w_to_bold": "",
                        "t1w_to_bold_mat": "",
                        "wm_mask": "",
                        "subcortical_gm_mask": "",
                        "cortical_gm_mask": "",
                        "vcsf_mask": "",
                        "wm_mask_t1w": "",
                        "subcortical_gm_mask_t1w": "",
                        "cortical_gm_mask_t1w": "",
                        "vcsf_mask_t1w": "",
                    }
                )
        summary = out_root / "segmentation_summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        console.print(
            f"Wrote FastSurfer/FSL BOLD-space tissue masks for {len(rows)} sessions to {out_root}."
        )

    @app.command("segment-vessels")
    def segment_vessels(
        config: Path = typer.Option(..., "--config"),
        manifest: Path = typer.Option(..., "--manifest"),
    ) -> None:
        cfg = load_yaml(config)
        if not manifest.exists():
            raise typer.BadParameter(f"Manifest does not exist: {manifest}")
        from hybrid_cvr.segmentation.vessel_likelihood import (
            VesselLikelihoodConfig,
            apply_repeatability_boost,
            save_vessel_qc_plot,
            segment_vessels_session,
        )

        vessel_cfg = cfg.get("vessel_likelihood", {})
        default_cfg = VesselLikelihoodConfig()
        weights = {**default_cfg.weights, **vessel_cfg.get("weights", {})}
        seg_cfg = cfg.get("segmentation", {})
        out_root = Path(vessel_cfg.get("output_dir", "data/derivatives/vessels"))
        real_cvr_root = Path(vessel_cfg.get("real_cvr_dir", "data/derivatives/real_cvr"))
        segmentation_root = Path(
            vessel_cfg.get("segmentation_dir", seg_cfg.get("output_dir", "data/derivatives/segmentation"))
        )
        vessel_fit_cfg = VesselLikelihoodConfig(
            weights=weights,
            high_percentile=float(vessel_cfg.get("high_percentile", default_cfg.high_percentile)),
            medium_percentile=float(vessel_cfg.get("medium_percentile", default_cfg.medium_percentile)),
            min_component_size=int(vessel_cfg.get("min_component_size", default_cfg.min_component_size)),
            closing_iterations=int(vessel_cfg.get("closing_iterations", default_cfg.closing_iterations)),
            vcsf_buffer_iterations=int(
                vessel_cfg.get("vcsf_buffer_iterations", default_cfg.vcsf_buffer_iterations)
            ),
            tissue_boundary_buffer_iterations=int(
                vessel_cfg.get(
                    "tissue_boundary_buffer_iterations",
                    default_cfg.tissue_boundary_buffer_iterations,
                )
            ),
            frangi_sigmas=tuple(float(v) for v in vessel_cfg.get("frangi_sigmas", default_cfg.frangi_sigmas)),
            repeatability_weight=float(vessel_cfg.get("repeatability_weight", default_cfg.repeatability_weight)),
            use_repeatability=bool(vessel_cfg.get("use_repeatability", default_cfg.use_repeatability)),
            repeatability_resample_if_needed=bool(
                vessel_cfg.get(
                    "repeatability_resample_if_needed",
                    default_cfg.repeatability_resample_if_needed,
                )
            ),
            generate_qc_plots=bool(vessel_cfg.get("generate_qc_plots", default_cfg.generate_qc_plots)),
        )
        records = list(read_manifest(manifest))
        n_jobs = max(1, int(vessel_cfg.get("n_jobs", 1)))

        def run_one(record: object) -> tuple[dict[str, str], dict[str, Path | str | int | float]]:
            console.print(f"Segmenting vessels {record.subject}/{record.session}")
            base_row = {"subject": record.subject, "session": record.session, "status": "ok", "error": ""}
            try:
                outputs = segment_vessels_session(
                    processed_dir=Path(record.output_dir),
                    real_cvr_dir=real_cvr_root / record.subject / record.session,
                    segmentation_dir=segmentation_root / record.subject / record.session,
                    out_dir=out_root / record.subject / record.session,
                    config=vessel_fit_cfg,
                )
                row = {**base_row, **{key: str(value) for key, value in outputs.items()}}
                return row, outputs
            except Exception as exc:
                row = {
                    **base_row,
                    "status": "failed",
                    "error": str(exc),
                    "vessel_likelihood": "",
                    "vessel_mask_high_confidence": "",
                    "vessel_mask_medium_confidence": "",
                    "vcsf_exclusion_mask": "",
                    "candidate_mask": "",
                    "vessel_qc": "",
                    "n_candidate_voxels": "",
                    "n_high_voxels": "",
                    "n_medium_voxels": "",
                    "high_percentile": "",
                    "medium_percentile": "",
                    "repeatability_boosted": "",
                }
                return row, {}

        rows = []
        session_outputs: dict[tuple[str, str], dict[str, Path | str | int | float]] = {}
        if n_jobs == 1:
            for record in records:
                row, outputs = run_one(record)
                rows.append(row)
                if outputs:
                    session_outputs[(record.subject, record.session)] = outputs
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                futures = {pool.submit(run_one, record): record for record in records}
                for future in as_completed(futures):
                    record = futures[future]
                    row, outputs = future.result()
                    rows.append(row)
                    if outputs:
                        session_outputs[(record.subject, record.session)] = outputs
                    console.print(f"[{row['status']}] vessels {record.subject}/{record.session}")
            order = {(record.subject, record.session): i for i, record in enumerate(records)}
            rows.sort(key=lambda row: order[(row["subject"], row["session"])])

        apply_repeatability_boost(session_outputs, vessel_fit_cfg)
        if vessel_fit_cfg.generate_qc_plots:
            import nibabel as nib

            for record in records:
                outputs = session_outputs.get((record.subject, record.session))
                if not outputs or outputs.get("repeatability_boosted") != "true":
                    continue
                processed_dir = Path(record.output_dir)
                mean_bold = nib.load(str(processed_dir / "mean_bold.nii.gz")).get_fdata()
                likelihood = nib.load(str(outputs["vessel_likelihood"])).get_fdata()
                high = nib.load(str(outputs["vessel_mask_high_confidence"])).get_fdata() > 0
                medium = nib.load(str(outputs["vessel_mask_medium_confidence"])).get_fdata() > 0
                outputs["vessel_qc"] = save_vessel_qc_plot(
                    mean_bold,
                    likelihood,
                    high,
                    medium,
                    out_root / record.subject / record.session / "vessel_qc.png",
                )

        refreshed_rows = []
        for row in rows:
            outputs = session_outputs.get((row["subject"], row["session"]))
            if outputs:
                refreshed_rows.append(
                    {
                        **row,
                        **{key: str(value) for key, value in outputs.items()},
                    }
                )
            else:
                refreshed_rows.append(row)
        rows = refreshed_rows

        summary = out_root / "vessel_segmentation_summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "subject",
            "session",
            "status",
            "error",
            "vessel_likelihood",
            "vessel_mask_high_confidence",
            "vessel_mask_medium_confidence",
            "vcsf_exclusion_mask",
            "candidate_mask",
            "vessel_qc",
            "n_candidate_voxels",
            "n_high_voxels",
            "n_medium_voxels",
            "high_percentile",
            "medium_percentile",
            "repeatability_boosted",
        ]
        with summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"Wrote vessel masks for {len(rows)} sessions to {out_root}; summary: {summary}")

    @app.command("exclude-vessels-from-tissues")
    def exclude_vessels_from_tissues(
        config: Path = typer.Option(..., "--config"),
        manifest: Path = typer.Option(..., "--manifest"),
    ) -> None:
        cfg = load_yaml(config)
        if not manifest.exists():
            raise typer.BadParameter(f"Manifest does not exist: {manifest}")
        from hybrid_cvr.segmentation.tissue_vessel_exclusion import save_vessel_excluded_tissue_masks

        seg_cfg = cfg.get("segmentation", {})
        vessel_cfg = cfg.get("vessel_likelihood", {})
        out_root = Path(
            vessel_cfg.get("tissue_excluded_output_dir", "data/derivatives/tissue_masks_vessel_excluded")
        )
        segmentation_root = Path(
            vessel_cfg.get("segmentation_dir", seg_cfg.get("output_dir", "data/derivatives/segmentation"))
        )
        vessel_root = Path(vessel_cfg.get("output_dir", "data/derivatives/vessels"))
        rows = []
        for record in read_manifest(manifest):
            console.print(f"Excluding high-confidence vessel voxels {record.subject}/{record.session}")
            base_row = {"subject": record.subject, "session": record.session, "status": "ok", "error": ""}
            try:
                outputs = save_vessel_excluded_tissue_masks(
                    segmentation_root / record.subject / record.session,
                    vessel_root / record.subject / record.session,
                    out_root / record.subject / record.session,
                )
                rows.append({**base_row, **{key: str(value) for key, value in outputs.items()}})
            except Exception as exc:
                rows.append({**base_row, "status": "failed", "error": str(exc)})

        summary = out_root / "tissue_vessel_exclusion_summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        preferred = [
            "subject",
            "session",
            "status",
            "error",
            "vessel_likelihood",
            "vessel_mask_high_confidence",
            "cortical_gm_mask_no_vessel",
            "subcortical_gm_mask_no_vessel",
            "wm_mask_no_vessel",
            "vcsf_mask_no_vessel",
            "cortical_gm_mask_exclusive_no_vessel",
            "subcortical_gm_mask_exclusive_no_vessel",
            "wm_mask_exclusive_no_vessel",
            "vcsf_mask_exclusive_no_vessel",
            "tissue_label_map_no_vessel",
            "n_vessel_voxels",
        ]
        fieldnames = preferred + sorted({key for row in rows for key in row if key not in preferred})
        with summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"Wrote vessel-excluded tissue masks for {len(rows)} sessions to {out_root}.")

    @app.command("plot-etco2-qc")
    def plot_etco2_qc(
        manifest: Path = typer.Option(..., "--manifest"),
        processed_root: Path = typer.Option(Path("data/processed"), "--processed-root"),
    ) -> None:
        if not manifest.exists():
            raise typer.BadParameter(f"Manifest does not exist: {manifest}")
        from hybrid_cvr.preprocessing.qc import save_etco2_qc_plot

        rows = []
        for record in read_manifest(manifest):
            console.print(f"Plotting ETCO2 QC {record.subject}/{record.session}")
            out_dir = Path(record.output_dir)
            raw_gas = out_dir / "raw_gas.tsv"
            peaks = out_dir / "etco2_peaks.tsv"
            resampled = out_dir / "etco2_resampled.tsv"
            png = out_dir / "etco2_qc.png"
            base_row = {
                "subject": record.subject,
                "session": record.session,
                "status": "ok",
                "error": "",
                "raw_gas": str(raw_gas),
                "etco2_peaks": str(peaks),
                "etco2_resampled": str(resampled),
                "etco2_qc": str(png),
            }
            try:
                for path in (raw_gas, peaks, resampled):
                    if not path.exists():
                        raise FileNotFoundError(path)
                save_etco2_qc_plot(
                    raw_gas,
                    peaks,
                    resampled,
                    png,
                    title=f"{record.subject}_{record.session} ETCO2 plot",
                )
                rows.append(base_row)
            except Exception as exc:
                rows.append({**base_row, "status": "failed", "error": str(exc)})

        summary = processed_root / "etco2_qc_summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"Wrote ETCO2 QC plots for {len(rows)} sessions; summary: {summary}")

    @app.command("plot-fit-alignment-qc")
    def plot_fit_alignment_qc(
        config: Path = typer.Option(..., "--config"),
        manifest: Path = typer.Option(..., "--manifest"),
        real_fit_dir: Path = typer.Option(Path("data/derivatives/real_cvr"), "--real-fit-dir"),
    ) -> None:
        cfg = load_yaml(config)
        if not manifest.exists():
            raise typer.BadParameter(f"Manifest does not exist: {manifest}")
        from hybrid_cvr.cvr.alignment_qc import save_fit_alignment_plot
        from hybrid_cvr.cvr.fit_real import RealCVRFitConfig

        delay_cfg = cfg.get("delay_grid_seconds", {})
        T_cfg = cfg.get("T_grid_seconds", {})
        glm_cfg = cfg.get("glm", {})
        glm_delay_cfg = glm_cfg.get("delay_grid_seconds", {})
        fit_cfg = RealCVRFitConfig(
            output_dir=Path(cfg.get("output_dir", "data/derivatives/real_cvr")),
            delay_min=float(delay_cfg.get("min", 0.0)),
            delay_max=float(delay_cfg.get("max", 80.0)),
            delay_step=float(delay_cfg.get("step", 1.55)),
            T_min=float(T_cfg.get("min", 2.0)),
            T_max=float(T_cfg.get("max", 100.0)),
            T_step=float(T_cfg.get("step", 2.0)),
            include_drift=bool(cfg.get("drift_regressor", True)),
            candidate_batch_size=int(cfg.get("candidate_batch_size", 256)),
            glm_delay_mode=str(glm_cfg.get("delay_mode", cfg.get("glm_delay_mode", "voxelwise"))),
            glm_global_delay_seconds=glm_cfg.get("global_delay_seconds"),
            glm_delay_min=(
                float(glm_delay_cfg["min"]) if glm_delay_cfg.get("min") is not None else None
            ),
            glm_delay_max=(
                float(glm_delay_cfg["max"]) if glm_delay_cfg.get("max") is not None else None
            ),
            glm_delay_step=(
                float(glm_delay_cfg["step"]) if glm_delay_cfg.get("step") is not None else None
            ),
        )
        rows = []
        for record in read_manifest(manifest):
            console.print(f"Plotting fit alignment QC {record.subject}/{record.session}")
            processed_dir = Path(record.output_dir)
            out_dir = real_fit_dir / record.subject / record.session
            png = out_dir / "fit_alignment_qc.png"
            base_row = {
                "subject": record.subject,
                "session": record.session,
                "status": "ok",
                "error": "",
                "alignment_qc": str(png),
            }
            try:
                outputs = save_fit_alignment_plot(
                    processed_dir / "bold_psc.nii.gz",
                    processed_dir / "etco2_resampled.tsv",
                    processed_dir / "valid_signal_mask.nii.gz",
                    png,
                    fit_cfg,
                    f"{record.subject}_{record.session} BOLD-ETCO2 fit alignment",
                )
                rows.append({**base_row, **{key: str(value) for key, value in outputs.items()}})
            except Exception as exc:
                rows.append({**base_row, "status": "failed", "error": str(exc)})

        summary = real_fit_dir / "fit_alignment_qc_summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"Wrote fit alignment QC plots for {len(rows)} sessions; summary: {summary}")

    @app.command("fit-real-cvr")
    def fit_real_cvr(
        config: Path = typer.Option(..., "--config"),
        manifest: Path = typer.Option(..., "--manifest"),
    ) -> None:
        cfg = load_yaml(config)
        if not manifest.exists():
            raise typer.BadParameter(f"Manifest does not exist: {manifest}")
        from hybrid_cvr.cvr.fit_real import RealCVRFitConfig, fit_real_cvr_session

        delay_cfg = cfg.get("delay_grid_seconds", {})
        T_cfg = cfg.get("T_grid_seconds", {})
        out_root = Path(cfg.get("output_dir", "data/derivatives/real_cvr"))
        glm_cfg = cfg.get("glm", {})
        glm_delay_cfg = glm_cfg.get("delay_grid_seconds", {})
        fit_cfg = RealCVRFitConfig(
            output_dir=out_root,
            delay_min=float(delay_cfg.get("min", 0.0)),
            delay_max=float(delay_cfg.get("max", 80.0)),
            delay_step=float(delay_cfg.get("step", 1.55)),
            T_min=float(T_cfg.get("min", 2.0)),
            T_max=float(T_cfg.get("max", 100.0)),
            T_step=float(T_cfg.get("step", 2.0)),
            include_drift=bool(cfg.get("drift_regressor", True)),
            chunk_voxels=int(cfg.get("chunk_voxels", 6000)),
            candidate_batch_size=int(cfg.get("candidate_batch_size", 256)),
            save_reconstruction=bool(cfg.get("save_reconstruction", False)),
            glm_delay_mode=str(glm_cfg.get("delay_mode", cfg.get("glm_delay_mode", "global"))),
            glm_global_delay_seconds=glm_cfg.get("global_delay_seconds"),
            glm_delay_min=(
                float(glm_delay_cfg["min"]) if glm_delay_cfg.get("min") is not None else None
            ),
            glm_delay_max=(
                float(glm_delay_cfg["max"]) if glm_delay_cfg.get("max") is not None else None
            ),
            glm_delay_step=(
                float(glm_delay_cfg["step"]) if glm_delay_cfg.get("step") is not None else None
            ),
        )
        records = list(read_manifest(manifest))
        n_jobs = max(1, int(cfg.get("n_jobs", 1)))
        rows = []

        def run_one_fit(record: object) -> dict[str, str]:
            console.print(f"Fitting real CVR {record.subject}/{record.session}")
            base_row = {"subject": record.subject, "session": record.session, "status": "ok", "error": ""}
            processed_dir = Path(record.output_dir)
            out_dir = fit_cfg.output_dir / record.subject / record.session
            try:
                outputs = fit_real_cvr_session(
                    psc_path=processed_dir / "bold_psc.nii.gz",
                    etco2_resampled_path=processed_dir / "etco2_resampled.tsv",
                    brain_mask_path=processed_dir / "valid_signal_mask.nii.gz",
                    mean_bold_path=processed_dir / "mean_bold.nii.gz",
                    out_dir=out_dir,
                    config=fit_cfg,
                )
                return {**base_row, **{key: str(value) for key, value in outputs.items()}}
            except Exception as exc:
                return {
                    **base_row,
                    "status": "failed",
                    "error": str(exc),
                    "n_voxels": "",
                    "n_timepoints": "",
                    "n_delay_candidates": "",
                    "n_glm_delay_candidates": "",
                    "n_T_candidates": "",
                    "delay_min": "",
                    "delay_max": "",
                    "glm_delay_min": "",
                    "glm_delay_max": "",
                    "T_min": "",
                    "T_max": "",
                    "glm_delay_mode": "",
                    "glm_global_delay_seconds": "",
                    "glm_global_r2": "",
                    "valid_fit_mask": "",
                    "glm_cvr": "",
                    "glm_delay": "",
                    "glm_r2": "",
                    "glm_ssr": "",
                    "glm_rmse": "",
                    "glm_drift": "",
                    "hrf_cvr": "",
                    "hrf_delay": "",
                    "hrf_T": "",
                    "hrf_effective_delay": "",
                    "hrf_r2": "",
                    "hrf_ssr": "",
                    "hrf_rmse": "",
                    "tCNR": "",
                    "etco2_correlation_peak": "",
                    "fit_quality": "",
                    "hrf_residual_std": "",
                    "ode_residual_rms": "",
                    "hrf_reconstructed_bold_psc": "",
                }

        if n_jobs == 1:
            rows = [run_one_fit(record) for record in records]
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                futures = {pool.submit(run_one_fit, record): record for record in records}
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    console.print(f"[{row['status']}] {row['subject']}/{row['session']}")
            order = {(record.subject, record.session): i for i, record in enumerate(records)}
            rows.sort(key=lambda row: order[(row["subject"], row["session"])])
        summary = out_root / "real_cvr_summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        console.print(f"Wrote real CVR parameter maps for {len(rows)} sessions to {out_root}.")

    @app.command("pool-distributions")
    def pool_distributions(
        config: Path = typer.Option(..., "--config"),
        real_fit_dir: Path = typer.Option(..., "--real-fit-dir"),
        seg_dir: Path = typer.Option(..., "--seg-dir"),
        out: Path = typer.Option(..., "--out"),
    ) -> None:
        cfg = load_yaml(config)
        manifest = Path("data/processed/manifest.csv")
        if not manifest.exists():
            raise typer.BadParameter(f"Manifest does not exist: {manifest}")
        from hybrid_cvr.distributions.pool import (
            DistributionPoolConfig,
            pool_real_parameter_distributions,
        )

        qc_cfg = cfg.get("qc", {})
        pool_cfg = cfg.get("distribution_pooling", {})
        pool_config = DistributionPoolConfig(
            cvr_min=float(qc_cfg.get("cvr_min", -0.4)),
            cvr_max=float(qc_cfg.get("cvr_max", 1.8)),
            delay_min=float(qc_cfg.get("delay_min", 0.0)),
            delay_max=float(qc_cfg.get("delay_max", 80.0)),
            T_min=float(qc_cfg.get("T_min", 2.0)),
            T_max=float(qc_cfg.get("T_max", 100.0)),
            r2_min=float(qc_cfg.get("r2_min", 0.05)),
            max_voxels_per_region_session=(
                int(pool_cfg["max_voxels_per_region_session"])
                if pool_cfg.get("max_voxels_per_region_session") is not None
                else None
            ),
            include_whole_brain=bool(pool_cfg.get("include_whole_brain", True)),
            include_boundary_region=bool(pool_cfg.get("include_boundary_region", True)),
            random_seed=int(pool_cfg.get("random_seed", 17)),
        )
        records = list(read_manifest(manifest))
        outputs = pool_real_parameter_distributions(
            records=records,
            real_fit_dir=real_fit_dir,
            mask_dir=seg_dir,
            segmentation_dir=Path(pool_cfg.get("segmentation_dir", "data/derivatives/segmentation")),
            out_path=out,
            config=pool_config,
        )
        console.print(
            f"Pooled {outputs['n_rows']} parameter rows across {outputs['n_sessions']} sessions. "
            f"Samples: {outputs['samples']}; summary: {outputs['summary']}"
        )

    @app.command("segment-mida")
    def segment_mida_command(
        config: Path = typer.Option(Path("configs/simulation.yaml"), "--config"),
    ) -> None:
        cfg = load_yaml(config)
        from hybrid_cvr.simulation.mida import segment_mida

        outputs = segment_mida(_mida_segmentation_config(cfg))
        console.print(
            f"Wrote MIDA high-res masks to {outputs['highres_dir']} and 2.5 mm fraction maps "
            f"to {outputs['lowres_dir']}; summary: {outputs['summary']}"
        )

    @app.command("generate-mida-parameter-maps")
    def generate_mida_parameter_maps_command(
        config: Path = typer.Option(Path("configs/simulation.yaml"), "--config"),
        n_cases: int | None = typer.Option(None, "--n-cases"),
        out: Path | None = typer.Option(None, "--out"),
    ) -> None:
        cfg = load_yaml(config)
        from dataclasses import replace

        from hybrid_cvr.simulation.mida import generate_mida_parameter_maps

        map_config = _mida_parameter_config(cfg)
        if n_cases is not None:
            map_config = replace(map_config, n_cases=n_cases)
        if out is not None:
            map_config = replace(map_config, output_dir=out)
        outputs = generate_mida_parameter_maps(map_config)
        console.print(
            f"Wrote {outputs['n_cases']} MIDA partial-volume GT cases to {outputs['output_dir']}; "
            f"summary: {outputs['summary']}"
        )

    @app.command("prepare-case-index")
    def prepare_case_index_command(
        sim_root: Path = typer.Option(Path("data/simulated"), "--sim-root"),
        parameter_dir: Path | None = typer.Option(None, "--parameter-dir"),
        case_index: Path | None = typer.Option(None, "--case-index"),
        split_json: Path | None = typer.Option(None, "--split-json"),
    ) -> None:
        from hybrid_cvr.simulation.case_index import prepare_case_index

        result = prepare_case_index(
            sim_root,
            parameter_dir=parameter_dir,
            case_index_path=case_index,
            split_json_path=split_json,
        )
        console.print(f"Found GT cases at: {result['found_location']}")
        console.print(
            "Case organisation summary: "
            f"found={result['cases_found']} moved={result['cases_moved']} "
            f"valid={result['cases_valid']} missing={len(result['missing_files'])} "
            f"final={result['final_location']}"
        )
        console.print(f"Case index: {result['case_index']}")
        console.print(f"Split JSON: {result['split_json']}")

    @app.command("simulate")
    def simulate(
        config: Path = typer.Option(..., "--config"),
        out: Path = typer.Option(..., "--out"),
        phantom_debug: bool = typer.Option(False, "--phantom-debug"),
    ) -> None:
        cfg = load_yaml(config)
        if phantom_debug:
            np = __import__("numpy")
            from hybrid_cvr.simulation.co2_paradigms import block_paradigm
            from hybrid_cvr.simulation.dataset_writer import write_npz_case
            from hybrid_cvr.simulation.phantom import default_parameter_maps, make_phantom_labels
            from hybrid_cvr.simulation.simulate_bold import simulate_bold_psc

            phantom = cfg.get("phantom", {})
            shape = tuple(phantom.get("shape", [32, 32, 8]))
            n_time = int(phantom.get("timepoints", 80))
            tr = float(phantom.get("tr_seconds", 1.55))
            time_grid = np.arange(n_time, dtype=float) * tr
            labels = make_phantom_labels(shape)
            maps = default_parameter_maps(labels)
            z = shape[2] // 2
            etco2 = block_paradigm(time_grid)
            sim = simulate_bold_psc(
                maps["GT_CVR"][..., z], maps["GT_delay"][..., z], maps["GT_T"][..., z], time_grid, etco2
            )
            write_npz_case(
                Path(out) / "sim_000_slice.npz",
                features=np.stack([sim["bold_psc"].mean(axis=0), sim["bold_psc"].std(axis=0)], axis=0),
                bold_psc=sim["bold_psc"],
                etco2=etco2,
                time_grid=time_grid,
                mask=(labels[..., z] > 0).astype("float32"),
                GT_CVR=maps["GT_CVR"][..., z],
                GT_delay=maps["GT_delay"][..., z],
                GT_T=maps["GT_T"][..., z],
            )
            console.print(f"Wrote debug simulation to {out}")
            return

        from hybrid_cvr.simulation.mida_bold import simulate_mida_bold_dataset

        outputs = simulate_mida_bold_dataset(_mida_bold_simulation_config(cfg, out))
        console.print(
            f"Wrote {outputs['n_cases']} MIDA BOLD simulations to {outputs['output_dir']}; "
            f"summary: {outputs['summary']}"
        )

    @app.command("train")
    def train(
        config: Path = typer.Option(..., "--config"),
        sim_root: Path = typer.Option(..., "--sim-root"),
        out: Path = typer.Option(..., "--out"),
        case_index: Path | None = typer.Option(None, "--case-index"),
        split_json: Path | None = typer.Option(None, "--split-json"),
        on_the_fly: bool = typer.Option(True, "--on-the-fly/--saved-npz"),
        max_epochs: int | None = typer.Option(None, "--max-epochs"),
        samples_per_epoch: int | None = typer.Option(None, "--samples-per-epoch"),
    ) -> None:
        cfg = load_yaml(config)
        if case_index is not None:
            cfg.setdefault("dataset", {})["case_index_path"] = str(case_index)
        if split_json is not None:
            cfg.setdefault("dataset", {})["split_json_path"] = str(split_json)
        from hybrid_cvr.training.train import train_unet_pinn, validate_training_config

        try:
            validate_training_config(cfg)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if on_the_fly:
            result = train_unet_pinn(
                cfg,
                sim_root,
                out,
                max_epochs=max_epochs,
                samples_per_epoch=samples_per_epoch,
            )
            console.print(
                f"Training complete: best={result['best_checkpoint']} "
                f"loss={result['best_loss']:.6g} device={result['device']}"
            )
        else:
            _write_status(out, f"Saved-NPZ training scaffold checked {sim_root}.")
            console.print(f"Wrote training status to {out}")

    @app.command("evaluate")
    def evaluate(
        checkpoint: Path = typer.Option(..., "--checkpoint"),
        sim_root: Path = typer.Option(..., "--sim-root"),
        out: Path = typer.Option(..., "--out"),
    ) -> None:
        _write_status(out, f"Evaluation may use simulation GT maps. checkpoint={checkpoint} sim_root={sim_root}")
        console.print(f"Wrote evaluation status to {out}")

    @app.command("predict-sim")
    def predict_sim(
        checkpoint: Path = typer.Option(..., "--checkpoint"),
        config: Path = typer.Option(..., "--config"),
        sim_root: Path = typer.Option(Path("data/simulated"), "--sim-root"),
        out: Path = typer.Option(..., "--out"),
        split: str = typer.Option("test", "--split"),
        case_id: str | None = typer.Option(None, "--case-id"),
        paradigm: str = typer.Option("block", "--paradigm"),
        tcnr: float = typer.Option(2.0, "--tcnr"),
    ) -> None:
        cfg = load_yaml(config)
        from hybrid_cvr.inference.predict_sim import predict_sim_parameter_maps

        result = predict_sim_parameter_maps(
            checkpoint,
            cfg,
            sim_root,
            out,
            split=split,
            case_id=case_id,
            paradigm=paradigm,
            tcnr=tcnr,
        )
        console.print(
            f"Wrote predicted simulated maps to {out} "
            f"({result['processed_slices']} slices, device={result['device']})"
        )

    @app.command("summarize-training")
    def summarize_training(
        model_dir: Path = typer.Option(Path("data/models/unet_pinn"), "--model-dir"),
    ) -> None:
        from hybrid_cvr.training.log_summary import summarize_training_logs

        result = summarize_training_logs(model_dir)
        console.print(
            f"Wrote readable training summary after {result['epochs']} epochs: "
            f"{result['summary_markdown']}"
        )

    @app.command("infer-real")
    def infer_real(
        checkpoint: Path = typer.Option(..., "--checkpoint"),
        config: Path = typer.Option(..., "--config"),
        subject: str = typer.Option(..., "--subject"),
        session: str = typer.Option(..., "--session"),
        bids_root: Path = typer.Option(Path("data/raw"), "--bids-root"),
        processed_root: Path = typer.Option(Path("data/processed"), "--processed-root"),
        segmentation_root: Path = typer.Option(Path("data/derivatives/segmentation"), "--segmentation-root"),
        vessel_root: Path = typer.Option(Path("data/derivatives/vessels"), "--vessel-root"),
        out: Path = typer.Option(..., "--out"),
    ) -> None:
        cfg = load_yaml(config)
        from hybrid_cvr.inference.infer_real import infer_real_session

        processed_dir = processed_root / subject / session
        segmentation_dir = segmentation_root / subject / session
        vessel_dir = vessel_root / subject / session
        result = infer_real_session(
            checkpoint,
            cfg,
            processed_dir,
            segmentation_dir,
            out,
            vessel_dir=vessel_dir if vessel_dir.exists() else None,
        )
        _write_status(
            out,
            (
                f"Real inference complete for {subject}/{session} using {checkpoint} at {bids_root}; "
                f"processed_slices={result['processed_slices']} device={result['device']}"
            ),
        )
        console.print(
            f"Wrote real inference maps to {out} "
            f"({result['processed_slices']} slices, device={result['device']})"
        )

    @app.command("adapt-real")
    def adapt_real(
        checkpoint: Path = typer.Option(..., "--checkpoint"),
        config: Path = typer.Option(..., "--config"),
        subject: str = typer.Option(..., "--subject"),
        session: str = typer.Option(..., "--session"),
        out: Path = typer.Option(..., "--out"),
    ) -> None:
        _ = load_yaml(config)
        _write_status(out, f"Self-supervised adaptation scaffold for {subject}/{session} using {checkpoint}")
        console.print(f"Wrote adaptation status to {out}")


def main() -> None:  # pragma: no cover
    if app is None:
        raise RuntimeError("Missing CLI dependencies. Install with: pip install -e '.[dev]'")
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
