from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_CASE_FILES = {
    "gt_cvr_path": "GT_CVR.nii.gz",
    "gt_delay_path": "GT_delay.nii.gz",
    "gt_T_path": "GT_T.nii.gz",
    "region_labels_path": "GT_region_labels.nii.gz",
    "frac_cortical_gm_path": "GT_fraction_cortical_gm.nii.gz",
    "frac_subcortical_gm_path": "GT_fraction_subcortical_gm.nii.gz",
    "frac_wm_path": "GT_fraction_wm.nii.gz",
    "frac_vcsf_path": "GT_fraction_vcsf.nii.gz",
    "frac_vessel_like_path": "GT_fraction_vessel_like.nii.gz",
}


def prepare_case_index(
    sim_root: str | Path,
    *,
    parameter_dir: str | Path | None = None,
    case_index_path: str | Path | None = None,
    split_json_path: str | Path | None = None,
    validation_conditions_path: str | Path | None = None,
) -> dict[str, Any]:
    sim_root = Path(sim_root)
    parameter_dir = Path(parameter_dir) if parameter_dir is not None else sim_root / "mida_parameters"
    case_index_path = Path(case_index_path) if case_index_path else sim_root / "case_index.csv"
    split_json_path = Path(split_json_path) if split_json_path else sim_root / "splits" / "train_val_test_split.json"
    validation_conditions_path = (
        Path(validation_conditions_path)
        if validation_conditions_path
        else sim_root / "splits" / "validation_conditions.json"
    )
    case_dirs = sorted(p for p in parameter_dir.glob("case_*") if p.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f"No case_* directories found under {parameter_dir}")

    train_cases = [f"case_{idx:03d}" for idx in range(0, min(60, len(case_dirs)))]
    val_cases = [f"case_{idx:03d}" for idx in range(60, min(80, len(case_dirs)))]
    test_cases = [f"case_{idx:03d}" for idx in range(80, min(100, len(case_dirs)))]
    split_by_case = {case: "train" for case in train_cases}
    split_by_case.update({case: "validation" for case in val_cases})
    split_by_case.update({case: "test" for case in test_cases})

    rows = []
    missing: dict[str, list[str]] = {}
    valid = 0
    for case_dir in case_dirs:
        case_id = case_dir.name
        row = {"case_id": case_id, "case_dir": str(case_dir)}
        case_missing = []
        for column, filename in REQUIRED_CASE_FILES.items():
            path = case_dir / filename
            row[column] = str(path)
            if not path.exists():
                case_missing.append(filename)
        row["split"] = split_by_case.get(case_id, "unused")
        rows.append(row)
        if case_missing:
            missing[case_id] = case_missing
        else:
            valid += 1
        _write_case_metadata(case_dir, row, case_missing)

    case_index_path.parent.mkdir(parents=True, exist_ok=True)
    with case_index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    split_json_path.parent.mkdir(parents=True, exist_ok=True)
    split_json_path.write_text(
        json.dumps(
            {
                "train_cases": train_cases,
                "val_cases": val_cases,
                "test_cases": test_cases,
                "split_unit": "gt_parameter_case",
                "notes": (
                    "GT maps are used only for simulation generation and final evaluation, "
                    "not for training loss."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    validation_conditions_path.parent.mkdir(parents=True, exist_ok=True)
    validation_conditions_path.write_text(
        json.dumps(
            {
                "validation_cases": val_cases,
                "paradigms": [
                    "block",
                    "multi_step",
                    "pseudo_random_binary",
                    "sinusoidal",
                    "breath_hold_like",
                    "resting_state_like",
                ],
                "tcnr_levels": [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                "selected_axial_slices": "deterministic_from_seed",
                "fixed_noise_seed_offset": 100000,
                "exclude_ramp": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "found_location": str(parameter_dir),
        "cases_found": len(case_dirs),
        "cases_moved": 0,
        "cases_valid": valid,
        "missing_files": missing,
        "final_location": str(parameter_dir),
        "case_index": str(case_index_path),
        "split_json": str(split_json_path),
        "validation_conditions": str(validation_conditions_path),
    }


def _write_case_metadata(case_dir: Path, row: dict[str, str], missing: list[str]) -> None:
    metadata = {
        "case_id": row["case_id"],
        "split": row["split"],
        "required_files_valid": not missing,
        "missing_files": missing,
        "notes": (
            "GT parameter maps are simulator/evaluation artifacts and must not be "
            "used as model input or training targets."
        ),
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
