from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BidsRecord:
    subject: str
    session: str
    t1w_path: str
    bold_path: str
    gas_path: str
    output_dir: str


def _first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def _find_subject_t1w(sub_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        anat = ses_dir / "anat"
        candidates.extend(sorted(anat.glob("*_T1w.nii")))
        candidates.extend(sorted(anat.glob("*_T1w.nii.gz")))
    return _first_existing(candidates)


def scan_bids_like(root: str | Path, output_base: str | Path = "data/processed") -> list[BidsRecord]:
    root = Path(root)
    records: list[BidsRecord] = []
    if not root.exists():
        return records
    for sub_dir in sorted(root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        subject_t1w = _find_subject_t1w(sub_dir)
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            anat = ses_dir / "anat"
            cvr = ses_dir / "cvr"
            t1w = _first_existing(sorted(anat.glob("*_T1w.nii")) + sorted(anat.glob("*_T1w.nii.gz")))
            if t1w is None:
                t1w = subject_t1w
            bold = _first_existing(sorted(cvr.glob("*_bold.nii.gz")))
            gas = _first_existing(sorted(cvr.glob("*_gas_traces.txt")))
            if t1w and bold and gas:
                output_dir = Path(output_base) / sub_dir.name / ses_dir.name
                records.append(
                    BidsRecord(
                        subject=sub_dir.name,
                        session=ses_dir.name,
                        t1w_path=str(t1w),
                        bold_path=str(bold),
                        gas_path=str(gas),
                        output_dir=str(output_dir),
                    )
                )
    return records


def write_manifest(records: list[BidsRecord], out: str | Path) -> Path:
    import csv

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "session", "t1w_path", "bold_path", "gas_path", "output_dir"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
    return out


def read_manifest(path: str | Path) -> list[BidsRecord]:
    import csv

    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return [BidsRecord(**row) for row in csv.DictReader(f)]
