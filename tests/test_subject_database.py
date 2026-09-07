import json
import zipfile
from pathlib import Path

from hybrid_cvr.preprocessing.subject_database import load_scan_start_times_by_participants


def test_scan_start_times_match_participants_by_age_sex_and_acquisition_time(tmp_path: Path):
    database = tmp_path / "database.xlsx"
    _write_minimal_database_xlsx(
        database,
        [
            ["ID", "Scan", "Age", "Sex", "Start_time"],
            ["CVR_reliability_05", 1, 26, "M", 53105 / 86400],
            ["CVR_reliability_12", 1, 26, "M", 42960 / 86400],
        ],
    )
    participants = tmp_path / "participants.txt"
    participants.write_text(
        "participant_id\tage\tsex\nsub-07\t26\tm\nsub-09\t26\tm\n",
        encoding="utf-8",
    )
    bold_paths = {
        "sub-07": {"ses-01": tmp_path / "sub-07_ses-01_bold.nii.gz"},
        "sub-09": {"ses-01": tmp_path / "sub-09_ses-01_bold.nii.gz"},
    }
    (tmp_path / "sub-07_ses-01_bold.json").write_text(
        json.dumps({"AcquisitionTime": "14:45:14.942500"}),
        encoding="utf-8",
    )
    (tmp_path / "sub-09_ses-01_bold.json").write_text(
        json.dumps({"AcquisitionTime": "11:56:11.015000"}),
        encoding="utf-8",
    )

    starts = load_scan_start_times_by_participants(database, participants, bold_paths)

    assert starts[("sub-07", "ses-01")] == 53105
    assert starts[("sub-09", "ses-01")] == 42960


def _write_minimal_database_xlsx(path: Path, rows: list[list[object]]) -> None:
    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            ref = f"{_column_name(col_index)}{row_index}"
            if isinstance(value, str):
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    sheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/worksheets/sheet1.xml", sheet)


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name
