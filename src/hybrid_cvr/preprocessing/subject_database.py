from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def load_scan_start_times(path: str | Path) -> dict[tuple[str, str], float]:
    return load_scan_start_times_by_database_id(path)


def load_scan_start_times_by_database_id(path: str | Path) -> dict[tuple[str, str], float]:
    rows = _read_first_sheet_rows(Path(path))
    if not rows:
        return {}
    headers = [str(value).strip() if value is not None else "" for value in rows[0]]
    required = {"ID", "Scan", "Start_time"}
    missing = required - set(headers)
    if missing:
        raise ValueError(f"Subject database missing required columns: {sorted(missing)}")

    id_i = headers.index("ID")
    scan_i = headers.index("Scan")
    start_i = headers.index("Start_time")
    starts: dict[tuple[str, str], float] = {}
    for row in rows[1:]:
        if id_i >= len(row) or scan_i >= len(row) or start_i >= len(row):
            continue
        subject = _subject_from_database_id(row[id_i])
        session = _session_from_scan(row[scan_i])
        if subject is None or session is None or row[start_i] is None:
            continue
        starts[(subject, session)] = seconds_from_excel_time(row[start_i])
    return starts


def load_scan_start_times_by_participants(
    database_path: str | Path,
    participants_path: str | Path,
    bold_paths_by_subject: dict[str, dict[str, str | Path]] | None = None,
) -> dict[tuple[str, str], float]:
    participants = load_participants(participants_path)
    database_subjects = _read_database_subject_groups(database_path)
    available = list(database_subjects)
    starts: dict[tuple[str, str], float] = {}
    for subject in sorted(participants):
        age, sex = participants[subject]
        candidates = [row for row in available if row["age"] == age and row["sex"] == sex]
        if not candidates:
            raise ValueError(f"No subject-database row matches {subject} age={age} sex={sex}")
        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            chosen = _choose_closest_to_bold_acquisition(subject, candidates, bold_paths_by_subject)
        available.remove(chosen)
        for session, start_time in chosen["starts"].items():
            starts[(subject, session)] = start_time
    return starts


def load_participants(path: str | Path) -> dict[str, tuple[int, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"participant_id", "age", "sex"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Participants file missing required columns: {sorted(missing)}")
        return {
            row["participant_id"]: (int(float(row["age"])), str(row["sex"]).strip().lower()[0])
            for row in reader
            if row.get("participant_id")
        }


def load_gas_clock_start_seconds(path: str | Path, clock_column: str = "Time") -> float | None:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        first = f.readline().strip()
        delimiter = "\t" if "\t" in first else None
        headers = first.split(delimiter) if delimiter else first.split()
        if clock_column not in headers:
            return None
        reader = csv.DictReader(f, fieldnames=headers, delimiter=delimiter)
        for row in reader:
            value = row.get(clock_column)
            if value:
                return seconds_from_clock_text(value)
    return None


def bold_acquisition_offset_seconds(bold_path: str | Path, gas_path: str | Path) -> float:
    bold_json = Path(bold_path).with_suffix("").with_suffix(".json")
    metadata = json.loads(bold_json.read_text(encoding="utf-8"))
    acquisition_time = metadata.get("AcquisitionTime")
    if acquisition_time is None:
        return 0.0
    gas_start = load_gas_clock_start_seconds(gas_path)
    if gas_start is None:
        return 0.0
    return _normalize_clock_offset(seconds_from_clock_text(str(acquisition_time)) - gas_start)


def scan_start_or_bold_acquisition_offset_seconds(
    subject: str,
    session: str,
    bold_path: str | Path,
    gas_path: str | Path,
    scan_start_times: dict[tuple[str, str], float],
) -> float:
    database_offset = scan_start_offset_seconds(subject, session, gas_path, scan_start_times)
    gas_range = load_gas_elapsed_range_seconds(gas_path)
    if gas_range is not None:
        gas_min, gas_max = gas_range
        if gas_min <= database_offset <= gas_max:
            return database_offset
    return bold_acquisition_offset_seconds(bold_path, gas_path)


def scan_start_offset_seconds(
    subject: str,
    session: str,
    gas_path: str | Path,
    scan_start_times: dict[tuple[str, str], float],
) -> float:
    scan_start = scan_start_times.get((subject, session))
    if scan_start is None:
        return 0.0
    gas_start = load_gas_clock_start_seconds(gas_path)
    if gas_start is None:
        return 0.0
    return _normalize_clock_offset(scan_start - gas_start)


def load_gas_elapsed_range_seconds(path: str | Path, elapsed_column: str = "sec") -> tuple[float, float] | None:
    path = Path(path)
    first_value: float | None = None
    last_value: float | None = None
    with path.open("r", encoding="utf-8", newline="") as f:
        first = f.readline().strip()
        delimiter = "\t" if "\t" in first else None
        headers = first.split(delimiter) if delimiter else first.split()
        if elapsed_column not in headers:
            return None
        reader = csv.DictReader(f, fieldnames=headers, delimiter=delimiter)
        for row in reader:
            value = row.get(elapsed_column)
            if value in (None, ""):
                continue
            seconds = float(value)
            if first_value is None:
                first_value = seconds
            last_value = seconds
    if first_value is None or last_value is None:
        return None
    return first_value, last_value


def _normalize_clock_offset(offset: float) -> float:
    if offset < -12 * 3600:
        offset += 24 * 3600
    if offset > 12 * 3600:
        offset -= 24 * 3600
    return float(offset)


def _read_database_subject_groups(path: str | Path) -> list[dict[str, Any]]:
    rows = _read_first_sheet_rows(Path(path))
    if not rows:
        return []
    headers = [str(value).strip() if value is not None else "" for value in rows[0]]
    required = {"ID", "Scan", "Age", "Sex", "Start_time"}
    missing = required - set(headers)
    if missing:
        raise ValueError(f"Subject database missing required columns: {sorted(missing)}")
    id_i = headers.index("ID")
    scan_i = headers.index("Scan")
    age_i = headers.index("Age")
    sex_i = headers.index("Sex")
    start_i = headers.index("Start_time")
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        if max(id_i, scan_i, age_i, sex_i, start_i) >= len(row):
            continue
        if row[id_i] is None or row[scan_i] is None or row[start_i] is None:
            continue
        database_id = str(row[id_i])
        group = grouped.setdefault(
            database_id,
            {
                "database_id": database_id,
                "age": int(float(row[age_i])),
                "sex": str(row[sex_i]).strip().lower()[0],
                "starts": {},
            },
        )
        session = _session_from_scan(row[scan_i])
        if session is not None:
            group["starts"][session] = seconds_from_excel_time(row[start_i])
    return list(grouped.values())


def _choose_closest_to_bold_acquisition(
    subject: str,
    candidates: list[dict[str, Any]],
    bold_paths_by_subject: dict[str, dict[str, str | Path]] | None,
) -> dict[str, Any]:
    if not bold_paths_by_subject or subject not in bold_paths_by_subject:
        raise ValueError(f"Ambiguous age/sex match for {subject}; provide BOLD paths to break ties")
    bold_paths = bold_paths_by_subject[subject]

    def score(candidate: dict[str, Any]) -> float:
        total = 0.0
        n = 0
        for session, start_time in candidate["starts"].items():
            bold_path = bold_paths.get(session)
            if bold_path is None:
                continue
            bold_json = Path(bold_path).with_suffix("").with_suffix(".json")
            if not bold_json.exists():
                continue
            metadata = json.loads(bold_json.read_text(encoding="utf-8"))
            acquisition_time = metadata.get("AcquisitionTime")
            if acquisition_time is None:
                continue
            total += _circular_clock_difference(start_time, seconds_from_clock_text(str(acquisition_time)))
            n += 1
        return total / max(n, 1)

    return min(candidates, key=score)


def _circular_clock_difference(a: float, b: float) -> float:
    diff = abs(float(a) - float(b))
    return min(diff, 24 * 3600 - diff)


def seconds_from_excel_time(value: Any) -> float:
    if hasattr(value, "hour") and hasattr(value, "minute") and hasattr(value, "second"):
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    if isinstance(value, str):
        return seconds_from_clock_text(value)
    return float(value) * 24.0 * 3600.0


def seconds_from_clock_text(value: str) -> float:
    text = value.strip()
    match = re.match(r"^(?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{1,2}(?:\.\d+)?))?$", text)
    if not match:
        raise ValueError(f"Could not parse clock time: {value!r}")
    seconds = float(match.group("s") or 0.0)
    return float(int(match.group("h")) * 3600 + int(match.group("m")) * 60 + seconds)


def _subject_from_database_id(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"(\d+)$", str(value))
    if not match:
        return None
    return f"sub-{int(match.group(1)):02d}"


def _session_from_scan(value: Any) -> str | None:
    if value is None:
        return None
    return f"ses-{int(float(value)):02d}"


def _read_first_sheet_rows(path: Path) -> list[list[Any]]:
    with zipfile.ZipFile(path) as zf:
        shared = _read_shared_strings(zf)
        xml = zf.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(xml)
    rows: list[list[Any]] = []
    for row_el in root.findall(".//a:sheetData/a:row", NS):
        values: list[Any] = []
        for cell in row_el.findall("a:c", NS):
            col_index = _cell_column_index(cell.attrib["r"])
            while len(values) < col_index:
                values.append(None)
            values.append(_cell_value(cell, shared))
        rows.append(values)
    return rows


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        xml = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml)
    values: list[str] = []
    for item in root.findall("a:si", NS):
        values.append("".join(text.text or "" for text in item.findall(".//a:t", NS)))
    return values


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t")
    value_el = cell.find("a:v", NS)
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", NS))
    if value_el is None or value_el.text is None:
        return None
    raw = value_el.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    return int(numeric) if numeric.is_integer() else numeric


def _cell_column_index(reference: str) -> int:
    letters = re.match(r"^[A-Z]+", reference)
    if not letters:
        raise ValueError(f"Invalid cell reference: {reference}")
    index = 0
    for char in letters.group(0):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1
