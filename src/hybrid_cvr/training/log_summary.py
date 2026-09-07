from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DISPLAY_COLUMNS = (
    "epoch",
    "stage",
    "stage_epoch",
    "T_mode",
    "learning_rate",
    "train_total",
    "val_total",
    "val_recon",
    "val_residual",
    "val_uncertainty",
    "val_mean_cvr",
    "val_mean_delay",
    "val_mean_T",
    "val_mean_sigma",
    "epoch_time_seconds",
    "gpu_memory_mb",
)


def summarize_training_logs(model_dir: str | Path) -> dict[str, Path | int | str]:
    model_dir = Path(model_dir)
    history_path = model_dir / "training_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing training history: {history_path}")
    rows = _read_rows(history_path)
    if not rows:
        raise ValueError(f"No rows found in {history_path}")

    logs_dir = model_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    readable_csv = logs_dir / "training_history_readable.csv"
    markdown = logs_dir / "training_summary.md"
    latest_txt = logs_dir / "latest_status.txt"
    stage_json = logs_dir / "stage_history_pretty.json"

    compact_rows = [_compact_row(row) for row in rows]
    _write_csv(readable_csv, compact_rows)
    _write_markdown(markdown, compact_rows)
    _write_latest(latest_txt, compact_rows)
    _write_stage_json(stage_json, compact_rows)
    return {
        "history": history_path,
        "readable_csv": readable_csv,
        "summary_markdown": markdown,
        "latest_status": latest_txt,
        "stage_json": stage_json,
        "epochs": len(rows),
        "latest_stage": str(rows[-1].get("stage", "")),
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _compact_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in DISPLAY_COLUMNS:
        value = row.get(column, "")
        if column in {"epoch", "stage_epoch"}:
            out[column] = int(float(value)) if value else ""
        elif column in {"stage", "T_mode"}:
            out[column] = value
        else:
            out[column] = _round_float(value)
    return out


def _round_float(value: str) -> float | str:
    if value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if abs(number) >= 100:
        return round(number, 2)
    if abs(number) >= 10:
        return round(number, 3)
    return round(number, 4)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    best = min(rows, key=lambda row: float(row["val_total"]))
    recent = rows[-20:]
    lines = [
        "# Training Summary",
        "",
        f"- Epochs completed: {len(rows)}",
        f"- Current stage: {rows[-1]['stage']}",
        f"- Best validation loss: {best['val_total']} at epoch {best['epoch']} ({best['stage']})",
        f"- Latest validation loss: {rows[-1]['val_total']}",
        "",
        "## Recent Epochs",
        "",
        _markdown_table(recent),
        "",
        "## Best Epoch By Stage",
        "",
        _markdown_table(_best_rows_by_stage(rows)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_latest(path: Path, rows: list[dict[str, Any]]) -> None:
    latest = rows[-1]
    best = min(rows, key=lambda row: float(row["val_total"]))
    text = (
        f"latest_epoch={latest['epoch']}\n"
        f"latest_stage={latest['stage']}\n"
        f"latest_stage_epoch={latest['stage_epoch']}\n"
        f"latest_train_total={latest['train_total']}\n"
        f"latest_val_total={latest['val_total']}\n"
        f"best_epoch={best['epoch']}\n"
        f"best_stage={best['stage']}\n"
        f"best_val_total={best['val_total']}\n"
    )
    path.write_text(text, encoding="utf-8")


def _write_stage_json(path: Path, rows: list[dict[str, Any]]) -> None:
    stage_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stage_rows.setdefault(str(row["stage"]), []).append(row)
    payload = {}
    for stage, items in stage_rows.items():
        best = min(items, key=lambda row: float(row["val_total"]))
        latest = items[-1]
        payload[stage] = {
            "epochs_completed": len(items),
            "best_epoch": best["epoch"],
            "best_val_total": best["val_total"],
            "latest_epoch": latest["epoch"],
            "latest_val_total": latest["val_total"],
        }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _best_rows_by_stage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_stage: dict[str, dict[str, Any]] = {}
    for row in rows:
        stage = str(row["stage"])
        if stage not in best_by_stage or float(row["val_total"]) < float(best_by_stage[stage]["val_total"]):
            best_by_stage[stage] = row
    return list(best_by_stage.values())


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "epoch",
        "stage",
        "stage_epoch",
        "T_mode",
        "train_total",
        "val_total",
        "val_recon",
        "val_residual",
        "val_mean_cvr",
        "val_mean_delay",
        "val_mean_T",
    ]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])
