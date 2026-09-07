from __future__ import annotations

from typing import Any

from hybrid_cvr.config import require_dependency


def sample_joint_rows(table: Any, region: str, n: int, columns: tuple[str, ...] = ("CVR", "delay", "T")) -> Any:
    np = require_dependency("numpy", "pip install numpy")
    rows = table[table["region"] == region]
    if len(rows) == 0:
        rows = table
    if len(rows) == 0:
        raise ValueError("Cannot sample from an empty distribution table")
    idx = np.random.default_rng().integers(0, len(rows), size=n)
    return rows.iloc[idx][list(columns)].to_numpy()
