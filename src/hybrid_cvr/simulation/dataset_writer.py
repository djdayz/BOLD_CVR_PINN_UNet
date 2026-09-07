from __future__ import annotations

from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency


def write_npz_case(out: str | Path, **arrays: Any) -> Path:
    np = require_dependency("numpy", "pip install numpy")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return out
