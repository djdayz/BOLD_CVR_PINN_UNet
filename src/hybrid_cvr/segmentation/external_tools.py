from __future__ import annotations

import shutil
from pathlib import Path


def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found is not None:
        return found
    for directory in (Path("/Users/mac/fsl/share/fsl/bin"), Path("/usr/local/fsl/bin"), Path("/opt/fsl/bin")):
        candidate = directory / name
        if candidate.exists():
            return str(candidate)
    return None


def require_tool(name: str) -> str:
    path = find_tool(name)
    if path is None:
        raise RuntimeError(
            f"External neuroimaging tool '{name}' is not installed or not on PATH. "
            "Install it or use the documented test fallback only for synthetic checks."
        )
    return path
