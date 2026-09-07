from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib


def require_dependency(module: str, install_hint: str | None = None) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        hint = install_hint or module
        raise RuntimeError(f"Missing dependency '{module}'. Install with: {hint}") from exc


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml = require_dependency("yaml", "pip install pyyaml")
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
