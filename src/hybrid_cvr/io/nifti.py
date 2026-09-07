from __future__ import annotations

from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency


def load_nifti(path: str | Path) -> tuple[Any, Any]:
    nib = require_dependency("nibabel", "pip install nibabel")
    img = nib.load(str(path))
    return img.get_fdata(dtype="float32"), img


def save_like(data: Any, reference_img: Any, out: str | Path) -> Path:
    nib = require_dependency("nibabel", "pip install nibabel")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    nib.save(img, str(out))
    return out
