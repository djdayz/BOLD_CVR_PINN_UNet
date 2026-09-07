from __future__ import annotations

from pathlib import Path
from typing import Any

from hybrid_cvr.config import require_dependency


def save_map_png(map_2d: Any, out: str | Path, title: str = "") -> Path:
    plt = require_dependency("matplotlib.pyplot", "pip install matplotlib")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(map_2d, cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
