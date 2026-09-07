from __future__ import annotations

from pathlib import Path


class SimulatedSliceDataset(__import__("torch").utils.data.Dataset):
    def __init__(self, root: str | Path):
        np = __import__("numpy")
        self.paths = sorted(Path(root).glob("*.npz"))
        self.np = np

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        torch = __import__("torch")
        data = self.np.load(self.paths[idx])
        return {key: torch.as_tensor(data[key]).float() for key in data.files if not key.startswith("GT_")}
