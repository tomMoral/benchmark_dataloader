from benchopt import BaseSolver

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
import pandas as pd
import numpy as np


class _WindowDataset(TorchDataset):
    """Slice parquet files into fixed-length windows on-the-fly."""

    def __init__(self, parquet_paths, window_size, stride, n_channels):
        self.windows = []  # list of (path, start_idx)
        self.n_series = len(parquet_paths)

        for path in parquet_paths:
            df = pd.read_parquet(path)
            length = len(df)
            starts = range(0, length - window_size + 1, stride)
            self.windows.extend((path, s) for s in starts)

        self.window_size = window_size
        self.n_channels = n_channels
        self._cache = {}  # simple path → array cache

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        path, start = self.windows[idx]
        if path not in self._cache:
            df = pd.read_parquet(path)
            cols = [c for c in df.columns if c != "timestamp"]
            self._cache[path] = df[cols].values.astype(np.float32)
        arr = self._cache[path][start: start + self.window_size]
        return torch.from_numpy(arr)  # (window_size, n_channels)


def _collate(batch):
    series = torch.stack(batch)  # (B, window_size, n_channels)
    return {"series": series, "n_series": len(batch)}


class Solver(BaseSolver):
    """Pandas parquet reader with windowed PyTorch DataLoader."""

    name = "Pandas-Parquet"
    sampling_strategy = "run_once"
    requirements = ["pandas", "pyarrow"]

    parameters = {
        "num_workers": [4],
    }

    def set_objective(self, parquet_paths, n_channels, batch_size,
                      window_size, stride, device):
        self.device = device

        dataset = _WindowDataset(
            parquet_paths, window_size, stride, n_channels
        )
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=(device.type == "cuda"),
            persistent_workers=self.num_workers > 0,
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(dataloader=self.loader)
