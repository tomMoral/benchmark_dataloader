from benchopt import BaseSolver

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
import pyarrow.parquet as pq
import numpy as np


class _ArrowWindowDataset(TorchDataset):
    """Pre-load all series as a single Arrow table, then window it."""

    def __init__(self, parquet_paths, window_size, stride, n_channels):
        # Concatenate all parquet files into one Arrow table in set_objective.
        tables = [pq.read_table(p) for p in parquet_paths]
        self.n_series = len(tables)

        self.windows = []  # list of (array, start)
        for table in tables:
            cols = [c for c in table.schema.names if c != "timestamp"]
            arr = table.select(cols).to_pydict()
            mat = np.stack([arr[c] for c in cols], axis=1).astype(np.float32)
            length = len(mat)
            for start in range(0, length - window_size + 1, stride):
                self.windows.append((mat, start))

        self.window_size = window_size

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        mat, start = self.windows[idx]
        return torch.from_numpy(mat[start: start + self.window_size])


def _collate(batch):
    return {"series": torch.stack(batch), "n_series": len(batch)}


class Solver(BaseSolver):
    """PyArrow: load all parquet into memory as Arrow tables, then window."""

    name = "PyArrow"
    sampling_strategy = "run_once"
    requirements = ["pyarrow", "torch", "numpy"]

    def set_objective(self, parquet_paths, n_channels, batch_size,
                      window_size, stride, device):
        self.device = device

        dataset = _ArrowWindowDataset(
            parquet_paths, window_size, stride, n_channels
        )
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,  # in memory already; workers add overhead
            collate_fn=_collate,
            pin_memory=(device.type == "cuda"),
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(dataloader=self.loader)
