import numpy as np

from benchopt import BaseDataset

from torch.utils.data import Dataset as TorchDataset


class _SyntheticDataset(TorchDataset):
    """In-memory dataset of random tensors — no I/O at read time."""

    def __init__(self, n_samples, shape, dtype=np.float32):
        self.data = np.random.default_rng(42).standard_normal(
            (n_samples, *shape),
        ).astype(dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import torch
        return torch.from_numpy(self.data[idx])


class Dataset(BaseDataset):
    """Synthetic in-memory dataset.

    Generates random tensors with the same shape as a target modality
    (default: quasiraw 3-D brain volume). This is an I/O-free baseline:
    cold/warm ratio should be ~1.0, establishing the upper bound on
    DataLoader throughput.
    """

    name = "Synthetic"

    parameters = {
        "n_samples": [2000],
        "shape": [(1, 64, 64, 64)],
        "batch_size": [32],
    }

    requirements = ["numpy"]

    def get_data(self):

        dataset = _SyntheticDataset(
            self.n_samples, self.shape,
        )

        return dict(
            dataset=dataset,
            batch_size=self.batch_size,
            file_paths=[],
        )
