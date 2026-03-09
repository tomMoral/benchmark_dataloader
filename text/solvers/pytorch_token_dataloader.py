from benchopt import BaseSolver

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_from_disk


class _TokenDataset(TorchDataset):
    """Thin wrapper to expose a HF Arrow dataset as a PyTorch Dataset."""

    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        return {k: torch.tensor(v) for k, v in row.items()}


class Solver(BaseSolver):
    """Standard torch.utils.data.DataLoader over pre-tokenized Arrow data."""

    name = "PyTorch-TokenDataLoader"
    sampling_strategy = "run_once"
    requirements = ["datasets", "torch"]

    parameters = {
        "num_workers": [4],
    }

    def set_objective(self, dataset_path, tokenizer_name, batch_size, device):
        self.device = device

        ds = load_from_disk(dataset_path)
        self.loader = DataLoader(
            _TokenDataset(ds),
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=self.num_workers > 0,
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(dataloader=self.loader)
