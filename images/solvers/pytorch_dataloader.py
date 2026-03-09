from benchopt import BaseSolver

from PIL import Image
from torch.utils.data import DataLoader, Dataset as TorchDataset


class _ImageDataset(TorchDataset):
    def __init__(self, file_paths, transform):
        self.file_paths = list(file_paths)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.file_paths[idx]).convert("RGB"))


class Solver(BaseSolver):
    """Baseline: torch.utils.data.DataLoader with standard transforms."""

    name = "PyTorch-DataLoader"
    sampling_strategy = "run_once"
    requirements = ["Pillow"]

    parameters = {
        "num_workers": [4],
    }

    def set_objective(
            self, file_paths, transform, image_size, batch_size, device
    ):
        self.device = device
        self.loader = DataLoader(
            _ImageDataset(file_paths, transform),
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(dataloader=self.loader)
