from benchopt import BaseObjective

import torch
from torchvision import transforms
from benchmark_utils import compute_throughput
from benchmark_utils import run_epoch_loop


class Objective(BaseObjective):
    """Dataloader throughput for image datasets.

    Metrics: samples/sec and pixels/sec (H*W*3), split into
    cold (epoch 1, OS page cache cold) and warm (epochs 2+).
    """

    name = "Dataloader Throughput - Images"
    url = "https://github.com/tommoral/benchmark_dataloader/image"
    min_benchopt_version = "1.8"

    requirements = ["torch", "torchvision"]

    # Disable performance curves: each solver runs once to completion.
    sampling_strategy = "run_once"

    parameters = {
        "batch_size": [64],
        "n_epochs": [4],
    }

    def set_data(self, file_paths, image_size):
        self.file_paths = file_paths
        self.image_size = image_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def get_objective(self):
        return dict(
            file_paths=self.file_paths,
            transform=self.transform,
            image_size=self.image_size,
            batch_size=self.batch_size,
            device=self.device,
        )

    def evaluate_result(self, dataloader):
        epoch_stats = run_epoch_loop(dataloader, self.n_epochs, self.device)
        return compute_throughput(epoch_stats, image_size=self.image_size)

    def get_one_result(self):
        # Minimal valid result for testing: 4 fake epochs.
        def dataloader():
            for _ in range(len(self.file_paths) // self.batch_size):
                yield torch.zeros(
                    (self.batch_size, 3, self.image_size, self.image_size)
                )
        return dict(dataloader=dataloader())
