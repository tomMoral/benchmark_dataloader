from benchopt import BaseObjective

import torch
from benchmark_utils import run_series_epoch_loop, compute_series_throughput


class Objective(BaseObjective):
    """Dataloader throughput for time series datasets.

    Series are windowed into fixed-length segments in set_objective.
    Primary metric: windows/sec (warm epochs).
    Also reports series/sec.
    """

    name = "Dataloader Throughput - TimeSeries"
    url = "https://github.com/tommoral/benchmark_dataloader/timeseries"
    min_benchopt_version = "1.8"
    sampling_strategy = "run_once"

    requirements = ["torch"]

    parameters = {
        "batch_size": [64, 256],
        "n_epochs": [4],
        "window_size": [512],
        "stride": [256],
    }

    def set_data(self, parquet_paths, n_channels):
        self.parquet_paths = parquet_paths
        self.n_channels = n_channels
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_objective(self):
        return dict(
            parquet_paths=self.parquet_paths,
            n_channels=self.n_channels,
            batch_size=self.batch_size,
            window_size=self.window_size,
            stride=self.stride,
            device=self.device,
        )

    def evaluate_result(self, dataloader):
        epoch_stats = run_series_epoch_loop(
            dataloader, self.n_epochs, self.device
        )
        return compute_series_throughput(epoch_stats)

    def get_one_result(self):
        # Minimal valid result for testing: fake loader with 10 batches.
        fake_batch = {
            "series": torch.zeros(64, self.window_size, 1),
            "n_series": 64,
        }
        return dict(dataloader=[fake_batch] * 10)
