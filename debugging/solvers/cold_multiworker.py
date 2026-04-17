from benchopt import BaseSolver

from torch.utils.data import DataLoader

from benchmark_utils.cache_control import evict_cache


class Solver(BaseSolver):
    """DataLoader with multiple workers after cache eviction.

    Parametrized over num_workers and pin_memory to measure the impact
    of parallelism and memory pinning on cold-start throughput.
    """

    name = "Cold-MultiWorker"
    sampling_strategy = "run_once"

    parameters = {
        "num_workers": [1, 2, 4, 8],
        "pin_memory": [False, True],
    }

    def set_objective(self, dataset, batch_size, file_paths):
        self.dataset = dataset
        self.batch_size = batch_size

        # Evict all dataset files from OS page cache
        if file_paths:
            evict_cache(file_paths)

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2,
            persistent_workers=True,
            shuffle=False,
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(dataloader=iter(self.loader))
