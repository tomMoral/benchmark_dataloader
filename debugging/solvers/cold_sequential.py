from benchopt import BaseSolver

from torch.utils.data import DataLoader

from benchmark_utils.cache_control import evict_cache


class Solver(BaseSolver):
    """Single-process DataLoader after evicting page cache.

    This is the worst-case baseline — sequential reads with no OS cache.
    """

    name = "Cold-Sequential"
    sampling_strategy = "run_once"

    parameters = {}

    def set_objective(self, dataset, batch_size, file_paths):
        self.dataset = dataset
        self.batch_size = batch_size

        # Evict all dataset files from OS page cache
        if file_paths:
            evict_cache(file_paths)

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(dataloader=iter(self.loader))
