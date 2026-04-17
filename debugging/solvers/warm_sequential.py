from benchopt import BaseSolver

import numpy as np

from torch.utils.data import DataLoader

from benchmark_utils.cache_control import readahead_cache


class Solver(BaseSolver):
    """Single-process DataLoader with fully warm page cache.

    Evicts then pre-loads every file via np.load to guarantee warm cache.
    This is the upper-bound baseline for file-backed datasets.
    """

    name = "Warm-Sequential"
    sampling_strategy = "run_once"

    parameters = {}

    requirements = ["numpy"]

    def set_objective(self, dataset, batch_size, file_paths):
        self.dataset = dataset
        self.batch_size = batch_size

        if file_paths:
            # Request readahead
            readahead_cache(file_paths)
            # Do a full sequential pass to truly warm the cache
            for fp in file_paths:
                try:
                    np.load(str(fp), mmap_mode=None)
                except Exception:
                    pass

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
