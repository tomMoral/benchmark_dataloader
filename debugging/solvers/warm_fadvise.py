from benchopt import BaseSolver

from torch.utils.data import DataLoader

from benchmark_utils.cache_control import evict_cache, readahead_cache


class Solver(BaseSolver):
    """Warm fadvise: evict then request async readahead, no blocking warmup.

    Tests whether POSIX_FADV_WILLNEED alone (without a full np.load pass)
    is enough to warm the cache via asynchronous kernel readahead.
    """

    name = "Warm-Fadvise"
    sampling_strategy = "run_once"

    parameters = {}

    def set_objective(self, dataset, batch_size, file_paths):
        self.dataset = dataset
        self.batch_size = batch_size

        if file_paths:
            # Evict to start from a clean state
            evict_cache(file_paths)
            # Request async readahead — the kernel may or may not finish
            # before the objective starts iterating
            readahead_cache(file_paths)

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
