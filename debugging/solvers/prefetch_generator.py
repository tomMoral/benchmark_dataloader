import threading

from benchopt import BaseSolver

from torch.utils.data import DataLoader

from benchmark_utils.cache_control import evict_cache, readahead_cache


class _PrefetchWrapper:
    """Wraps a DataLoader iterator, issuing POSIX_FADV_WILLNEED on upcoming
    batches' files in a background thread while the current batch is
    being consumed.
    """

    def __init__(self, loader, file_paths, batch_size, lookahead):
        self.loader = loader
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.lookahead = lookahead

    def __iter__(self):
        file_paths = self.file_paths
        bs = self.batch_size
        lookahead = self.lookahead

        # Pre-compute batch→file mapping
        n_files = len(file_paths)
        prefetch_thread = None

        for i, batch in enumerate(self.loader):
            # Issue readahead for the next `lookahead` batches' files
            start = (i + 1) * bs
            end = min((i + 1 + lookahead) * bs, n_files)
            if start < n_files and prefetch_thread is None or (
                prefetch_thread is not None
                and not prefetch_thread.is_alive()
            ):
                upcoming = file_paths[start:end]
                prefetch_thread = threading.Thread(
                    target=readahead_cache,
                    args=(upcoming,),
                    daemon=True,
                )
                prefetch_thread.start()

            yield batch


class Solver(BaseSolver):
    """Prefetch generator: cold start with lookahead readahead.

    Starts cold (POSIX_FADV_DONTNEED), then issues POSIX_FADV_WILLNEED
    on the next `lookahead` batches' files in a background thread while
    the current batch is being processed.
    """

    name = "Prefetch-Generator"
    sampling_strategy = "run_once"

    parameters = {
        "lookahead": [2],
    }

    def set_objective(self, dataset, batch_size, file_paths):
        self.dataset = dataset
        self.batch_size = batch_size
        self.file_paths = file_paths

        # Evict all dataset files from OS page cache — guaranteed cold start
        if self.file_paths:
            evict_cache(self.file_paths)

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
        if self.file_paths:
            wrapped = _PrefetchWrapper(
                self.loader, self.file_paths, self.batch_size, self.lookahead,
            )
            return dict(dataloader=iter(wrapped))
        return dict(dataloader=iter(self.loader))
