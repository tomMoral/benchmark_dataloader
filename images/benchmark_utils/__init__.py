# `benchmark_utils` is importable by objective, datasets, and solvers
# of this benchmark using standard import syntax:
#   from benchmark_utils import run_epoch_loop, compute_throughput

import time
import torch


def run_epoch_loop(loader, n_epochs, device, unpack_fn=None):
    """Run `n_epochs` over `loader`, moving batches to `device`.

    Parameters
    ----------
    loader : iterable
        Any iterable yielding batches.
    n_epochs : int
    device : torch.device
    unpack_fn : callable, optional
        Called on each batch before sending to device.
        Defaults to identity (batch is already a tensor).

    Returns
    -------
    epoch_stats : list of dict
        One dict per epoch with keys `n_samples` and `elapsed_sec`.
    """
    is_cuda = device.type == "cuda"
    epoch_stats = []

    for _ in range(n_epochs):
        t0 = time.perf_counter()
        n_samples = 0
        for batch in loader:
            if unpack_fn is not None:
                batch = unpack_fn(batch)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device, non_blocking=True)
            if is_cuda:
                torch.cuda.synchronize()
            n_samples += batch.shape[0]
        epoch_stats.append(dict(n_samples=n_samples,
                                elapsed_sec=time.perf_counter() - t0))

    return epoch_stats


def compute_throughput(epoch_stats, image_size=None):
    """Compute cold/warm throughput metrics from epoch_stats.

    Parameters
    ----------
    epoch_stats : list of dict with keys `n_samples`, `elapsed_sec`
    image_size : int or None
        If provided, also compute pixels/sec (n_samples * image_size^2 * 3).

    Returns
    -------
    dict with cold_samples_per_sec, warm_samples_per_sec,
    and optionally cold_pixels_per_sec, warm_pixels_per_sec.
    """
    def _throughput(stats):
        return stats["n_samples"] / stats["elapsed_sec"]

    cold = _throughput(epoch_stats[0])
    warm = sum(
        _throughput(s) for s in epoch_stats[1:]
    ) / max(len(epoch_stats) - 1, 1)

    result = dict(
        cold_samples_per_sec=cold,
        warm_samples_per_sec=warm,
        value=warm,  # primary metric for benchopt plots
    )

    if image_size is not None:
        n_pixels = image_size * image_size * 3
        result["cold_pixels_per_sec"] = cold * n_pixels
        result["warm_pixels_per_sec"] = warm * n_pixels

    return result
