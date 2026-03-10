# `benchmark_utils` is importable by objective, datasets, and solvers
# of this benchmark using standard import syntax:
#   from benchmark_utils import (
#       run_series_epoch_loop, compute_series_throughput)

import time
import torch


def run_series_epoch_loop(loader, n_epochs, device):
    """Run n_epochs over a loader yielding (series_tensor, metadata) pairs.

    Returns
    -------
    list of dict with keys: n_series, n_windows, elapsed_sec
    """
    is_cuda = device.type == "cuda"
    epoch_stats = []

    for _ in range(n_epochs):
        t0 = time.perf_counter()
        n_series = n_windows = 0
        for batch in loader:
            x = batch["series"].to(device, non_blocking=True)
            if is_cuda:
                torch.cuda.synchronize()
            n_series += batch["n_series"]
            n_windows += x.shape[0]
        epoch_stats.append(dict(
            n_series=n_series,
            n_windows=n_windows,
            elapsed_sec=time.perf_counter() - t0,
        ))

    return epoch_stats


def compute_series_throughput(epoch_stats):
    def _metrics(s):
        return (s["n_series"] / s["elapsed_sec"],
                s["n_windows"] / s["elapsed_sec"])

    cold_sr, cold_w = _metrics(epoch_stats[0])
    warm = epoch_stats[1:] or epoch_stats
    warm_sr = sum(s["n_series"] / s["elapsed_sec"] for s in warm) / len(warm)
    warm_w = sum(s["n_windows"] / s["elapsed_sec"] for s in warm) / len(warm)

    return dict(
        warm_series_per_sec=warm_sr,
        cold_series_per_sec=cold_sr,
        warm_windows_per_sec=warm_w,
        cold_windows_per_sec=cold_w,
        value=warm_w,  # primary metric: windows/sec (warm)
    )
