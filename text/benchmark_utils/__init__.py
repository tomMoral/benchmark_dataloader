# `benchmark_utils` is importable by objective, datasets, and solvers
# of this benchmark using standard import syntax:
#   from benchmark_utils import run_token_epoch_loop, compute_text_throughput

import time
import torch


def run_token_epoch_loop(loader, n_epochs, device):
    """Run n_epochs over a loader of pre-tokenized batches (dict with
    'input_ids' key), moving to device.

    Returns
    -------
    list of dict with keys: n_samples, n_tokens, elapsed_sec
    """
    is_cuda = device.type == "cuda"
    epoch_stats = []

    for _ in range(n_epochs):
        t0 = time.perf_counter()
        n_samples = n_tokens = 0
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            if is_cuda:
                torch.cuda.synchronize()
            n_samples += ids.shape[0]
            n_tokens += ids.numel()
        epoch_stats.append(dict(
            n_samples=n_samples,
            n_tokens=n_tokens,
            elapsed_sec=time.perf_counter() - t0,
        ))

    return epoch_stats


def compute_text_throughput(epoch_stats):
    """Return cold/warm samples/sec and tokens/sec."""
    def _metrics(s):
        return (s["n_samples"] / s["elapsed_sec"],
                s["n_tokens"] / s["elapsed_sec"])

    cold_s, cold_t = _metrics(epoch_stats[0])
    warm_stats = epoch_stats[1:] or epoch_stats  # fallback for n_epochs=1
    warm_s = (sum(s["n_samples"] / s["elapsed_sec"] for s in warm_stats)
              / len(warm_stats))
    warm_t = (sum(s["n_tokens"] / s["elapsed_sec"] for s in warm_stats)
              / len(warm_stats))

    return dict(
        cold_samples_per_sec=cold_s,
        warm_samples_per_sec=warm_s,
        cold_tokens_per_sec=cold_t,
        warm_tokens_per_sec=warm_t,
        value=warm_t,  # primary metric: tokens/sec (warm)
    )
