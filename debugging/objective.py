from benchopt import BaseObjective

import os
import time
from pathlib import Path

import torch


def _detect_fs_type(path):
    """Detect the filesystem type for the given path by parsing /proc/mounts.

    Returns a string like 'ext4', 'nfs', 'tmpfs', 'lustre', etc.
    Falls back to 'unknown' if detection fails.
    """
    path = os.path.realpath(path)
    best_mount = ""
    best_fs = "unknown"
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point, fs_type = parts[1], parts[2]
                if path.startswith(mount_point) and len(mount_point) > len(
                    best_mount
                ):
                    best_mount = mount_point
                    best_fs = fs_type
    except OSError:
        pass
    return best_fs


class Objective(BaseObjective):
    """DataLoader throughput diagnostics.

    Iterates over a DataLoader for a fixed number of batches, recording
    per-batch timings. Reports throughput, median/p95 batch times,
    and filesystem metadata.
    """

    name = "DataLoader Diagnostics"
    url = "https://github.com/tommoral/benchmark_dataloader/debugging"
    min_benchopt_version = "1.8"

    requirements = ["pytorch"]

    # Each solver runs once to completion (no performance curves).
    sampling_strategy = "run_once"

    parameters = {
        "n_batches": [200],
    }

    def set_data(self, dataset, batch_size, data_root, file_paths):
        self.dataset = dataset
        self.batch_size = batch_size
        self.file_paths = file_paths
        self.fs_type = (
            _detect_fs_type(Path(file_paths[0]).parent)
            if file_paths else "N/A"
        )

    def get_objective(self):
        return dict(
            dataset=self.dataset,
            batch_size=self.batch_size,
            file_paths=self.file_paths,
        )

    def evaluate_result(self, dataloader):
        batch_times = []
        batch_sizes = []

        # Time iterator creation separately — this is where workers spawn.
        t_init_start = time.perf_counter()
        it = iter(dataloader)
        init_time = time.perf_counter() - t_init_start

        for i in range(self.n_batches):
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            # Unpack if batch is a tuple/list (e.g. (data, label))
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to("cpu", non_blocking=False)
            elapsed = time.perf_counter() - t0
            batch_times.append(elapsed)
            batch_sizes.append(batch.shape[0])

        if not batch_times:
            return dict(
                value=0.0,
                batch_idx=0,
                batch_time_ms=float("inf"),
                throughput_samples_per_sec=0.0,
                throughput_excl_init_samples_per_sec=0.0,
                median_batch_time_ms=float("inf"),
                p95_batch_time_ms=float("inf"),
                init_time_ms=init_time * 1000,
                n_batches_measured=0,
                fs_type=self.fs_type,
            )

        # Compute aggregate stats once
        n_samples = sum(batch_sizes)
        total_time = sum(batch_times)
        batch_times_ms = [t * 1000 for t in batch_times]
        batch_times_ms_sorted = sorted(batch_times_ms)
        n = len(batch_times_ms_sorted)
        median = batch_times_ms_sorted[n // 2]
        p95 = batch_times_ms_sorted[int(n * 0.95)]
        p99 = batch_times_ms_sorted[int(n * 0.99)]

        # Full throughput (includes first batch which absorbs remaining
        # init cost not captured by iter()).
        throughput = n_samples / total_time if total_time > 0 else 0.0

        # Throughput excluding batch 0 — steady-state throughput once
        # workers are running.
        if n > 1:
            ss_samples = sum(batch_sizes[1:])
            ss_time = sum(batch_times[1:])
            throughput_excl_init = (
                ss_samples / ss_time if ss_time > 0 else 0.0
            )
        else:
            throughput_excl_init = throughput

        # Stall analysis: batches > 5x median
        stall_threshold = median * 5
        n_stalls = sum(1 for t in batch_times_ms if t > stall_threshold)
        stall_fraction = n_stalls / n if n > 0 else 0.0

        # First 10% vs last 10% batch times — detect warmup effect
        first_10 = batch_times_ms[: max(1, n // 10)]
        last_10 = batch_times_ms[-max(1, n // 10):]
        first_10_median = sorted(first_10)[len(first_10) // 2]
        last_10_median = sorted(last_10)[len(last_10) // 2]
        warmup_ratio = (
            first_10_median / last_10_median if last_10_median > 0 else 1.0
        )

        init_time_ms = init_time * 1000

        # Shared aggregate fields, repeated on every row
        shared = dict(
            throughput_samples_per_sec=throughput,
            throughput_excl_init_samples_per_sec=throughput_excl_init,
            init_time_ms=init_time_ms,
            median_batch_time_ms=median,
            p95_batch_time_ms=p95,
            p99_batch_time_ms=p99,
            min_batch_time_ms=batch_times_ms_sorted[0],
            max_batch_time_ms=batch_times_ms_sorted[-1],
            total_time_sec=total_time,
            n_batches_measured=n,
            n_samples_total=n_samples,
            batch_size=self.batch_size,
            stall_count=n_stalls,
            stall_fraction=stall_fraction,
            stall_threshold_ms=stall_threshold,
            warmup_ratio=warmup_ratio,
            first_10pct_median_ms=first_10_median,
            last_10pct_median_ms=last_10_median,
            fs_type=self.fs_type,
        )

        # Return one row per batch (multiple-evaluation pattern).
        results = []
        for idx, (bt_ms, bs) in enumerate(zip(batch_times_ms, batch_sizes)):
            results.append(dict(
                value=-throughput,
                batch_idx=idx,
                batch_time_ms=bt_ms,
                is_stall=int(bt_ms > stall_threshold),
                **shared,
            ))
        return results

    def get_one_result(self):
        # Minimal valid result for testing.
        loader = [
            torch.zeros((self.batch_size, 1, 4, 4))
            for _ in range(self.n_batches)
        ]
        return dict(dataloader=iter(loader))
