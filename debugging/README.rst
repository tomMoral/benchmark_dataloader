Benchmark: DataLoader Diagnostics
====================================

This benchopt benchmark automatically diagnoses DataLoader throughput
bottlenecks for a PyTorch dataset by comparing different cache states
and loading strategies.

What is measured
----------------

The objective iterates over a DataLoader for a fixed number of batches
(no model forward pass, CPU only). For each batch it records:

- **Throughput** (samples/second) — primary metric
- **Median batch time** (ms)
- **p95 / p99 batch time** (ms) — reveals occasional stalls
- **Stall count** — batches exceeding 5× the median time
- **Warmup ratio** — compares first 10% vs last 10% of batch times
- **Filesystem type** — auto-detected from ``/proc/mounts``

Datasets
--------

- **Synthetic**: in-memory random tensors matching a target shape.
  I/O-free upper bound — cold/warm ratio ≈ 1.0.
- **OpenBHB**: ``nidl.datasets.OpenBHB`` wrapping ``.npy`` files on disk.

Solvers (strategies)
---------------------

Each solver controls the OS page cache state **before** the objective runs,
using ``posix_fadvise`` via ctypes:

1. **Cold-Sequential** — evicts file pages (``POSIX_FADV_DONTNEED``),
   then reads with ``num_workers=0``. Worst-case baseline.
2. **Warm-Sequential** — evicts, requests readahead, does a full ``np.load``
   pass. Upper-bound baseline for file-backed datasets.
3. **Cold-MultiWorker** — cold start with ``num_workers`` in {1,2,4,8}
   and ``pin_memory`` True/False. Measures parallelism impact.
4. **Warm-Fadvise** — evict then ``POSIX_FADV_WILLNEED`` only (no blocking
   warm pass). Tests whether async readahead suffices.
5. **Prefetch-Generator** — cold start, background thread issues
   ``POSIX_FADV_WILLNEED`` on the next ``lookahead`` batches' files.

Interpreting results
--------------------

- **Cold/warm ratio**: compare Cold-Sequential vs Warm-Sequential throughput.
  A ratio ≪ 1 indicates a storage bottleneck (NFS, Lustre). A ratio ≈ 1
  means local NVMe or in-memory data.
- **Stall fraction**: high stall fraction in early batches suggests the
  page cache is not yet warm — readahead or prefetching may help.
- **MultiWorker scaling**: if 4 workers ≫ 1 worker, the bottleneck is
  CPU-bound (decoding/transforms). If scaling is flat, the bottleneck is I/O.

Note: results are storage-dependent. Local NVMe, NFS, and Lustre will
show very different profiles.

Running
-------

.. code-block:: bash

   cd debugging
   benchopt run . --n-repetitions 3

To run only the synthetic dataset (fast smoke test):

.. code-block:: bash

   benchopt run . -d Synthetic --n-repetitions 1
