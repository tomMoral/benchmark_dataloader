from benchopt import BaseSolver
import shutil
import tarfile
import tempfile
from pathlib import Path

import webdataset as wds


class Solver(BaseSolver):
    """WebDataset: sharded tar streaming."""

    name = "WebDataset"
    sampling_strategy = "run_once"
    requirements = ["webdataset"]

    parameters = {
        "num_workers": [4],
    }

    def set_objective(
            self, file_paths, transform, image_size, batch_size, device
    ):
        file_paths = list(file_paths)
        self.device = device
        self.batch_size = batch_size
        self.n_samples = len(file_paths)

        # Pack into sharded tars — excluded from throughput measurement.
        self._tmpdir = Path(tempfile.mkdtemp(prefix="benchopt_wds_"))
        shard_paths = []
        min_shard = (self.n_samples + self.num_workers - 1) // self.num_workers
        step = min(1000, min_shard)
        for i, start in enumerate(range(0, self.n_samples, step)):
            shard = self._tmpdir / f"shard-{i:04d}.tar"
            with tarfile.open(shard, "w") as tar:
                for fp in file_paths[start:start + step]:
                    tar.add(fp, arcname=f"{Path(fp).stem}.jpg")
            shard_paths.append(str(shard))

        dataset = (
            wds.WebDataset(shard_paths, shardshuffle=False)
            .decode("pil")
            .to_tuple("jpg")
            .map_tuple(transform)
            .batched(batch_size, partial=True)
        )

        self.loader = wds.WebLoader(
            dataset, num_workers=self.num_workers, batch_size=None
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(dataloader=self.loader)

    def __del__(self):
        if hasattr(self, "_tmpdir") and self._tmpdir.exists():
            shutil.rmtree(self._tmpdir, ignore_errors=True)
