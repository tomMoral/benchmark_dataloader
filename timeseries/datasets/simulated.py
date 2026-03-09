from benchopt import BaseDataset
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


class Dataset(BaseDataset):
    """Simulated multivariate time series saved as local parquet files."""

    name = "Simulated"

    parameters = {
        "n_series, series_len, n_channels": [
            (100, 10_000, 7),
            (500, 10_000, 7),
        ],
    }

    requirements = ["pandas", "pyarrow", "numpy"]

    def get_data(self):
        tmpdir = Path(tempfile.mkdtemp(prefix="benchopt_ts_"))
        rng = np.random.default_rng(0)

        for i in range(self.n_series):
            data = rng.standard_normal((self.series_len, self.n_channels))
            cols = {f"channel_{j}": data[:, j] for j in range(self.n_channels)}
            cols["timestamp"] = np.arange(self.series_len)
            pd.DataFrame(cols).to_parquet(
                tmpdir / f"series_{i:04d}.parquet", index=False
            )

        return dict(
            parquet_paths=sorted(str(p) for p in tmpdir.glob("*.parquet")),
            n_channels=self.n_channels,
        )
