from benchopt import BaseDataset
from pathlib import Path
import os

import pandas as pd


class Dataset(BaseDataset):
    """GIFT-eval time series dataset from local parquet/arrow files.

    Expects GIFT_EVAL_PATH env var pointing to a directory containing
    .parquet files, one per time series group.

    Each parquet file should have columns:
    [timestamp, channel_0, ..., channel_N]
    """

    name = "GIFT-Eval"

    parameters = {
        "split": ["train"],
        "max_series": [500],
    }

    requirements = ["pandas", "pyarrow"]

    def get_data(self):
        root = Path(os.environ.get("GIFT_EVAL_PATH", "/data/gift_eval"))

        if not root.exists():
            raise FileNotFoundError(
                f"GIFT-eval not found at {root}. "
                "Set the GIFT_EVAL_PATH environment variable."
            )

        parquet_paths = sorted(root.glob(f"{self.split}/*.parquet"))
        if not parquet_paths:
            # flat layout fallback
            parquet_paths = sorted(root.glob("*.parquet"))

        parquet_paths = parquet_paths[: self.max_series]

        # Infer n_channels from the first file.
        df = pd.read_parquet(parquet_paths[0])
        n_channels = len([c for c in df.columns if c != "timestamp"])

        return dict(
            parquet_paths=[str(p) for p in parquet_paths],
            n_channels=n_channels,
        )
