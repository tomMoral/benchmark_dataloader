from benchopt import BaseDataset
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from datasets import load_dataset


class Dataset(BaseDataset):
    """GIFT-eval time series streamed from HuggingFace (Salesforce/gift_eval).

    Each config name selects a specific dataset and split embedded
    together, e.g. 'ETTh1/train'.  Pre-processing (download, parquet
    conversion) is excluded from throughput measurement.
    """

    name = "GIFT-Eval"

    parameters = {
        "dataset_name": ["ETTh1/train"],
        "n_series": [100],
    }

    requirements = ["pandas", "pyarrow", "numpy", "datasets"]

    def get_data(self):
        cache_dir = Path(tempfile.mkdtemp(prefix="benchopt_gift_"))

        ds = load_dataset(
            "Salesforce/GiftEval",
            split="train",
            streaming=True,
        )

        parquet_paths = []
        n_channels = None
        for i, example in enumerate(ds):
            if i >= self.n_series:
                break
            target = np.array(example["target"], dtype=np.float32)
            # Univariate: (T,) → (T, 1); multivariate: (C, T) → (T, C)
            if target.ndim == 1:
                target = target[:, np.newaxis]
            else:
                target = target.T
            if n_channels is None:
                n_channels = target.shape[1]

            cols = {f"channel_{j}": target[:, j]
                    for j in range(target.shape[1])}
            cols["timestamp"] = np.arange(len(target))
            path = cache_dir / f"series_{i:04d}.parquet"
            pd.DataFrame(cols).to_parquet(path, index=False)
            parquet_paths.append(str(path))

        return dict(
            parquet_paths=parquet_paths,
            n_channels=n_channels,
        )
