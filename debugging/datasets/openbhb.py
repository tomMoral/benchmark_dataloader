from benchopt import BaseDataset
from benchopt.config import get_data_path


class Dataset(BaseDataset):
    """OpenBHB neuroimaging dataset via nidl.

    Wraps `nidl.datasets.OpenBHB` with `streaming=False` (data must be
    pre-downloaded). The dataset exposes `.samples` — a list of dicts
    containing at least a `"path"` key pointing to .npy files.
    """

    name = "OpenBHB"

    parameters = {
        "modality": ["quasiraw"],
        "split": ["train"],
        "batch_size": [32],
    }

    install_cmd = "conda"
    requirements = ["pip::nidl"]

    def get_data(self):
        from nidl.datasets import OpenBHB

        root = get_data_path("openbhb")

        dataset = OpenBHB(
            root=root,
            modality=self.modality,
            split=self.split,
            streaming=False,
        )

        # Expose backing file paths for cache control in solvers.
        file_paths = []
        if hasattr(dataset, "samples"):
            file_paths = [
                s["path"] if isinstance(s, dict) else s
                for s in dataset.samples
            ]

        return dict(
            dataset=dataset,
            batch_size=self.batch_size,
            file_paths=file_paths,
        )
