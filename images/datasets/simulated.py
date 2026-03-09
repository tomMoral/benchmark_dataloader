from benchopt import BaseDataset
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


class Dataset(BaseDataset):
    """Simulated dataset: random JPEG images written to a temp directory."""

    name = "Simulated"

    parameters = {
        "n_samples, image_size": [
            (1000, 224),
            (5000, 224),
        ],
    }

    requirements = ["Pillow", "numpy"]

    def get_data(self):
        n_samples, image_size = self.n_samples, self.image_size
        tmpdir = Path(tempfile.mkdtemp(prefix="benchopt_img_"))
        rng = np.random.default_rng(0)

        for i in range(n_samples):
            arr = rng.integers(0, 255, (image_size, image_size, 3),
                               dtype=np.uint8)
            Image.fromarray(arr).save(tmpdir / f"img_{i:06d}.jpg", quality=85)

        return dict(
            file_paths=sorted(tmpdir.glob("*.jpg")),
            image_size=image_size,
        )
