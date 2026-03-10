from benchopt import BaseDataset
from pathlib import Path
import tarfile
import tempfile
import urllib.request

from benchopt.config import get_data_path


class Dataset(BaseDataset):
    """Imagenette — freely available 10-class ImageNet subset.

    Downloaded from fast.ai's S3 bucket (no authentication required),
    extracted to a temp directory.  Pre-processing is excluded from
    throughput measurement.
    """

    name = "Imagenette"

    _URL = (
        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    )

    parameters = {
        "n_samples": [5_000],
        "image_size": [224],
        "split": ["train"],
    }

    requirements = ["Pillow"]

    def get_data(self):

        root = get_data_path("imagenette")  # get path for data

        if not root.exists():
            tmpdir = Path(tempfile.mkdtemp(prefix="benchopt_imagenette_"))

            tgz_path = tmpdir / "imagenette2-320.tgz"
            urllib.request.urlretrieve(self._URL, tgz_path)
            with tarfile.open(tgz_path) as tar:
                tar.extractall(root)
            tgz_path.unlink()

        split_dir = root / "imagenette2-320" / self.split
        all_paths = sorted(split_dir.rglob("*.JPEG"))
        file_paths = all_paths[: self.n_samples]

        return dict(file_paths=file_paths, image_size=self.image_size)
