from benchopt import BaseDataset

from benchopt.config import get_data_path


class Dataset(BaseDataset):
    """ImageNet dataset (validation split).

    Requires IMAGENET_PATH env var pointing to a directory with
    JPEG files (flat or class-subfolder layout).
    """

    name = "ImageNet"

    parameters = {
        "split": ["val"],
        "image_size": [224],
        "max_samples": [50000],
    }

    requirements = ["Pillow"]

    def get_data(self):
        root = get_data_path("imagenet")
        split_dir = root / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"ImageNet not found at {split_dir}. "
                "You need to download it yourself and specify its path "
                "in the benchmark config (see README)."
            )

        file_paths = sorted(split_dir.rglob("*.JPEG"))
        file_paths += sorted(split_dir.rglob("*.jpg"))

        if self.max_samples is not None:
            file_paths = file_paths[: self.max_samples]

        return dict(file_paths=file_paths, image_size=self.image_size)
