from benchopt import BaseSolver

from datasets import load_from_disk
from torch.utils.data import DataLoader


class Solver(BaseSolver):
    """Load pre-tokenized Arrow dataset via HuggingFace datasets + DataLoader.
    """

    name = "HuggingFace-Arrow"
    sampling_strategy = "run_once"
    requirements = ["datasets", "torch"]

    parameters = {
        "num_workers": [4],
    }

    def set_objective(self, dataset_path, tokenizer_name, batch_size, device):
        self.device = device

        ds = load_from_disk(dataset_path)
        ds.set_format("torch", columns=["input_ids", "attention_mask"])

        self.loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=self.num_workers > 0,
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(dataloader=self.loader)
