from benchopt import BaseObjective

import torch
from benchmark_utils import run_token_epoch_loop, compute_text_throughput


class Objective(BaseObjective):
    """Dataloader throughput for text datasets (pre-tokenized).

    Primary metric: tokens/sec (warm epochs).
    Also reports samples/sec, cold vs warm split.
    """

    name = "Dataloader Throughput - Text"
    url = "https://github.com/tommoral/benchmark_dataloader/text"
    min_benchopt_version = "1.8"
    sampling_strategy = "run_once"

    requirements = ["torch"]

    parameters = {
        "batch_size": [32, 128],
        "n_epochs": [4],
        "seq_len": [512],
    }

    def set_data(self, dataset_path, tokenizer_name):
        self.dataset_path = dataset_path
        self.tokenizer_name = tokenizer_name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_objective(self):
        return dict(
            dataset_path=self.dataset_path,
            tokenizer_name=self.tokenizer_name,
            batch_size=self.batch_size,
            device=self.device,
        )

    def evaluate_result(self, dataloader):
        epoch_stats = run_token_epoch_loop(
            dataloader, self.n_epochs, self.device
        )
        return compute_text_throughput(epoch_stats)

    def get_one_result(self):
        # Minimal valid result for testing: fake loader with 10 batches.
        fake_batch = {"input_ids": torch.zeros(32, 512, dtype=torch.long)}
        return dict(dataloader=[fake_batch] * 10)
