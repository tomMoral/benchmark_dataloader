from benchopt import BaseDataset
from pathlib import Path
import tempfile

from datasets import load_dataset, Dataset as HFDataset
from tokenizers import Tokenizer


class Dataset(BaseDataset):
    """FineWeb (HuggingFace) — pre-tokenized and saved to local Arrow cache.

    Pre-tokenization happens in get_data() and is excluded from
    the throughput measurement.
    """

    name = "FineWeb"

    requirements = [
        "datasets", "pip::tokenizers>=0.15", "pip::safetensors>=0.5.0"
    ]

    parameters = {
        "n_samples": [10_000, 100_000],
        "tokenizer_name": ["gpt2"],
        "seq_len": [512],
    }

    test_config = {
        "n_samples": 1024,
        "seq_len": 128,
    }

    def get_data(self):
        cache_dir = Path(tempfile.mkdtemp(prefix="benchopt_fineweb_"))

        # Stream a slice of FineWeb from HF Hub.
        raw = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.enable_padding(length=self.seq_len)
        tokenizer.enable_truncation(max_length=self.seq_len)

        records = []
        for i, example in enumerate(raw):
            if i >= self.n_samples:
                break
            records.append(example["text"])

        # Batch-tokenize with the low-level tokenizers library.
        encodings = tokenizer.encode_batch(records)
        tokenized = HFDataset.from_dict({
            "input_ids": [e.ids for e in encodings],
            "attention_mask": [e.attention_mask for e in encodings],
        })
        tokenized.save_to_disk(str(cache_dir / "tokenized"))

        return dict(
            dataset_path=str(cache_dir / "tokenized"),
            tokenizer_name=self.tokenizer_name,
        )
