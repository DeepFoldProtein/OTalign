import json
import logging
from pathlib import Path

from torch.utils.data import Dataset


class CATHDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train"):
        """
        Dataset for CATH domain pairs, loading from .jsonl files.

        Args:
            data_root (str): Path to the directory containing the dataset files (e.g., 'work/cath_dataset').
            split (str): One of "train", "validation", or "test".
        """
        self.data_root = Path(data_root)
        self.split = split
        self.data_file = self.data_root / f"{self.split}.jsonl"

        if not self.data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_file}")

        self.data = self._load_data()

    def _load_data(self):
        data = []
        with open(self.data_file, "r") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Could not decode line in {self.data_file}: {line.strip()}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]

        # The ground truth alignment is a list of tuples.
        # It will be converted to a dense tensor in the collator.
        return {
            "seq1": record["seq1"],
            "seq2": record["seq2"],
            "len1": len(record["seq1"]),
            "len2": len(record["seq2"]),
            "ref_alignment": record["ref_alignment"],
            "is_positive": record["label"] == "positive",
        }
