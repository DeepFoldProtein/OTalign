import json
from pathlib import Path
from typing import Any, Dict, Iterator, cast

from datasets import load_dataset


def iter_pairs_from_dataset(dataset_path: str) -> Iterator[Dict[str, Any]]:
    """
    Loads pairs from a dataset (JSONL or Hugging Face) and yields them.
    """
    print(f"Loading pairs from {dataset_path}...")

    if Path(dataset_path).is_file() and dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    else:
        parts = dataset_path.split(",")
        ds_name = parts[0]
        config_name = parts[1] if len(parts) > 1 else None
        split = parts[2] if len(parts) > 2 else "test"

        print(f"Loading Hugging Face dataset: {ds_name} (config: {config_name}, split: {split})")
        ds = load_dataset(ds_name, name=config_name, split=split)
        for ex_raw in ds:
            yield cast(dict, ex_raw)
