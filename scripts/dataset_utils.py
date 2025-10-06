import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

from datasets import load_dataset

from otalign.metrics.alignment import alignment_scores


def iter_pairs_from_dataset(dataset_path: str) -> Iterator[Dict[str, Any]]:
    """
    Loads pairs from a dataset (JSONL or Hugging Face) and yields them.
    """
    if not dataset_path:
        raise ValueError("dataset_path cannot be None or empty.")
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


def alignment_metrics(pred: List[Tuple[int, int]], ref: Optional[List[List[int]]]) -> Dict[str, float]:
    """
    Computes alignment scores (precision, recall, F1) between predicted and reference alignments.
    """
    ref_pairs = {cast(tuple[int, int], tuple(p)) for p in ref} if ref else set()
    pred_pairs = set(pred) if pred else set()
    metrics = alignment_scores(pred_pairs, ref_pairs)
    return asdict(metrics)
