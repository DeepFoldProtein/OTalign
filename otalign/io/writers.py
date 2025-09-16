import json
from pathlib import Path
from typing import Dict, Iterable, Tuple


Pair = Tuple[int, int]


def record_from_aligned_pair(
    pair_id: str,
    seq1_id: str,
    seq2_id: str,
    seq1_aligned: str,
    seq2_aligned: str,
    set_name: str = "custom",
    extra: Dict | None = None,
) -> Dict:
    """
    Build a JSONL-ready record using two aligned strings (with gaps).
    Stores ungapped sequences and computed pairs.
    """
    # Compute ungapped indices and pairs
    from .parser import gapped_to_pairs, ungap

    pairs = gapped_to_pairs(seq1_aligned, seq2_aligned)
    rec = {
        "pair_id": pair_id,
        "group_id": pair_id.split(":")[0] if ":" in pair_id else pair_id,
        "set_name": set_name,
        "seq1_id": seq1_id,
        "seq2_id": seq2_id,
        "seq1": ungap(seq1_aligned),
        "seq2": ungap(seq2_aligned),
        "ref_alignment": pairs,
        "percent_identity": None,  # optionally fill later
        "scop_labels": [],
        "meta": json.dumps(extra or {}),
    }
    return rec


def dump_jsonl_records(path: str | Path, records: Iterable[Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
