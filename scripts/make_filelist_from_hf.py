"""
Create single-sequence FASTA files from a Hugging Face dataset and write a FILELIST
for SLURM array jobs (e.g., hhblits/hhmake pipeline).

Assumes dataset examples contain:
  - seq1_id (str), seq2_id (str)
  - seq1 (str, ungapped AA letters), seq2 (str)

If your dataset uses different field names, use --seq-fields to map them.
"""

import argparse
import re
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from otalign.io.fasta_utils import write_fasta


def sanitize_id(s: str) -> str:
    """Make a filesystem-safe identifier (keep alnum, underscore, dash)."""
    s2 = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    # avoid empty name
    return s2 if s2 else "seq"


def iter_pairs_default(example: dict) -> Iterable[tuple[str, str]]:
    """Yield (seq_id, seq) for seq1 and seq2 from a default schema."""
    yield example["seq1_id"], example["seq1"]
    yield example["seq2_id"], example["seq2"]


def iter_pairs_custom(example: dict, seq1_id: str, seq1: str, seq2_id: str, seq2: str) -> Iterable[tuple[str, str]]:
    """Yield (seq_id, seq) using custom field names."""
    yield example[seq1_id], example[seq1]
    yield example[seq2_id], example[seq2]


def main():
    ap = argparse.ArgumentParser(description="Export single-sequence FASTA files + FILELIST from an HF dataset.")
    ap.add_argument("--dataset", required=True, help="HF dataset name, e.g. DeepFoldProtein/SABmark")
    ap.add_argument("--name", required=True, help="HF config name, e.g. twi/sup/...")  # SABmark uses names
    ap.add_argument("--split", default="test", help="Split to use (default: test)")
    ap.add_argument("--out_dir", default="data/fasta", help="Directory to write FASTA files")
    ap.add_argument("--filelist", default="work/fasta.list", help="Path to write FILELIST for SLURM array")
    ap.add_argument("--wrap", type=int, default=60, help="FASTA line wrap width (default: 60)")
    ap.add_argument("--seq-fields", nargs=4, metavar=("SEQ1_ID", "SEQ1", "SEQ2_ID", "SEQ2"), help="Custom field names if they differ from defaults (seq1_id seq1 seq2_id seq2)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    filelist_path = Path(args.filelist)
    filelist_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, name=args.name, split=args.split)

    seen: set[str] = set()
    paths: list[Path] = []

    if args.seq_fields:
        f_seq1_id, f_seq1, f_seq2_id, f_seq2 = args.seq_fields
        _ = ((ex[f_seq1_id], ex[f_seq1]) if (k % 2 == 0) else (ex[f_seq2_id], ex[f_seq2]) for k, ex in enumerate(ds for _ in (0, 1)))
        # The above comprehension is awkward for clarity; just handle explicitly instead:
        paths.clear()
        seen.clear()
        for ex in ds:
            for sid, s in iter_pairs_custom(ex, f_seq1_id, f_seq1, f_seq2_id, f_seq2):
                if sid in seen:
                    continue
                seen.add(sid)
                p = out_dir / f"{sanitize_id(sid)}.fasta"
                write_fasta(p, sid, s, width=args.wrap)
                paths.append(p)
    else:
        for ex in ds:
            for sid, s in iter_pairs_default(ex):
                if sid in seen:
                    continue
                seen.add(sid)
                p = out_dir / f"{sanitize_id(sid)}.fasta"
                write_fasta(p, sid, s, width=args.wrap)
                paths.append(p)

    # Write FILELIST with absolute paths
    with filelist_path.open("w", encoding="utf-8") as f:
        for p in sorted(paths):
            f.write(str(p.resolve()) + "\n")

    print(f"[ok] wrote {len(paths)} FASTA files to {out_dir}")
    print(f"[ok] wrote FILELIST -> {filelist_path}")
    print("set your SLURM array to: 0-{}".format(max(0, len(paths) - 1)))


if __name__ == "__main__":
    main()
