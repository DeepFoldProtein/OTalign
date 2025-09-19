#!/usr/bin/env python3
"""
Build ffindex (queries.ff{data,index}) from a Hugging Face dataset.

Dataset schema assumed:
  - seq1_id, seq1
  - seq2_id, seq2
Duplicates by id are deduplicated.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Set, cast

from datasets import load_dataset


def wrap(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_prefix", default="work/queries", help="Output prefix (path without extension)")
    ap.add_argument("--wrap", type=int, default=60)
    ap.add_argument("--ffindex_from_fasta", default="ffindex_from_fasta")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ffdata = out_prefix.with_suffix(".ffdata")
    ffindex = out_prefix.with_suffix(".ffindex")
    names_file = out_prefix.with_suffix(".names")
    fasta_file = out_prefix.with_suffix(".fasta")

    ds = load_dataset(args.dataset, name=args.name, split=args.split)

    tmp = Path(tempfile.mkdtemp(prefix="ffq_"))
    try:
        seen: Set[str] = set()
        pairs: List[tuple[str, str]] = []
        for ex_raw in ds:
            ex = cast(dict, ex_raw)
            for sid, seq in ((ex["seq1_id"], ex["seq1"]), (ex["seq2_id"], ex["seq2"])):
                if sid in seen:
                    continue
                seen.add(sid)
                pairs.append((sid, seq))

        # Write sequences
        with fasta_file.open("w") as fp:
            for sid, seq in pairs:
                fp.write(f"{sid}\n{seq}\n")

        # Write names (optional, useful later)
        with names_file.open("w") as fp:
            for sid, _ in pairs:
                fp.write(f"{sid}\n")

        # Build ffindex
        # Many ffindex builds accept: ffindex_build -s OUT.ffdata OUT.ffindex list_of_files...
        cmd = [args.ffindex_from_fasta, "-s", str(ffdata), str(ffindex), str(fasta_file)]
        try:
            subprocess.check_call(cmd)
            print(f"[ok] wrote {ffdata} and {ffindex}")
            print(f"[ok] wrote {names_file} ({len(pairs)} entries)")
        except subprocess.CalledProcessError as e:
            print(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
