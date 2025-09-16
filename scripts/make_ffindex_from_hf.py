#!/usr/bin/env python3
"""
Build ffindex (queries.ff{data,index}) from a Hugging Face dataset.

Dataset schema assumed:
  - seq1_id, seq1
  - seq2_id, seq2
Duplicates by id are deduplicated.
"""

import argparse, subprocess, tempfile, shutil
from pathlib import Path
from typing import Set, List
from datasets import load_dataset

def wrap(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_prefix", default="work/queries", help="Output prefix (path without extension)")
    ap.add_argument("--wrap", type=int, default=60)
    ap.add_argument("--ffindex_build", default="ffindex_build")
    args = ap.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    ffdata = out_prefix.with_suffix(".ffdata")
    ffindex = out_prefix.with_suffix(".ffindex")
    names_file = out_prefix.with_suffix(".names")

    ds = load_dataset(args.dataset, name=args.name, split=args.split)

    tmp = Path(tempfile.mkdtemp(prefix="ffq_"))
    try:
        seen: Set[str] = set()
        files: List[Path] = []
        for ex in ds:
            for sid, seq in ((ex["seq1_id"], ex["seq1"]), (ex["seq2_id"], ex["seq2"])):
                if sid in seen:
                    continue
                seen.add(sid)
                p = tmp / f"{sid}.fasta"
                with p.open("w", encoding="utf-8") as f:
                    f.write(f">{sid}\n{wrap(seq, width=args.wrap)}\n")
                files.append(p)

        # Write names (optional, useful later)
        with names_file.open("w", encoding="utf-8") as f:
            for p in files:
                f.write(p.stem + "\n")

        # Build ffindex
        # Many ffindex builds accept: ffindex_build -s OUT.ffdata OUT.ffindex list_of_files...
        cmd = [args.ffindex_build, "-s", str(ffdata), str(ffindex)] + [str(p) for p in files]
        subprocess.check_call(cmd)
        print(f"[ok] wrote {ffdata} and {ffindex}")
        print(f"[ok] wrote {names_file} ({len(files)} entries)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()

