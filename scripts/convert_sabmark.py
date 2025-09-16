#!/usr/bin/env python3
"""
SABmark -> JSONL converter.

Assumptions (based on provided snippets):
- Folder layout:
    ROOT/
      SABmark/
        twi/
          group10/
            group.summary
            reference/
              d1a32__-d1ail__.fasta
              ...
        sup/
        twi_fp/
        sup_fp/
- group.summary format (tab or whitespace separated):
    Name  Length  True pos  SCOP classification class-fold-superfamily-family-domain-species-protein
    d1a32__ 85 1 46456 47059 47060 47064 47065 47066 16384
    ...
- reference/*.fasta are two-entry gapped alignments for a single pair.

Output JSONL fields per line:
{
  "pair_id": "<group_id>:<seq1_id>-<seq2_id>",
  "group_id": "group10",
  "set_name": "twi" | "sup" | "twi_fp" | "sup_fp",
  "seq1_id": "d1a32__",
  "seq2_id": "d1ail__",
  "seq1": "<ungapped seq1>",
  "seq2": "<ungapped seq2>",
  "ref_alignment": [[i0,j0],[i1,j1],...],  # 0-based indices in ungapped sequences
  "percent_identity": <float>,              # matches / aligned_positions * 100
  "scop_labels": ["class:46456","fold:47059","superfamily:47060","family:47064","domain:47065","species:47066","protein:16384"],
  "meta": "{\"length1\":85,\"length2\":70,\"true_pos1\":1,\"true_pos2\":1}"
}
"""

import argparse
import json
import pathlib
from typing import Dict, List, Tuple


SUBSET_MAP = {
    "twi": "twi",
    "sup": "sup",
    "twi_fp": "twi_fp",
    "sup_fp": "sup_fp",
}


def read_fasta_pairs(path: pathlib.Path) -> Tuple[str, str, str, str]:
    """
    Read a two-entry FASTA file and return (id1, aln1, id2, aln2).
    Headers are taken as IDs up to first whitespace.
    """
    headers: List[str] = []
    seqs: List[str] = []
    cur = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                headers.append(line[1:].split()[0])
            else:
                cur.append(line)
        if cur:
            seqs.append("".join(cur))
    if len(headers) != 2 or len(seqs) != 2:
        raise ValueError(f"Expected 2 sequences in {path}, got {len(headers)} headers / {len(seqs)} seqs.")
    return headers[0], seqs[0], headers[1], seqs[1]


def parse_gapped_alignment(aln1: str, aln2: str) -> Tuple[str, str, List[Tuple[int, int]], float]:
    """
    Convert gapped aligned strings into:
      - ungapped seq1, seq2
      - index pairs in 0-based ungapped coordinates
      - percent identity over aligned (non-gap) positions
    """
    if len(aln1) != len(aln2):
        raise ValueError("Aligned sequences must have the same length.")
    # Build ungapped sequences
    ung1 = [a for a in aln1 if a != "-"]
    ung2 = [b for b in aln2 if b != "-"]
    seq1 = "".join(ung1)
    seq2 = "".join(ung2)

    # Walk alignment to build index pairs and identity
    i = j = 0
    pairs: List[Tuple[int, int]] = []
    matches = 0
    aligned = 0
    for a, b in zip(aln1, aln2):
        a_is = a != "-"
        b_is = b != "-"
        if a_is and b_is:
            pairs.append((i, j))
            aligned += 1
            if a == b:
                matches += 1
            i += 1
            j += 1
        elif a_is and not b_is:
            i += 1
        elif not a_is and b_is:
            j += 1
        else:
            # gap-gap; no index advance
            pass
    pid = (matches / aligned * 100.0) if aligned > 0 else 0.0
    return seq1, seq2, pairs, pid


def parse_group_summary(path: pathlib.Path) -> Dict[str, dict]:
    """
    Parse group.summary into a dict: {seq_id: {"length": int, "true_pos": int, "scop": [ints...]}}
    The header line contains column descriptions; we assume whitespace-separated columns.
    """
    meta: Dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Name"):
                continue
            parts = line.split()
            # Expected: Name Length True pos SCOP classification <7 integers>
            if len(parts) < 10:
                # Be forgiving; skip malformed lines
                continue
            name = parts[0]
            length = int(parts[1])
            true_pos = int(parts[2])
            # Next 7 integers: class fold superfamily family domain species protein
            scop_vals = list(map(int, parts[3:10]))
            meta[name] = {
                "length": length,
                "true_pos": true_pos,
                "scop": scop_vals,
            }
    return meta


def format_scop_labels(scop_vals: List[int]) -> List[str]:
    keys = ["class", "fold", "superfamily", "family", "domain", "species", "protein"]
    return [f"{k}:{v}" for k, v in zip(keys, scop_vals)]


def detect_subset_from_path(path: pathlib.Path) -> str:
    """
    Map folder names to canonical subset names.
    """
    s = path.as_posix()
    for key, label in SUBSET_MAP.items():
        if f"/{key}/" in s or s.endswith(f"/{key}"):
            return label
    # fallback: try direct tokens
    s_lower = s.lower()
    if "twi" in s_lower:
        return "twi"
    if "superfamil" in s_lower:
        return "sup"
    return "twi"


def convert_group(group_dir: pathlib.Path, subset_label: str, out_handle) -> int:
    """
    Convert one group directory:
      - read group.summary for per-sequence metadata
      - iterate reference/*.fasta pairs
      - write JSON objects to out_handle
    Returns count of pairs written.
    """
    summary_path = group_dir / "group.summary"
    meta_by_seq = parse_group_summary(summary_path) if summary_path.exists() else {}

    ref_dir = group_dir / "reference"
    if not ref_dir.exists():
        return 0

    written = 0
    for fasta_path in sorted(ref_dir.glob("*.fasta")):
        seq1_id, aln1, seq2_id, aln2 = read_fasta_pairs(fasta_path)
        seq1, seq2, ref_pairs, pid = parse_gapped_alignment(aln1, aln2)

        m1 = meta_by_seq.get(seq1_id, {})
        m2 = meta_by_seq.get(seq2_id, {})

        scop = m1.get("scop") or m2.get("scop") or []
        scop_labels = format_scop_labels(scop) if scop else []

        meta = {
            "length1": m1.get("length"),
            "length2": m2.get("length"),
            "true_pos1": m1.get("true_pos"),
            "true_pos2": m2.get("true_pos"),
            "source_file": fasta_path.name,
        }

        group_id = group_dir.name
        pair_id = f"{group_id}:{seq1_id}-{seq2_id}"

        ex = {
            "pair_id": pair_id,
            "group_id": group_id,
            "set_name": subset_label,
            "seq1_id": seq1_id,
            "seq2_id": seq2_id,
            "seq1": seq1,
            "seq2": seq2,
            "ref_alignment": ref_pairs,  # list of [i,j] pairs in 0-based ungapped coords
            "percent_identity": pid,
            "scop_labels": scop_labels,  # formatted strings
            "meta": json.dumps(meta),
        }
        out_handle.write(json.dumps(ex) + "\n")
        written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description="Convert SABmark to JSONL (HF-ready).")
    ap.add_argument("--root", type=str, required=True, help="Root containing SABmark subsets (e.g., data/SABmark).")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory to write JSONL files (one per subset plus all.jsonl).")
    ap.add_argument(
        "--subsets",
        type=str,
        nargs="*",
        default=["twi", "sup", "twi_fp", "sup_fp"],
        help="Which raw subset folders to include (default: twi sup twi_fp sup_fp).",
    )
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    outdir = pathlib.Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    total_all = 0
    all_path = outdir / "all.jsonl"
    with all_path.open("w", encoding="utf-8") as all_out:
        for raw_subset in args.subsets:
            subset_dir = root / raw_subset
            if not subset_dir.exists():
                print(f"[warn] subset not found: {subset_dir}")
                continue
            subset_label = SUBSET_MAP.get(raw_subset, detect_subset_from_path(subset_dir))
            subset_path = outdir / f"{raw_subset}.jsonl"
            cnt_subset = 0
            with subset_path.open("w", encoding="utf-8") as sub_out:
                for group_dir in sorted(subset_dir.glob("group*")):
                    if not group_dir.is_dir():
                        continue
                    written = convert_group(group_dir, subset_label, sub_out)
                    cnt_subset += written
            # append subset file content to all.jsonl
            with subset_path.open("r", encoding="utf-8") as sub_in:
                for line in sub_in:
                    all_out.write(line)
                    total_all += 1
            print(f"[ok] wrote {cnt_subset} pairs to {subset_path}")

    print(f"[done] wrote {total_all} total pairs to {all_path}")


if __name__ == "__main__":
    main()
