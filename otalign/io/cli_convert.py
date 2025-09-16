import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .alignment_parser import parse_a3m_alignment_text, parse_fasta_alignment_text
from .fasta_utils import write_fasta_pair
from .readers import read_text
from .writers import dump_jsonl_records


def _detect_format(path: Path, user_fmt: str | None) -> str:
    """
    Infer input format from --format or file suffix.
    Returns one of {"ali","a3m","fasta"}.
    """
    if user_fmt:
        fmt = user_fmt.lower()
        if fmt not in {"a3m", "fasta"}:
            raise ValueError("--format must be one of: ali, a3m, fasta")
        return fmt
    suf = path.suffix.lower()
    if suf == ".a3m":
        return "a3m"
    if suf in {".fa", ".fasta", ".faa", ".fas"}:
        return "fasta"
    raise ValueError(f"Cannot infer format from suffix: {path.suffix!r}. Use --format.")


def _parse_fasta(text: str) -> Tuple[str, str, str, str, List[Tuple[int, int]], str, str]:
    """
    Returns (id1, id2, seq1_ung, seq2_ung, pairs, seq1_aligned, seq2_aligned).
    For pairwise FASTA we assume aligned sequences already (same length).
    """
    id1, id2, seq1_ung, seq2_ung, pairs = parse_fasta_alignment_text(text)
    # Recover aligned strings from original text (concatenate sequence lines per record)
    recs = []
    cur_id = None
    cur_seq = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                recs.append((cur_id, "".join(cur_seq)))
            cur_id = line[1:].split()[0]
            cur_seq = []
        else:
            cur_seq.append(line)
    if cur_id is not None:
        recs.append((cur_id, "".join(cur_seq)))
    assert len(recs) == 2
    seq1_aligned, seq2_aligned = recs[0][1], recs[1][1]
    return id1, id2, seq1_ung, seq2_ung, pairs, seq1_aligned, seq2_aligned


def _parse_a3m(text: str) -> Tuple[str, str, str, str, List[Tuple[int, int]], str, str]:
    """
    Returns (id1, id2, seq1_ung, seq2_ung, pairs, seq1_aligned_projected, seq2_aligned_projected).
    We use projected aligned strings (lowercase removed) as the aligned representation.
    """
    id1, id2, seq1_ung, seq2_ung, pairs = parse_a3m_alignment_text(text)

    # Recompute projected aligned strings exactly as parser did (remove lowercase)
    def project(s: str) -> str:
        return "".join(ch for ch in s if ch == "-" or ("A" <= ch <= "Z"))

    # Grab raw sequences for selected IDs
    # Re-scan records to find the two chosen sequences
    by_id = {}
    cur_id = None
    cur_seq = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                by_id[cur_id] = "".join(cur_seq)
            cur_id = line[1:].split()[0]
            cur_seq = []
        else:
            if cur_id is not None:
                cur_seq.append(line.strip())
    if cur_id is not None:
        by_id[cur_id] = "".join(cur_seq)
    if id1 not in by_id or id2 not in by_id:
        raise ValueError("Selected IDs not found in A3M text (unexpected).")
    seq1_aligned = project(by_id[id1])
    seq2_aligned = project(by_id[id2])
    if len(seq1_aligned) != len(seq2_aligned):
        raise ValueError("Projected aligned strings differ in length.")
    return id1, id2, seq1_ung, seq2_ung, pairs, seq1_aligned, seq2_aligned


def main():
    ap = argparse.ArgumentParser(description="Convert alignment (.ali, .a3m, pairwise .fasta) to JSONL; optionally dump FASTA.")
    ap.add_argument("--input", "-i", required=True, help="Input alignment file.")
    ap.add_argument("--format", "-f", choices=["a3m", "fasta"], default=None, help="Input format; if omitted, inferred from suffix.")
    ap.add_argument("--output_jsonl", "-o", required=True, help="Output JSONL path.")
    ap.add_argument("--set_name", default="custom", help="set_name field for record (default: custom).")
    ap.add_argument("--dump_ungapped_fasta", default=None, help="Optional path to write ungapped pair as FASTA (two records).")
    ap.add_argument("--dump_aligned_fasta", default=None, help="Optional path to write aligned pair as FASTA (two records).")
    ap.add_argument("--wrap_width", type=int, default=None, help="FASTA wrap width (default: None).")
    args = ap.parse_args()

    in_path = Path(args.input)
    fmt = _detect_format(in_path, args.format)
    text = read_text(in_path)

    if fmt == "fasta":
        seq1_id, seq2_id, s1, s2, pairs, a1, a2 = _parse_fasta(text)
    elif fmt == "a3m":
        seq1_id, seq2_id, s1, s2, pairs, a1, a2 = _parse_a3m(text)
    else:
        raise AssertionError("unreachable")

    # Build one JSONL record (compatible with your schema)
    record: Dict = {
        "pair_id": f"{seq1_id}-{seq2_id}",
        "group_id": f"{seq1_id}-{seq2_id}",
        "set_name": args.set_name,
        "seq1_id": seq1_id,
        "seq2_id": seq2_id,
        "seq1": s1,
        "seq2": s2,
        "ref_alignment": pairs,
        "percent_identity": None,
        "scop_labels": [],
        "meta": json.dumps({"source": str(in_path.name), "format": fmt}),
    }
    dump_jsonl_records(args.output_jsonl, [record])

    # Optional FASTA dumps
    if args.dump_ungapped_fasta:
        write_fasta_pair(args.dump_ungapped_fasta, seq1_id, s1, seq2_id, s2, width=args.wrap_width)
    if args.dump_aligned_fasta:
        write_fasta_pair(args.dump_aligned_fasta, seq1_id, a1, seq2_id, a2, width=args.wrap_width)

    print(f"[ok] wrote JSONL → {args.output_jsonl}")
    if args.dump_ungapped_fasta:
        print(f"[ok] wrote ungapped FASTA → {args.dump_ungapped_fasta}")
    if args.dump_aligned_fasta:
        print(f"[ok] wrote aligned FASTA → {args.dump_aligned_fasta}")


if __name__ == "__main__":
    main()
