#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import re
from typing import Dict, List, Tuple


###############################################################################
# Utilities: parsing alignments and metadata
###############################################################################


def clean_aligned_line(line: str) -> str:
    """
    Keep amino-acid characters and gaps. Normalize to uppercase.
    Some MALIDUP lines may include lowercase residues (keep them as residues).
    """
    line = line.strip()
    # Remove spaces; keep letters and '-' only
    line = re.sub(r"[^A-Za-z\-]", "", line)
    return line


def parse_manual_alignment_file(path: pathlib.Path) -> Tuple[str, str]:
    """
    Read a .manual.ali file and return the two aligned (gapped) strings.
    We skip lines that start with '#' and blank lines. We expect exactly two
    non-empty aligned strings.
    """
    aligned: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            s = clean_aligned_line(s)
            if not s:
                continue
            aligned.append(s)
    if len(aligned) < 2:
        raise ValueError(f"Expected at least 2 alignment lines in {path}, got {len(aligned)}.")
    if len(aligned[0]) != len(aligned[1]):
        raise ValueError(f"Gapped lines must have equal length in {path}.")
    # In case files contain more lines, use first two
    return aligned[0], aligned[1]


def gapped_to_mapping(aln1: str, aln2: str) -> Tuple[str, str, List[Tuple[int, int]], float]:
    """
    Convert a pair of gapped aligned strings into:
      - ungapped seq1, seq2 (uppercase)
      - list of 0-based index pairs (i, j) in ungapped coordinates
      - percent identity over aligned (non-gap) positions
    """
    if len(aln1) != len(aln2):
        raise ValueError("Aligned strings must have the same length.")
    ung1 = [c for c in aln1 if c != "-"]
    ung2 = [c for c in aln2 if c != "-"]
    seq1 = "".join(ung1).upper()
    seq2 = "".join(ung2).upper()

    i = j = 0
    pairs: List[Tuple[int, int]] = []
    matches = 0
    aligned = 0
    for a, b in zip(aln1, aln2):
        a_is = a != "-"
        b_is = b != "-"
        if a_is and b_is:
            if a.isupper() and b.isupper():
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
            # gap-gap: advance neither
            pass
    pid = (matches / aligned * 100.0) if aligned > 0 else 0.0
    return seq1, seq2, pairs, pid


def parse_ranges(rng: str) -> List[str]:
    """
    Parse Range strings like '(A1-A45)' or '(A17-A49,A50-A118)'.
    Return a list of normalized segment strings, e.g. ['A1-A45', 'A50-A118'].
    """
    if not rng:
        return []
    m = re.findall(r"\(([^\)]*)\)", rng)
    if not m:
        return []
    segments = []
    for part in ",".join(m).split(","):
        s = part.strip()
        if s:
            segments.append(s)
    return segments


###############################################################################
# Reading dup.txt metadata
###############################################################################


def read_dup_table(path: pathlib.Path) -> Dict[str, dict]:
    """
    Read dup.txt (tab-separated) and build a dict keyed by Pair Name (or parent folder name).
    Expected header columns (based on your sample):
    Pair Name, Domain1, SCOP ID1, Range1, Domain2, SCOP ID2, Range2,
    SCOP Class, SCOP Fold, SCOP Superfamily, SCOP Family, SCOP Protein
    """
    by_pair: Dict[str, dict] = {}
    # Handle either tab or multiple spaces. We'll try csv with delimiter='\t' first.
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        # Attempt to sniff delimiter
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        reader = csv.DictReader(f, dialect=dialect)

        # Normalize field names
        def norm(k: str) -> str:
            return re.sub(r"\s+", " ", k.strip())

        fieldmap = {k: norm(k) for k in reader.fieldnames or []}
        rows = []
        for row in reader:
            # Some dup.txt variants may not have a proper header; fallback manual split
            rows.append({fieldmap.get(k, k): v.strip() for k, v in row.items()})
    # If header detection failed and we got empty dicts, fallback to manual parsing
    if not rows or all(not any(r.values()) for r in rows):
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split()

            def idx(name: str) -> int:
                for i, h in enumerate(header):
                    if h.lower().startswith(name.lower()):
                        return i
                return -1

            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"\t+|\s{2,}", line)
                if len(parts) < 12:
                    continue
                rows.append(
                    {
                        "Pair Name": parts[0],
                        "Domain1": parts[1],
                        "SCOP ID1": parts[2],
                        "Range1": parts[3],
                        "Domain2": parts[4],
                        "SCOP ID2": parts[5],
                        "Range2": parts[6],
                        "SCOP Class": parts[7],
                        "SCOP Fold": parts[8],
                        "SCOP Superfamily": parts[9],
                        "SCOP Family": parts[10],
                        "SCOP Protein": parts[11],
                    }
                )
    # Build dict
    for r in rows:
        pair_name = r.get("Pair Name") or r.get("Pair") or ""
        pair_name = pair_name.strip()
        by_pair[pair_name] = {
            "domain1": r.get("Domain1", "").strip(),
            "scop_id1": r.get("SCOP ID1", "").strip(),
            "range1": parse_ranges(r.get("Range1", "")),
            "domain2": r.get("Domain2", "").strip(),
            "scop_id2": r.get("SCOP ID2", "").strip(),
            "range2": parse_ranges(r.get("Range2", "")),
            "scop_class": r.get("SCOP Class", "").strip(),
            "scop_fold": r.get("SCOP Fold", "").strip(),
            "scop_superfamily": r.get("SCOP Superfamily", "").strip(),
            "scop_family": r.get("SCOP Family", "").strip(),
            "scop_protein": r.get("SCOP Protein", "").strip(),
        }
    return by_pair


###############################################################################
# Main conversion
###############################################################################


def convert_malidup(
    root: pathlib.Path,
    dup_txt: pathlib.Path,
    output_jsonl: pathlib.Path,
    ali_glob: str = "**/*.manual.ali",
) -> int:
    """
    Convert MALIDUP into JSONL entries.
    We assume each pair lives under a folder named by 'Pair Name', and contains a *.manual.ali file.
    Example: data/MALIDUP/d2dri__/2dri.manual.ali  -> pair_name = 'd2dri__'
    """
    meta_by_pair = read_dup_table(dup_txt)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_jsonl.open("w", encoding="utf-8") as fout:
        for ali_path in sorted(root.glob(ali_glob)):
            pair_name = ali_path.parent.name  # use parent directory name as Pair Name
            # Parse the alignment
            aln1, aln2 = parse_manual_alignment_file(ali_path)
            seq1, seq2, ref_pairs, pid = gapped_to_mapping(aln1, aln2)

            m = meta_by_pair.get(pair_name, {})
            # Prefer human-readable labels (strings), assemble scop_labels like SABmark
            scop_labels: List[str] = []
            for key in ["scop_class", "scop_fold", "scop_superfamily", "scop_family", "scop_protein"]:
                val = m.get(key, "")
                if val:
                    scop_labels.append(f"{key.replace('scop_', '')}:{val}")

            # Domain IDs (sequence identifiers)
            seq1_id = m.get("domain1", "") or "seq1"
            seq2_id = m.get("domain2", "") or "seq2"

            # Build JSONL record
            ex = {
                "pair_id": f"{pair_name}:{seq1_id}-{seq2_id}",
                "group_id": pair_name,
                "set_name": "malidup",  # dataset tag
                "seq1_id": seq1_id,
                "seq2_id": seq2_id,
                "seq1": seq1,
                "seq2": seq2,
                "ref_alignment": ref_pairs,  # [[i, j], ...] 0-based indices
                "percent_identity": pid,
                "scop_labels": scop_labels,  # human-readable SCOP strings from dup.txt
                "meta": json.dumps(
                    {
                        "scop_id1": m.get("scop_id1", ""),
                        "range1": m.get("range1", []),
                        "scop_id2": m.get("scop_id2", ""),
                        "range2": m.get("range2", []),
                        "source_file": str(ali_path.name),
                    }
                ),
            }
            fout.write(json.dumps(ex) + "\n")
            written += 1
    return written


def main():
    ap = argparse.ArgumentParser(description="Convert MALIDUP to JSONL (SABmark-like schema).")
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder containing MALIDUP subfolders with *.manual.ali files (e.g., data/MALIDUP).",
    )
    ap.add_argument("--dup_txt", type=str, required=True, help="Path to dup.txt (tab-separated metadata).")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL path (e.g., datasets/malidup/data/all.jsonl).")
    ap.add_argument(
        "--ali_glob",
        type=str,
        default="**/*.manual.ali",
        help="Glob pattern to find manual alignment files under root.",
    )
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    dup_txt = pathlib.Path(args.dup_txt).resolve()
    out = pathlib.Path(args.output).resolve()

    count = convert_malidup(root, dup_txt, out, ali_glob=args.ali_glob)
    print(f"[done] wrote {count} pairs to {out}")


if __name__ == "__main__":
    main()
