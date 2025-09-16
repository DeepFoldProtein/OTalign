#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import List, Tuple


###############################################################################
# Regex rules for MALISAM IDs and file layout
###############################################################################

# Exactly 7-character SCOP domain ID: 'd' + 5 alnum + trailing underscore
# Examples: d1aa7a_, d1b68a_
SCOP7_RE = re.compile(r"d[a-z0-9\._]{6}", re.IGNORECASE)

# Directory names look like: d1aa7a_d1b68a_ (optionally with trailing slash)
# We want to extract exactly two 7-char IDs.
PAIR_DIR_RE = re.compile(r"(d[a-z0-9_\.]{6})(d[a-z0-9_\.]{6})", re.IGNORECASE)

# Alignment filenames typically repeat the pair: d1aa7a_d1b68a_.manual.ali
PAIR_FILE_RE = re.compile(r"(d[a-z0-9_\.]{6})(d[a-z0-9_\.]{6})\.manual\.ali$", re.IGNORECASE)

###############################################################################
# Alignment parsing utilities
###############################################################################


def clean_aligned_line(line: str) -> str:
    """
    Keep amino-acid letters (any case) and gap '-'. Drop everything else.
    Uppercase for consistency.
    """
    s = line.strip()
    s = re.sub(r"[^A-Za-z\-]", "", s)
    return s


def read_manual_ali(path: pathlib.Path) -> Tuple[str, str]:
    """
    Read a *.manual.ali file and return two aligned (gapped) strings.
    - Skips blank lines and lines starting with '#'
    - Cleans lines to keep only letters and '-' (uppercased)
    - Uses the first two non-empty aligned lines
    """
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            s = clean_aligned_line(raw)
            if s:
                lines.append(s)
    if len(lines) < 2:
        raise ValueError(f"Expected >=2 aligned lines in {path}, found {len(lines)}.")
    if len(lines[0]) != len(lines[1]):
        raise ValueError(f"Gapped lines must have equal length in {path}.")
    return lines[0], lines[1]


def gapped_to_mapping(aln1: str, aln2: str) -> Tuple[str, str, List[Tuple[int, int]], float]:
    """
    Convert gapped aligned strings into:
      - ungapped seq1, seq2 (uppercase)
      - list of 0-based index pairs (i, j) in ungapped coordinates
      - percent identity over aligned (non-gap) positions
    """
    if len(aln1) != len(aln2):
        raise ValueError("Aligned strings must have the same length.")
    seq1 = "".join(c for c in aln1 if c != "-").upper()
    seq2 = "".join(c for c in aln2 if c != "-").upper()

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
            # gap-gap
            pass
    pid = (matches / aligned * 100.0) if aligned > 0 else 0.0
    return seq1, seq2, pairs, pid


###############################################################################
# ID parsing from directory/file names
###############################################################################


def parse_pair_ids_from_dir(dir_name: str) -> Tuple[str, str] | None:
    """
    Parse two strict 7-character SCOP IDs from a pair directory name.
    Expected: d1aa7a_d1b68a_  (two tokens, each 7 chars, separated by underscore)
    Returns (seq1_id, seq2_id) or None if not matched.
    """
    m = PAIR_DIR_RE.search(dir_name.strip("_"))
    if not m:
        return None
    return m.group(1), m.group(2)


def parse_pair_ids_from_file(file_name: str) -> Tuple[str, str] | None:
    """
    Parse two strict 7-character SCOP IDs from a *.manual.ali filename.
    Expected: d1aa7a_d1b68a_.manual.ali
    Returns (seq1_id, seq2_id) or None if not matched.
    """
    m = PAIR_FILE_RE.search(file_name)
    if not m:
        return None
    return m.group(1), m.group(2)


def read_optional_notes(dir_path: pathlib.Path) -> str:
    """
    Read optional free-text note file (e.g., '1aa7_1b68.txt') if present.
    Returns the content (truncated to a reasonable size) or empty string.
    """
    for p in sorted(dir_path.glob("*.txt")):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            return txt[:50000]  # avoid oversized blobs
        except Exception:
            continue
    return ""


###############################################################################
# Main conversion
###############################################################################


def convert_malisam(
    root: pathlib.Path,
    output_jsonl: pathlib.Path,
    ali_glob: str = "*.manual.ali",
) -> int:
    """
    Convert MALISAM directories to JSONL records with SABmark-compatible fields.

    Expected layout:
      root/
        d1aa7a_d1b68a_/
          d1aa7a_d1b68a_.manual.ali
          1aa7_1b68.txt           # optional notes (will be stored in meta)
        dXXXXXXXdYYYYYYX/ ...
    """
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_jsonl.open("w", encoding="utf-8") as fout:
        # Iterate over pair directories
        for pair_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            dir_name = pair_dir.name

            # Parse the two 7-char IDs from dir name; fallback to filename if needed
            ids = parse_pair_ids_from_dir(dir_name)
            ali_files = sorted(pair_dir.glob(ali_glob))
            if not ali_files:
                # No alignment file found -> skip
                continue
            ali_path = ali_files[0]

            if ids is None:
                ids = parse_pair_ids_from_file(ali_path.name)
            if ids is None:
                # Cannot confirm two strict 7-char IDs -> skip
                continue
            seq1_id, seq2_id = ids

            # Sanity check: enforce 7-char rule strictly
            if not (SCOP7_RE.fullmatch(seq1_id) and SCOP7_RE.fullmatch(seq2_id)):
                # IDs do not meet the 7-char rule -> skip
                continue

            # Parse alignment (two gapped lines), convert to mapping
            aln1, aln2 = read_manual_ali(ali_path)
            seq1, seq2, ref_pairs, pid = gapped_to_mapping(aln1, aln2)

            # Optional notes
            notes = read_optional_notes(pair_dir)

            # Assemble JSON record
            pair_id = f"{dir_name}:{seq1_id}-{seq2_id}"
            ex = {
                "pair_id": pair_id,
                "group_id": dir_name,
                "set_name": "malisam",
                "seq1_id": seq1_id,
                "seq2_id": seq2_id,
                "seq1": seq1,
                "seq2": seq2,
                "ref_alignment": ref_pairs,  # [[i, j], ...] in 0-based ungapped coords
                "percent_identity": pid,
                "scop_labels": [],  # MALISAM focuses on analogs; SCOP tags typically omitted
                "meta": json.dumps(
                    {
                        "notes": notes,
                        "source_file": ali_path.name,
                    }
                ),
            }
            fout.write(json.dumps(ex) + "\n")
            written += 1

    return written


def main():
    ap = argparse.ArgumentParser(description="Convert MALISAM to JSONL (strict 7-char SCOP IDs).")
    ap.add_argument("--root", type=str, required=True, help="Root folder containing MALISAM pair directories (e.g., data/MALISAM).")
    ap.add_argument("--output", type=str, required=True, help="Output JSONL path (e.g., datasets/malisam/data/all.jsonl).")
    ap.add_argument(
        "--ali_glob",
        type=str,
        default="*.manual.ali",
        help="Glob pattern to find manual alignment files per directory.",
    )
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    out = pathlib.Path(args.output).resolve()

    count = convert_malisam(root, out, ali_glob=args.ali_glob)
    print(f"[done] wrote {count} pairs to {out}")


if __name__ == "__main__":
    main()
