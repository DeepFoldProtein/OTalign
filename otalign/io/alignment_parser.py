import re
from typing import Dict, List, Tuple


Pair = Tuple[int, int]


### Core mapping


def gapped_to_pairs(aln1: str, aln2: str) -> List[Pair]:
    """
    Convert two gapped aligned strings (same length) into 0-based residue index pairs.
    Rules:
      - When both positions are residues (not '-'), emit the current ungapped indices.
      - Advance the ungapped index of a sequence only when its char != '-'.
      - Gap-gap columns advance neither index.
    """
    if len(aln1) != len(aln2):
        raise ValueError("Aligned strings must have equal length.")
    i = j = 0
    pairs: List[Pair] = []
    for a, b in zip(aln1, aln2):
        a_is = a != "-" and a != "."
        b_is = b != "-" and b != "."
        if a_is and b_is:
            pairs.append((i, j))
            i += 1
            j += 1
        elif a_is and not b_is:
            i += 1
        elif not a_is and b_is:
            j += 1
        else:
            # gap-gap
            pass
    return pairs


def ungap(seq: str) -> str:
    return "".join(c for c in seq if c.isupper())


### FASTA-like parsers


def split_fasta_blocks(text: str) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []
    cur: Dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                recs.append(cur)
            cur = {"id": line[1:].split()[0], "seq": ""}
        else:
            if cur is None:
                continue  # tolerate garbage before first header
            cur["seq"] += line.strip()
    if cur:
        recs.append(cur)
    return recs


def parse_fasta_alignment_text(text: str) -> Tuple[str, str, str, str, List[Pair]]:
    """
    Parse a pairwise FASTA alignment (exactly two sequences).
    Returns (id1, id2, seq1_ungapped, seq2_ungapped, pairs).
    """
    recs = split_fasta_blocks(text)
    if len(recs) != 2:
        raise ValueError(f"Expected exactly 2 sequences in FASTA, got {len(recs)}.")
    s1 = recs[0]["seq"]
    s2 = recs[1]["seq"]
    if len(s1) != len(s2):
        raise ValueError("Aligned FASTA sequences must have equal length.")
    pairs = gapped_to_pairs(s1, s2)
    return recs[0]["id"], recs[1]["id"], ungap(s1), ungap(s2), pairs


def convert_a3m_text_to_a2m(text: str) -> List[Dict[str, str]]:
    """
    Parses a multi-sequence alignment in A2M format.
    This format allows for non-rectangular alignments where insertions
    are represented by lowercase letters and can have varying lengths.
    This function rectifies the alignment by padding insertions with '.'
    to make all sequences of equal length.
    """
    records = split_fasta_blocks(text)
    if not records:
        return []

    split_pattern = re.compile(r"([A-Z-])")
    sequences_parts = [split_pattern.split(rec["seq"]) for rec in records]

    # Check for consistent number of match columns
    if sequences_parts:
        num_parts = len(sequences_parts[0])
        for i, parts in enumerate(sequences_parts):
            if len(parts) != num_parts:
                raise ValueError(f"A3M sequences have inconsistent number of match columns. Seq 0 has {len(sequences_parts[0]) // 2} matches, seq {i} ({records[i]['id']}) has {len(parts) // 2}.")

    # Find max length for each part (even indices are inserts)
    max_lens = [0] * (len(sequences_parts[0]) if sequences_parts else 0)
    for parts in sequences_parts:
        for i, part in enumerate(parts):
            if len(part) > max_lens[i]:
                max_lens[i] = len(part)

    # Rebuild sequences with padding
    new_records = []
    for i, rec in enumerate(records):
        parts = sequences_parts[i]
        new_seq_parts = []
        for j, part in enumerate(parts):
            if j % 2 == 0:  # Insert part
                new_seq_parts.append(part.ljust(max_lens[j], "."))
            else:  # Match part
                new_seq_parts.append(part)

        new_records.append({"id": rec["id"], "seq": "".join(new_seq_parts)})

    return new_records


def parse_a3m_alignment_text(text: str) -> Tuple[str, str, str, str, List[Pair]]:
    recs = convert_a3m_text_to_a2m(text)
    if len(recs) != 2:
        raise ValueError(f"Expected exactly 2 sequences in FASTA, got {len(recs)}.")
    s1 = recs[0]["seq"].upper().replace(".", "-")
    s2 = recs[1]["seq"].upper().replace(".", "-")
    if len(s1) != len(s2):
        raise ValueError("Aligned FASTA sequences must have equal length.")
    pairs = gapped_to_pairs(s1, s2)
    return recs[0]["id"], recs[1]["id"], ungap(s1), ungap(s2), pairs


### Other formats


def parse_stockholm_alignment_text(text: str) -> List[Dict[str, str]]:
    """
    Parses a multi-sequence alignment in Stockholm (sto) format.
    """
    seqs_by_id: Dict[str, List[str]] = {}
    # Order of sequences is important
    ordered_ids: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        seq_id, seq_part = parts[0], parts[1]

        if seq_id not in seqs_by_id:
            seqs_by_id[seq_id] = []
            ordered_ids.append(seq_id)

        seqs_by_id[seq_id].append(seq_part)

    records = []
    for seq_id in ordered_ids:
        records.append({"id": seq_id, "seq": "".join(seqs_by_id[seq_id])})
    return records


def parse_clustal_alignment_text(text: str) -> List[Dict[str, str]]:
    """
    Parses a multi-sequence alignment in Clustal (clu) format.
    Note: This is a simplified parser and might not handle all Clustal variations.
    """
    records: List[Dict[str, str]] = []

    lines_iter = iter(text.splitlines())

    # Skip until CLUSTAL line, but don't fail if it's not there
    try:
        while "CLUSTAL" not in next(lines_iter):
            pass
    except StopIteration:
        # Reset iterator if CLUSTAL keyword not found
        lines_iter = iter(text.splitlines())

    # Process blocks
    block_lines: List[Tuple[str, str]] = []
    for line in lines_iter:
        line = line.strip()
        if not line:
            if block_lines:
                # End of a block, process it
                if not records:  # First block
                    for seq_id, seq_part in block_lines:
                        records.append({"id": seq_id, "seq": seq_part})
                else:
                    if len(records) != len(block_lines):
                        raise ValueError("Inconsistent number of sequences in Clustal blocks.")
                    for i, (seq_id, seq_part) in enumerate(block_lines):
                        # The perl script warns if names don't match.
                        # We will just append based on order.
                        records[i]["seq"] += seq_part
                block_lines = []
            continue

        if line.startswith(("*", ":", ".")):
            continue

        parts = line.split()
        if len(parts) >= 2:
            block_lines.append((parts[0], parts[1]))

    # Process the last block
    if block_lines:
        if not records:  # First block
            for seq_id, seq_part in block_lines:
                records.append({"id": seq_id, "seq": seq_part})
        else:
            if len(records) != len(block_lines):
                raise ValueError("Inconsistent number of sequences in Clustal blocks.")
            for i, (seq_id, seq_part) in enumerate(block_lines):
                records[i]["seq"] += seq_part

    return records
