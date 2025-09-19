from pathlib import Path


def wrap_fasta_sequence(seq: str, width: int = 60) -> str:
    """
    Wrap a sequence string to fixed line width (default: 60).
    """
    if width <= 0:
        raise ValueError("width must be a positive integer")
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def write_fasta(path: str | Path, seq_id: str, seq: str, width: int = 60) -> None:
    """
    Write a single FASTA record with optional line wrapping.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wrapped = wrap_fasta_sequence(seq, width) if width else seq
    with p.open("w", encoding="utf-8") as f:
        f.write(f">{seq_id}\n{wrapped}\n")


def write_fasta_pair(
    path: str | Path,
    seq1_id: str,
    seq1: str,
    seq2_id: str,
    seq2: str,
    width: int | None = None,
) -> None:
    """
    Write two FASTA records to the same file (seq1 followed by seq2).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    w1 = wrap_fasta_sequence(seq1, width) if width else seq1
    w2 = wrap_fasta_sequence(seq2, width) if width else seq2
    with p.open("w", encoding="utf-8") as f:
        f.write(f">{seq1_id}\n{w1}\n>{seq2_id}\n{w2}\n")


def reconstruct_alignment(seq1: str, seq2: str, pred_alignment: list[list[int]]) -> tuple[str, str]:
    """
    Reconstructs the aligned sequences from the original sequences and the predicted alignment pairs.
    """
    aligned_seq1 = []
    aligned_seq2 = []

    match_pairs = sorted([tuple(p) for p in pred_alignment])

    last_i = -1
    last_j = -1

    for i, j in match_pairs:
        # Unaligned region before the current match
        gap_in_seq2 = seq1[last_i + 1 : i]
        gap_in_seq1 = seq2[last_j + 1 : j]

        # Add unaligned parts of seq1 (gaps in seq2)
        if gap_in_seq2:
            aligned_seq1.extend(list(gap_in_seq2))
            aligned_seq2.extend(["-"] * len(gap_in_seq2))

        # Add unaligned parts of seq2 (gaps in seq1)
        if gap_in_seq1:
            aligned_seq1.extend(["-"] * len(gap_in_seq1))
            aligned_seq2.extend(list(gap_in_seq1))

        # Add the matched pair
        aligned_seq1.append(seq1[i])
        aligned_seq2.append(seq2[j])

        last_i = i
        last_j = j

    # Tail part
    gap_in_seq2 = seq1[last_i + 1 :]
    if gap_in_seq2:
        aligned_seq1.extend(list(gap_in_seq2))
        aligned_seq2.extend(["-"] * len(gap_in_seq2))

    gap_in_seq1 = seq2[last_j + 1 :]
    if gap_in_seq1:
        aligned_seq1.extend(["-"] * len(gap_in_seq1))
        aligned_seq2.extend(list(gap_in_seq1))

    return "".join(aligned_seq1), "".join(aligned_seq2)
