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
