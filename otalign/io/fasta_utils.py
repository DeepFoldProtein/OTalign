from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_fasta_sequences(fasta_path: str | Path) -> List[Tuple[str, str]]:
    """
    Read multiple sequences from a FASTA file.

    Returns:
        List of (sequence_id, sequence) tuples
    """
    sequences = []
    path = Path(fasta_path)

    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    current_id = None
    current_seq = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous sequence if exists
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq)))

                # Start new sequence
                current_id = line[1:].split()[0]  # Take first word after >
                current_seq = []
            else:
                if current_id is not None:
                    current_seq.append(line)

        # Save last sequence
        if current_id is not None:
            sequences.append((current_id, "".join(current_seq)))

    return sequences


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


def write_a3m_msa(
    output_path: str | Path,
    query_id: str,
    query_seq: str,
    target_alignments: List[Tuple[str, str, List[List[int]]]],
    width: int = 80,
    alignment_metadata: Optional[List[Dict[str, Any]]] = None,
    filter_params: Optional[Dict[str, float]] = None,
) -> None:
    """
    Write multiple sequence alignment in A3M format.

    Args:
        output_path: Path to output A3M file
        query_id: Query sequence ID
        query_seq: Query sequence (no gaps)
        target_alignments: List of (target_id, target_seq, alignment_pairs) tuples
        width: Line width for sequence wrapping
        alignment_metadata: Optional list of metadata dicts for each target alignment
        filter_params: Optional dict with filtering parameters (w_transport, w_sinkhorn, bias)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        # Write query sequence first (no gaps) with filtering parameters in header
        if filter_params:
            w_transport = filter_params.get("w_transport", "N/A")
            w_sinkhorn = filter_params.get("w_sinkhorn", "N/A")
            bias = filter_params.get("bias", "N/A")
            filter_threshold = filter_params.get("filter_threshold", "N/A")

            query_header = f">{query_id} w_transport={w_transport} w_sinkhorn={w_sinkhorn} bias={bias} filter_threshold={filter_threshold}\n"
            f.write(query_header)
        else:
            f.write(f">{query_id}\n")
        if width:
            wrapped_query = wrap_fasta_sequence(query_seq, width)
            f.write(f"{wrapped_query}\n")
        else:
            f.write(f"{query_seq}\n")

        # Write aligned target sequences
        for i, (target_id, target_seq, alignment_pairs) in enumerate(target_alignments):
            aligned_query, aligned_target = reconstruct_alignment(query_seq, target_seq, alignment_pairs)

            # Convert to A3M format: query gaps become deletions (lowercase), target gaps become gaps (-)
            a3m_target = convert_to_a3m_target(aligned_query, aligned_target)

            # Build header with metadata if provided
            if alignment_metadata and i < len(alignment_metadata):
                metadata = alignment_metadata[i]
                transport_cost = metadata.get("transport_cost", "N/A")
                sinkhorn_divergence = metadata.get("sinkhorn_divergence", "N/A")
                logistic_prob = metadata.get("filter_score", "N/A")
                passes_filter = metadata.get("passes_filter", "N/A")

                # Format metrics in header
                header = f">{target_id} transport_cost={transport_cost} sinkhorn_divergence={sinkhorn_divergence} logistic_prob={logistic_prob} passes_filter={passes_filter}\n"
                f.write(header)
            else:
                f.write(f">{target_id}\n")
            if width:
                wrapped_target = wrap_fasta_sequence(a3m_target, width)
                f.write(f"{wrapped_target}\n")
            else:
                f.write(f"{a3m_target}\n")


def convert_to_a3m_target(aligned_query: str, aligned_target: str) -> str:
    """
    Convert aligned target sequence to A3M format.

    In A3M format:
    - Positions where query has residue and target has residue: uppercase target residue
    - Positions where query has residue and target has gap: deletion (-)
    - Positions where query has gap and target has residue: lowercase target residue (insertion)
    - Positions where both have gaps: skip

    Args:
        aligned_query: Query sequence with gaps
        aligned_target: Target sequence with gaps

    Returns:
        A3M formatted target sequence
    """
    if len(aligned_query) != len(aligned_target):
        raise ValueError("Aligned sequences must have the same length")

    a3m_sequence = []

    for q_char, t_char in zip(aligned_query, aligned_target):
        if q_char != "-":  # Query has residue
            if t_char != "-":  # Target has residue -> match (uppercase)
                a3m_sequence.append(t_char.upper())
            else:  # Target has gap -> deletion (-)
                a3m_sequence.append("-")
        else:  # Query has gap
            if t_char != "-":  # Target has residue -> insertion (lowercase)
                a3m_sequence.append(t_char.lower())
            # Both have gaps -> skip (don't add anything)

    return "".join(a3m_sequence)
