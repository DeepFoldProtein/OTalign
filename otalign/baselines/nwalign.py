import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from otalign.io.fasta_utils import write_fasta
from otalign.io.parser import gapped_to_pairs


def parse_nwalign_stdout(text: str) -> Tuple[str, str]:
    """
    Parse NWalign output and return the two aligned strings (with gaps).
    We look for the final alignment block(s) consisting of three lines:
      seq1_aligned
      match_line (':', spaces)
      seq2_aligned
    NWalign may wrap; we concatenate contiguous blocks.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    # Collect alignment section(s) after the summary; strategy:
    # Take all lines that contain only letters, '-' and spaces or ':'.
    # We then reassemble as triplets (seq1, match, seq2) possibly repeated across wrapped blocks.
    blocks: List[Tuple[str, str, str]] = []

    def is_seq_line(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        # letters + dashes only
        return all((c.isalpha() or c == "-") for c in s)

    def is_match_line(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        # ':' and spaces only (NWalign uses ':' for identical)
        return all((c == ":" or c == " ") for c in s)

    i = 0
    while i < len(lines):
        s = lines[i]
        if is_seq_line(s):
            seq1 = s.strip().replace(" ", "")
            # next should be match line
            if i + 1 < len(lines) and is_match_line(lines[i + 1]):
                match = lines[i + 1].strip()
                if i + 2 < len(lines) and is_seq_line(lines[i + 2]):
                    seq2 = lines[i + 2].strip().replace(" ", "")
                    blocks.append((seq1, match, seq2))
                    i += 3
                    continue
        i += 1

    if not blocks:
        raise ValueError("Could not find alignment block in NWalign output.")

    # Concatenate wrapped blocks
    seq1_aln = "".join(b[0] for b in blocks)
    seq2_aln = "".join(b[2] for b in blocks)
    if len(seq1_aln) != len(seq2_aln):
        raise ValueError("Aligned strings have different lengths.")

    return seq1_aln, seq2_aln


def run_nwalign_for_pair(
    seq1_id: str,
    seq1: str,
    seq2_id: str,
    seq2: str,
    *,
    nwalign_bin: str = "NWalign",
    infmt1: int = 4,
    infmt2: int = 4,
    glocal: int = 0,
    extra_args: Optional[List[str]] = None,
    tmpdir: Optional[Path] = None,
    timeout: Optional[int] = 120,
) -> Tuple[List[Tuple[int, int]], str, str]:
    """
    Run NWalign externally and return: (pairs, seq1_aligned, seq2_aligned).
    - infmt 4: FASTA
    - glocal 0: global (Needleman-Wunsch); 1: local-global (glocal)
    """
    extra_args = extra_args or []
    close_tmp = False
    if tmpdir is None:
        t = tempfile.TemporaryDirectory()
        tmpdir = Path(t.name)
        close_tmp = True
    else:
        tmpdir.mkdir(parents=True, exist_ok=True)

    f1 = tmpdir / "q.fasta"
    f2 = tmpdir / "t.fasta"
    write_fasta(f1, seq1_id, seq1)  # NWalign expects one seq per file; keep id simple
    write_fasta(f2, seq2_id, seq2)

    cmd = [
        nwalign_bin,
        str(f1),
        str(f2),
        "-infmt1",
        str(infmt1),
        "-infmt2",
        str(infmt2),
        "-glocal",
        str(glocal),
    ] + extra_args

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"NWalign failed (code {e.returncode}): {e.output}") from e
    finally:
        if close_tmp:
            t.cleanup()  # type: ignore

    seq1_aln, seq2_aln = parse_nwalign_stdout(out)
    pairs = gapped_to_pairs(seq1_aln, seq2_aln)
    return pairs, seq1_aln, seq2_aln
