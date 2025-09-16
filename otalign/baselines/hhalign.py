import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from otalign.io.parser import convert_a3m_text_to_a2m


def _get_hhr_line_regex_groups(regex_pattern: str, line: str) -> Sequence[Optional[str]]:
    match = re.match(regex_pattern, line)
    if match is None:
        raise RuntimeError(f"Could not parse query line {line}")
    return match.groups()


def _update_hhr_residue_indices_list(sequence: str, start_index: int, indices_list: List[int]):
    """Computes the relative indices for each residue with respect to the original sequence."""
    counter = start_index
    for symbol in sequence:
        if symbol == "-":
            indices_list.append(-1)
        else:
            indices_list.append(counter)
            counter += 1


def _parse_hhr_hit(detailed_lines: Sequence[str]) -> dict:
    """Parses the detailed HMM HMM comparison section for a single Hit.
    This works on .hhr files generated from both HHBlits and HHSearch.
    Args:
      detailed_lines: A list of lines from a single comparison section between 2
        sequences (which each have their own HMM's)
    Returns:
      A dictionary with the information from that detailed comparison section
    Raises:
      RuntimeError: If a certain line cannot be processed
    """
    # Parse first 2 lines.
    number_of_hit = int(detailed_lines[0].split()[-1])
    name_hit = detailed_lines[1][1:]

    # Parse the summary line.
    pattern = "Probab=(.*)[\t ]*E-value=(.*)[\t ]*Score=(.*)[\t ]*Aligned_cols=(.*)[\t ]*Identities=(.*)%[\t ]*Similarity=(.*)[\t ]*Sum_probs=(.*)[\t ]*Template_Neff=(.*)"
    match = re.match(pattern, detailed_lines[2])
    if match is None:
        raise RuntimeError("Could not parse section: %s. Expected this: \n%s to contain summary." % (detailed_lines, detailed_lines[2]))
    (prob_true, e_value, _, aligned_cols, _, _, sum_probs, neff) = [float(x) for x in match.groups()]

    # The next section reads the detailed comparisons. These are in a 'human
    # readable' format which has a fixed length. The strategy employed is to
    # assume that each block starts with the query sequence line, and to parse
    # that with a regexp in order to deduce the fixed length used for that block.
    query = ""
    hit_sequence = ""
    indices_query = []
    indices_hit = []
    length_block = None

    for line in detailed_lines[3:]:
        # Parse the query sequence line
        if line.startswith("Q ") and not line.startswith("Q ss_dssp") and not line.startswith("Q ss_pred") and not line.startswith("Q Consensus"):
            # Thus the first 17 characters must be 'Q <query_name> ', and we can parse
            # everything after that.
            #              start    sequence       end       total_sequence_length
            patt = r"[\t ]*([0-9]*) ([A-Z-]*)[\t ]*([0-9]*) \([0-9]*\)"
            groups = _get_hhr_line_regex_groups(patt, line[17:])

            # Get the length of the parsed block using the start and finish indices,
            # and ensure it is the same as the actual block length.
            start = int(groups[0]) - 1  # Make index zero based.
            delta_query = groups[1]
            end = int(groups[2])
            num_insertions = len([x for x in delta_query if x == "-"])
            length_block = end - start + num_insertions
            assert length_block == len(delta_query)

            # Update the query sequence and indices list.
            query += delta_query
            _update_hhr_residue_indices_list(delta_query, start, indices_query)

        elif line.startswith("T "):
            # Parse the hit sequence.
            if not line.startswith("T ss_dssp") and not line.startswith("T ss_pred") and not line.startswith("T Consensus"):
                # Thus the first 17 characters must be 'T <hit_name> ', and we can
                # parse everything after that.
                #              start    sequence       end     total_sequence_length
                patt = r"[\t ]*([0-9]*) ([A-Z-]*)[\t ]*[0-9]* \([0-9]*\)"
                groups = _get_hhr_line_regex_groups(patt, line[17:])
                start = int(groups[0]) - 1  # Make index zero based.
                delta_hit_sequence = groups[1]
                assert length_block == len(delta_hit_sequence)

                # Update the hit sequence and indices list.
                hit_sequence += delta_hit_sequence
                _update_hhr_residue_indices_list(delta_hit_sequence, start, indices_hit)

    return {
        "index": number_of_hit,
        "name": name_hit,
        "aligned_cols": int(aligned_cols),
        "sum_probs": sum_probs,
        "query": query,
        "hit_sequence": hit_sequence,
        "indices_query": indices_query,
        "indices_hit": indices_hit,
    }


def parse_hhr(hhr_string: str) -> Sequence[dict]:
    """Parses the content of an entire HHR file."""
    lines = hhr_string.splitlines()

    # Each .hhr file starts with a results table, then has a sequence of hit
    # "paragraphs", each paragraph starting with a line 'No <hit number>'. We
    # iterate through each paragraph to parse each hit.

    block_starts = [i for i, line in enumerate(lines) if line.startswith("No ")]

    hits = []
    if block_starts:
        block_starts.append(len(lines))  # Add the end of the final block.
        for i in range(len(block_starts) - 1):
            hits.append(_parse_hhr_hit(lines[block_starts[i] : block_starts[i + 1]]))
    return hits


def run_hhalign_hhm_pair(
    q_hhm: str | Path,
    t_hhm: str | Path,
    *,
    hhalign_bin: str = "hhalign",
    out_dir: Optional[str | Path] = None,
    mode: str = "local",  # "local" | "global" | "glocal"
    extra_args: Optional[List[str]] = None,
    timeout: Optional[int] = 300,
) -> Tuple[str, str]:
    """
    Run hhalign on two .hhm files and return pairwise alignment as two gapped A3M strings.
    We parse -oa3m and strip lowercase (insertions) per A3M convention.
    Robust 'mode' handling: try candidate flag patterns until one succeeds.
    """
    out_dir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_a3m = out_dir / "pair.aln.a3m"
    out_hhr = out_dir / "pair.aln.hhr"

    base_cmd = [hhalign_bin, "-i", str(q_hhm), "-t", str(t_hhm), "-oa3m", str(out_a3m), "-o", str(out_hhr)]

    # Known variants observed across HHsuite versions (not all builds support all flags).
    MODE_CANDIDATES = {
        "local": [
            [],  # default is typically local-ish
            ["-local", "1"],
            ["-global", "0"],
        ],
        "global": [
            ["-global", "1"],
            ["-glob", "1"],  # older alias in some builds
            ["-local", "0"],
        ],
        "glocal": [
            ["-glocal", "1"],  # present in some hhalign builds
            ["-local", "1", "-global", "1"],  # emulate if supported
        ],
    }

    if mode not in MODE_CANDIDATES:
        raise ValueError(f"Unknown mode={mode}. Choose from local|global|glocal")

    tried_errors: List[str] = []
    extra_args = list(extra_args or [])

    for cand in MODE_CANDIDATES[mode]:
        cmd = base_cmd + cand + extra_args
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=timeout)
            # success
            recs = convert_a3m_text_to_a2m(out_a3m.read_text())
            if len(recs) < 2:
                hhr_string = out_hhr.read_text()
                hits = parse_hhr(hhr_string)
                a1 = hits[0]["query"]
                a2 = hits[0]["hit_sequence"]
                # raise RuntimeError(f"hhalign produced no pairwise A3M: {out_a3m}")
            else:
                a1 = recs[0]["seq"].upper().replace(".", "-")
                a2 = recs[1]["seq"].upper().replace(".", "-")
            if len(strip_a3m_lowercase(a1)) != len(strip_a3m_lowercase(a2)):
                raise ValueError("Aligned strings have different lengths after stripping lowercase.")
            return a1, a2
        except subprocess.CalledProcessError:
            tried_errors.append(f"cmd failed: {' '.join(cmd)}")
        except Exception as e:
            tried_errors.append(str(e))

    raise RuntimeError(f"hhalign failed for mode={mode}. Tried {len(MODE_CANDIDATES[mode])} flag patterns.\n" + "\n".join(tried_errors[:4]) + ("\n..." if len(tried_errors) > 4 else ""))


def strip_a3m_lowercase(s: str) -> str:
    """
    In A3M, lowercase letters are insertions relative to the query/target.
    For a 2-sequence alignment, remove lowercase while keeping uppercase and '-'.
    """
    out = []
    for ch in s:
        if ch == "-" or ("A" <= ch <= "Z"):
            out.append(ch)
        # skip lowercase
    return "".join(out)
