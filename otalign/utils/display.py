import logging
import re
import sys
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch


def is_notebook() -> bool:
    """Checks if the code is running in a Jupyter Notebook."""
    try:
        # This function is available in Jupyter environments
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Not in an IPython environment


# --- Conditional Imports and Color Definitions for Rendering ---

IN_NOTEBOOK = is_notebook()

if IN_NOTEBOOK:
    from IPython.display import Markdown, display


# ANSI escape codes for terminal colors
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET_ALL = "\033[0m"


Fore = Colors
Style = Colors

# BLOSUM62 matrix for scoring amino acid substitutions.
# A positive score indicates a conservative substitution.
BLOSUM62 = {
    "A": {
        "A": 4,
        "R": -1,
        "N": -2,
        "D": -2,
        "C": 0,
        "Q": -1,
        "E": -1,
        "G": 0,
        "H": -2,
        "I": -1,
        "L": -1,
        "K": -1,
        "M": -1,
        "F": -2,
        "P": -1,
        "S": 1,
        "T": 0,
        "W": -3,
        "Y": -2,
        "V": 0,
    },
    "R": {
        "A": -1,
        "R": 5,
        "N": 0,
        "D": -2,
        "C": -3,
        "Q": 1,
        "E": 0,
        "G": -2,
        "H": 0,
        "I": -3,
        "L": -2,
        "K": 2,
        "M": -1,
        "F": -3,
        "P": -2,
        "S": -1,
        "T": -1,
        "W": -3,
        "Y": -2,
        "V": -3,
    },
    "N": {
        "A": -2,
        "R": 0,
        "N": 6,
        "D": 1,
        "C": -3,
        "Q": 0,
        "E": 0,
        "G": 0,
        "H": 1,
        "I": -3,
        "L": -3,
        "K": 0,
        "M": -2,
        "F": -3,
        "P": -2,
        "S": 1,
        "T": 0,
        "W": -4,
        "Y": -2,
        "V": -3,
    },
    "D": {
        "A": -2,
        "R": -2,
        "N": 1,
        "D": 6,
        "C": -3,
        "Q": 0,
        "E": 2,
        "G": -1,
        "H": -1,
        "I": -3,
        "L": -4,
        "K": -1,
        "M": -3,
        "F": -3,
        "P": -1,
        "S": 0,
        "T": -1,
        "W": -4,
        "Y": -3,
        "V": -3,
    },
    "C": {
        "A": 0,
        "R": -3,
        "N": -3,
        "D": -3,
        "C": 9,
        "Q": -3,
        "E": -4,
        "G": -3,
        "H": -3,
        "I": -1,
        "L": -1,
        "K": -3,
        "M": -1,
        "F": -2,
        "P": -3,
        "S": -1,
        "T": -1,
        "W": -2,
        "Y": -2,
        "V": -1,
    },
    "Q": {
        "A": -1,
        "R": 1,
        "N": 0,
        "D": 0,
        "C": -3,
        "Q": 5,
        "E": 2,
        "G": -2,
        "H": 0,
        "I": -3,
        "L": -2,
        "K": 1,
        "M": 0,
        "F": -3,
        "P": -1,
        "S": 0,
        "T": -1,
        "W": -2,
        "Y": -1,
        "V": -2,
    },
    "E": {
        "A": -1,
        "R": 0,
        "N": 0,
        "D": 2,
        "C": -4,
        "Q": 2,
        "E": 5,
        "G": -2,
        "H": 0,
        "I": -3,
        "L": -3,
        "K": 1,
        "M": -2,
        "F": -3,
        "P": -1,
        "S": 0,
        "T": -1,
        "W": -3,
        "Y": -2,
        "V": -2,
    },
    "G": {
        "A": 0,
        "R": -2,
        "N": 0,
        "D": -1,
        "C": -3,
        "Q": -2,
        "E": -2,
        "G": 6,
        "H": -2,
        "I": -4,
        "L": -4,
        "K": -2,
        "M": -3,
        "F": -3,
        "P": -2,
        "S": 0,
        "T": -2,
        "W": -2,
        "Y": -3,
        "V": -3,
    },
    "H": {
        "A": -2,
        "R": 0,
        "N": 1,
        "D": -1,
        "C": -3,
        "Q": 0,
        "E": 0,
        "G": -2,
        "H": 8,
        "I": -3,
        "L": -3,
        "K": -1,
        "M": -2,
        "F": -1,
        "P": -2,
        "S": -1,
        "T": -2,
        "W": -2,
        "Y": 2,
        "V": -3,
    },
    "I": {
        "A": -1,
        "R": -3,
        "N": -3,
        "D": -3,
        "C": -1,
        "Q": -3,
        "E": -3,
        "G": -4,
        "H": -3,
        "I": 4,
        "L": 2,
        "K": -3,
        "M": 1,
        "F": 0,
        "P": -3,
        "S": -2,
        "T": -1,
        "W": -3,
        "Y": -1,
        "V": 3,
    },
    "L": {
        "A": -1,
        "R": -2,
        "N": -3,
        "D": -4,
        "C": -1,
        "Q": -2,
        "E": -3,
        "G": -4,
        "H": -3,
        "I": 2,
        "L": 4,
        "K": -2,
        "M": 2,
        "F": 0,
        "P": -3,
        "S": -2,
        "T": -1,
        "W": -2,
        "Y": -1,
        "V": 1,
    },
    "K": {
        "A": -1,
        "R": 2,
        "N": 0,
        "D": -1,
        "C": -3,
        "Q": 1,
        "E": 1,
        "G": -2,
        "H": -1,
        "I": -3,
        "L": -2,
        "K": 5,
        "M": -1,
        "F": -3,
        "P": -1,
        "S": 0,
        "T": -1,
        "W": -3,
        "Y": -2,
        "V": -2,
    },
    "M": {
        "A": -1,
        "R": -1,
        "N": -2,
        "D": -3,
        "C": -1,
        "Q": 0,
        "E": -2,
        "G": -3,
        "H": -2,
        "I": 1,
        "L": 2,
        "K": -1,
        "M": 5,
        "F": 0,
        "P": -2,
        "S": -1,
        "T": -1,
        "W": -1,
        "Y": -1,
        "V": 1,
    },
    "F": {
        "A": -2,
        "R": -3,
        "N": -3,
        "D": -3,
        "C": -2,
        "Q": -3,
        "E": -3,
        "G": -3,
        "H": -1,
        "I": 0,
        "L": 0,
        "K": -3,
        "M": 0,
        "F": 6,
        "P": -4,
        "S": -2,
        "T": -2,
        "W": 1,
        "Y": 3,
        "V": -1,
    },
    "P": {
        "A": -1,
        "R": -2,
        "N": -2,
        "D": -1,
        "C": -3,
        "Q": -1,
        "E": -1,
        "G": -2,
        "H": -2,
        "I": -3,
        "L": -3,
        "K": -1,
        "M": -2,
        "F": -4,
        "P": 7,
        "S": -1,
        "T": -1,
        "W": -4,
        "Y": -3,
        "V": -2,
    },
    "S": {
        "A": 1,
        "R": -1,
        "N": 1,
        "D": 0,
        "C": -1,
        "Q": 0,
        "E": 0,
        "G": 0,
        "H": -1,
        "I": -2,
        "L": -2,
        "K": 0,
        "M": -1,
        "F": -2,
        "P": -1,
        "S": 4,
        "T": 1,
        "W": -3,
        "Y": -2,
        "V": -2,
    },
    "T": {
        "A": 0,
        "R": -1,
        "N": 0,
        "D": -1,
        "C": -1,
        "Q": -1,
        "E": -1,
        "G": -2,
        "H": -2,
        "I": -1,
        "L": -1,
        "K": -1,
        "M": -1,
        "F": -2,
        "P": -1,
        "S": 1,
        "T": 5,
        "W": -2,
        "Y": -2,
        "V": 0,
    },
    "W": {
        "A": -3,
        "R": -3,
        "N": -4,
        "D": -4,
        "C": -2,
        "Q": -2,
        "E": -3,
        "G": -2,
        "H": -2,
        "I": -3,
        "L": -2,
        "K": -3,
        "M": -1,
        "F": 1,
        "P": -4,
        "S": -3,
        "T": -2,
        "W": 11,
        "Y": 2,
        "V": -3,
    },
    "Y": {
        "A": -2,
        "R": -2,
        "N": -2,
        "D": -3,
        "C": -2,
        "Q": -1,
        "E": -2,
        "G": -3,
        "H": 2,
        "I": -1,
        "L": -1,
        "K": -2,
        "M": -1,
        "F": 3,
        "P": -3,
        "S": -2,
        "T": -2,
        "W": 2,
        "Y": 7,
        "V": -1,
    },
    "V": {
        "A": 0,
        "R": -3,
        "N": -3,
        "D": -3,
        "C": -1,
        "Q": -2,
        "E": -2,
        "G": -3,
        "H": -3,
        "I": 3,
        "L": 1,
        "K": -2,
        "M": 1,
        "F": -1,
        "P": -2,
        "S": -2,
        "T": 0,
        "W": -3,
        "Y": -1,
        "V": 4,
    },
}


def extract_alignment_intersect(
    path: List[Tuple[int, int, str]],
    query_map: Optional[Dict[int, int]] = None,
    template_map: Optional[Dict[int, int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Extracts the part of a full alignment path that corresponds to a specified
    query and/or template region of interest.

    Args:
        path: The alignment path for the full sequences.
        query_map: A mapping of {sequence index: PDB index} for the query's region of interest.
        template_map: A mapping of {sequence index: PDB index} for the template's region of interest.

    Returns:
        A dictionary with alignment information for the region of interest, or None if no alignment exists.
    """
    if not query_map and not template_map:
        # If no region is specified, do nothing (or could return the full path)
        return None

    valid_q_indices = set(query_map.keys()) if query_map else None
    valid_t_indices = set(template_map.keys()) if template_map else None

    filtered_path = []
    for i, j, move in path:
        # Apply filtering conditions based on the move type
        keep = False
        if move == "M":
            # Both query and template must be in the region of interest
            in_q = not valid_q_indices or i in valid_q_indices
            in_t = not valid_t_indices or j in valid_t_indices
            if in_q and in_t:
                keep = True
        elif move == "I":  # Insertion in query (gap in template)
            # Only check if the query is in the region of interest
            if not valid_q_indices or i in valid_q_indices:
                keep = True
        elif move == "D":  # Deletion in query (insertion in template)
            # Only check if the template is in the region of interest
            if not valid_t_indices or j in valid_t_indices:
                keep = True

        if keep:
            filtered_path.append((i, j, move))

    if not filtered_path:
        return None

    # Calculate coordinates based on the filtered path
    query_indices = sorted([i for i, j, move in filtered_path if move in ("M", "I")])
    template_indices = sorted([j for i, j, move in filtered_path if move in ("M", "D")])

    query_coords = (query_indices[0], query_indices[-1]) if query_indices else None
    template_coords = (template_indices[0], template_indices[-1]) if template_indices else None

    return {"filtered_path": filtered_path, "query_coords": query_coords, "template_coords": template_coords}


def print_alignment(
    query: str,
    templ: str,
    path: List[Tuple[int, int, str]],
    display_mode: str = "full_query",
    max_width: int = 80,
    max_label_width: int = 12,
    outfile: TextIO = sys.stdout,
    query_label: Optional[str] = None,
    templ_label: Optional[str] = None,
    query_map: Optional[Dict[int, int]] = None,
    template_map: Optional[Dict[int, int]] = None,
) -> tuple[str, str, dict, dict]:
    """Prints a sequence alignment and returns statistics and original sequence coordinate ranges.

    If `segment_info` is provided, `display_mode` is treated as 'core',
    and coordinate calculations use values from `segment_info`.

    Args:
        query: The query sequence.
        templ: The template sequence.
        path: A list of (i, j, move) tuples, where i and j are 1-based coordinates.
        display_mode: The display mode ('core', 'full_query', 'full_template').
            - 'core': Prints only the aligned core (formerly trim_gaps=True).
            - 'full_query': Pads template's terminal gaps with '-' to match the full query length (default).
            - 'full_template': Pads query's terminal gaps with '-' to match the full template length,
                             and prints unaligned ends of the template in lowercase.
            - 'global_view': Prints all unaligned parts of both sequences in lowercase.
        max_width: The maximum number of alignment columns per line.
        max_label_width: The maximum length for labels; longer labels are truncated with "...".
        outfile: The file object to write the output to (default: sys.stdout).
        query_label: A custom label for the query sequence (default: 'Query').
        templ_label: A custom label for the template sequence (default: 'Templ').
    Returns:
        A tuple: (aligned_query_str, aligned_template_str, stats_dict, coords_dict)
            coords_dict: {'query': (start, end), 'templ': (start, end)} (1-based)
    """
    if not path:
        stats = {
            "alignment_length": 0,
            "aligned_positions": 0,
            "identity": 0.0,
            "similarity": 0.0,
            "matches": 0,
            "conservative": 0,
            "mismatch": 0,
            "gaps": 0,
        }
        print("Empty path.", file=outfile)
        return "", "", stats, {"query": (0, 0), "templ": (0, 0)}

    # 1. Process labels
    label_q = query_label if query_label is not None else "Query"
    label_t = templ_label if templ_label is not None else "Templ"

    if len(label_q) > max_label_width:
        label_q = label_q[: max_label_width - 3] + "..."
    if len(label_t) > max_label_width:
        label_t = label_t[: max_label_width - 3] + "..."

    # 2. Generate the core alignment (same as before)
    core_aligned_q, core_aligned_t, core_match_line, core_pos_q, core_pos_t = [], [], [], [], []
    match_count, conservative_count, mismatch_count, gap_count = 0, 0, 0, 0

    for i, j, move in path:
        if move == "M":
            c_q, c_t = query[i - 1].upper(), templ[j - 1].upper()
            score = BLOSUM62.get(c_q, {}).get(c_t, -100)
            if c_q == c_t:
                core_aligned_q.append(f"{Fore.GREEN}{c_q}{Style.RESET_ALL}")
                core_aligned_t.append(f"{Fore.GREEN}{c_t}{Style.RESET_ALL}")
                core_match_line.append(":")
                match_count += 1
            elif score > 0:
                core_aligned_q.append(f"{Fore.YELLOW}{c_q}{Style.RESET_ALL}")
                core_aligned_t.append(f"{Fore.YELLOW}{c_t}{Style.RESET_ALL}")
                core_match_line.append("+")
                conservative_count += 1
            else:
                core_aligned_q.append(f"{Fore.RED}{c_q}{Style.RESET_ALL}")
                core_aligned_t.append(f"{Fore.RED}{c_t}{Style.RESET_ALL}")
                core_match_line.append(".")
                mismatch_count += 1
            core_pos_q.append(i)
            core_pos_t.append(j)
        elif move == "I":
            core_aligned_q.append(query[i - 1])
            core_aligned_t.append("-")
            core_match_line.append(" ")
            gap_count += 1
            core_pos_q.append(i)
            core_pos_t.append(None)
        elif move == "D":
            core_aligned_q.append("-")
            core_aligned_t.append(templ[j - 1])
            core_match_line.append(" ")
            gap_count += 1
            core_pos_q.append(None)
            core_pos_t.append(j)

    # 3. Generate the final alignment based on display_mode
    # Statistics and coordinates are calculated based on the 'core' alignment regardless of display_mode.
    core_query_pos = [p for p in core_pos_q if p is not None]
    core_templ_pos = [p for p in core_pos_t if p is not None]
    query_range = (core_query_pos[0], core_query_pos[-1]) if core_query_pos else None
    templ_range = (core_templ_pos[0], core_templ_pos[-1]) if core_templ_pos else None
    coords = {"query": query_range, "templ": templ_range}

    aligned_count = match_count + conservative_count + mismatch_count
    identity = (match_count / aligned_count * 100) if aligned_count > 0 else 0.0
    similarity = ((match_count + conservative_count) / aligned_count * 100) if aligned_count > 0 else 0.0

    if display_mode == "core":
        final_aligned_q, final_aligned_t, final_match_line, final_pos_q, final_pos_t = (
            core_aligned_q,
            core_aligned_t,
            core_match_line,
            core_pos_q,
            core_pos_t,
        )
    elif display_mode == "full_query":
        start_q_pos = path[0][0]
        prefix_len = start_q_pos - 1
        last_q_pos = max(p for p in core_pos_q if p is not None)
        suffix_len = len(query) - last_q_pos

        final_aligned_q = list(query[:prefix_len]) + core_aligned_q + list(query[len(query) - suffix_len :])
        final_aligned_t = ["-"] * prefix_len + core_aligned_t + ["-"] * suffix_len
        final_match_line = [" "] * prefix_len + core_match_line + [" "] * suffix_len
        final_pos_q = list(range(1, start_q_pos)) + core_pos_q + list(range(last_q_pos + 1, len(query) + 1))
        final_pos_t = [None] * prefix_len + core_pos_t + [None] * suffix_len
    elif display_mode == "full_template":
        # Handle the unaligned prefix of the template
        start_t_pos = path[0][1]
        prefix_templ = templ[: start_t_pos - 1]
        prefix_q = ["-"] * len(prefix_templ)
        prefix_match = [" "] * len(prefix_templ)
        prefix_pos_q = [None] * len(prefix_templ)
        prefix_pos_t = list(range(1, start_t_pos))

        # Handle the unaligned suffix of the template
        last_t_pos = max(p for p in core_pos_t if p is not None)
        suffix_templ = templ[last_t_pos:]
        suffix_q = ["-"] * len(suffix_templ)
        suffix_match = [" "] * len(suffix_templ)
        suffix_pos_q = [None] * len(suffix_templ)
        suffix_pos_t = list(range(last_t_pos + 1, len(templ) + 1))

        # Combine everything
        final_aligned_q = prefix_q + core_aligned_q + suffix_q
        final_aligned_t = list(prefix_templ.lower()) + core_aligned_t + list(suffix_templ.lower())
        final_match_line = prefix_match + core_match_line + suffix_match
        final_pos_q = prefix_pos_q + core_pos_q + suffix_pos_q
        final_pos_t = prefix_pos_t + core_pos_t + suffix_pos_t
    elif display_mode == "global_view":
        # Handle prefix
        start_q_pos, start_t_pos = path[0][0], path[0][1]
        prefix_q_str = query[: start_q_pos - 1]
        prefix_t_str = templ[: start_t_pos - 1]
        prefix_len = max(len(prefix_q_str), len(prefix_t_str))

        prefix_q = list(prefix_q_str.lower().ljust(prefix_len, "-"))
        prefix_t = list(prefix_t_str.lower().ljust(prefix_len, "-"))
        prefix_match = [" "] * prefix_len
        prefix_pos_q = list(range(1, len(prefix_q_str) + 1)) + [None] * (prefix_len - len(prefix_q_str))
        prefix_pos_t = list(range(1, len(prefix_t_str) + 1)) + [None] * (prefix_len - len(prefix_t_str))

        # Handle suffix
        last_q_pos = coords["query"][1] if coords["query"] else 0
        last_t_pos = coords["templ"][1] if coords["templ"] else 0
        suffix_q_str = query[last_q_pos:]
        suffix_t_str = templ[last_t_pos:]
        suffix_len = max(len(suffix_q_str), len(suffix_t_str))

        suffix_q = list(suffix_q_str.lower().ljust(suffix_len, "-"))
        suffix_t = list(suffix_t_str.lower().ljust(suffix_len, "-"))
        suffix_match = [" "] * suffix_len
        suffix_pos_q = list(range(last_q_pos + 1, len(query) + 1)) + [None] * (suffix_len - len(suffix_q_str))
        suffix_pos_t = list(range(last_t_pos + 1, len(templ) + 1)) + [None] * (suffix_len - len(suffix_t_str))

        # Combine everything
        final_aligned_q = prefix_q + core_aligned_q + suffix_q
        final_aligned_t = prefix_t + core_aligned_t + suffix_t
        final_match_line = prefix_match + core_match_line + suffix_match
        final_pos_q = prefix_pos_q + core_pos_q + suffix_pos_q
        final_pos_t = prefix_pos_t + core_pos_t + suffix_pos_t
    else:
        raise ValueError(f"Wrong display_mode: '{display_mode}'.")

    alignment_length = len(final_aligned_q)

    # 4. Print results
    output_buffer = []
    for start in range(0, alignment_length, max_width):
        end = min(start + max_width, alignment_length)
        seg_q = "".join(final_aligned_q[start:end])
        seg_m = "".join(final_match_line[start:end])
        seg_t = "".join(final_aligned_t[start:end])

        # Calculate and convert query coordinates to PDB numbers
        block_pos_q = [p for p in final_pos_q[start:end] if p is not None]
        if query_map:
            block_pos_q = [query_map.get(p, p) for p in block_pos_q]
        start_q = block_pos_q[0] if block_pos_q else None
        end_q = block_pos_q[-1] if block_pos_q else None
        num_q = f"({start_q}-{end_q})" if start_q is not None else "(---)"

        # Calculate and convert template coordinates to PDB numbers
        block_pos_t = [p for p in final_pos_t[start:end] if p is not None]
        if template_map:
            block_pos_t = [template_map.get(p, p) for p in block_pos_t]
        start_t = block_pos_t[0] if block_pos_t else None
        end_t = block_pos_t[-1] if block_pos_t else None
        num_t = f"({start_t}-{end_t})" if start_t is not None else "(---)"

        label_space = " " * (max_label_width + 1)

        # Change output format for notebook vs. terminal environments
        if IN_NOTEBOOK:
            # Convert ANSI codes to HTML for notebook display
            seg_q_html = (
                seg_q.replace(Colors.GREEN, '<span style="color: green;">')
                .replace(Colors.YELLOW, '<span style="color: orange;">')
                .replace(Colors.RED, '<span style="color: red;">')
                .replace(Colors.RESET_ALL, "</span>")
            )
            seg_t_html = (
                seg_t.replace(Colors.GREEN, '<span style="color: green;">')
                .replace(Colors.YELLOW, '<span style="color: orange;">')
                .replace(Colors.RED, '<span style="color: red;">')
                .replace(Colors.RESET_ALL, "</span>")
            )
            output_buffer.append(
                f"<pre style='font-family: monospace; line-height: 1.2;'>"
                f"{label_q:{max_label_width}s} {num_q:>10} : {seg_q_html}<br>"
                f"{label_space}{' ' * 10} : {seg_m}<br>"
                f"{label_t:{max_label_width}s} {num_t:>10} : {seg_t_html}"
                f"</pre>"
            )
        else:
            print(f"{label_q:{max_label_width}s} {num_q:>10} : {seg_q}", file=outfile)
            print(f"{label_space}{' ' * 10} : {seg_m}", file=outfile)
            print(f"{label_t:{max_label_width}s} {num_t:>10} : {seg_t}", file=outfile)
            print(file=outfile)
    if IN_NOTEBOOK:
        display(Markdown("".join(output_buffer)))

    # 5. Print statistics
    stats_lines = [
        f"Alignment length       : {alignment_length}",
        f"Aligned positions      : {aligned_count}",
        f"Identical matches      : {match_count}",
        f"Conservative matches   : {conservative_count}",
        f"Non-conservative       : {mismatch_count}",
        f"Gaps                   : {gap_count}",
        f"Identity               : {identity:.2f}%",
        f"Similarity (cons.)     : {similarity:.2f}%",
    ]
    # Branch output for notebook/terminal (uncomment if needed)
    for line in stats_lines:
        print(line, file=outfile)

    # Create and return statistics dictionary
    stats = {
        "alignment_length": alignment_length,
        "aligned_positions": aligned_count,
        "identity": round(identity, 2),
        "similarity": round(similarity, 2),
        "matches": match_count,
        "conservative": conservative_count,
        "mismatch": mismatch_count,
        "gaps": gap_count,
    }

    query_positions = [p for p in final_pos_q if p is not None]
    templ_positions = [p for p in final_pos_t if p is not None]

    query_range = (query_positions[0], query_positions[-1]) if query_positions else None
    templ_range = (templ_positions[0], templ_positions[-1]) if templ_positions else None

    coords = {"query": query_range, "templ": templ_range}

    # Create pure alignment strings without color codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    final_out_q = ansi_escape.sub("", "".join(final_aligned_q))
    final_out_t = ansi_escape.sub("", "".join(final_aligned_t))

    return final_out_q, final_out_t, stats, coords


def format_fasta_from_path(
    query: str,
    templ: str,
    path: List[Tuple[int, int, str]],
):
    """
    Aligns two sequences based on an alignment path and formats them
    as a gapped FASTA string.

    Args:
        query: The original query sequence.
        templ: The original template sequence.
        path: The alignment path as a list of (i, j, move) tuples.
        query_id: The query ID for the FASTA header.
        template_id: The template ID for the FASTA header.
        line_width: The maximum line width for the sequence.
        outfile: The file object to write the output to (default: sys.stdout).
    """
    aligned_q_chars = []
    aligned_t_chars = []

    # 1. Create two aligned sequence strings by following the path.
    for i, j, move in path:
        if move == "M":  # Match/Mismatch
            aligned_q_chars.append(query[i - 1])
            aligned_t_chars.append(templ[j - 1])
        elif move == "I":  # Insertion in Query (gap in template)
            aligned_q_chars.append(query[i - 1])
            aligned_t_chars.append("-")
        elif move == "D":  # Deletion in Query (gap in query)
            aligned_q_chars.append("-")
            aligned_t_chars.append(templ[j - 1])

    aligned_query = "".join(aligned_q_chars)
    aligned_template = "".join(aligned_t_chars)

    return aligned_query, aligned_template


def format_a3m_from_path(query: str, templ: str, path: List[Tuple[int, int, str]]) -> tuple[str, str]:
    """
    Generates a correct A3M-formatted template sequence using a path with coordinate information.
    Correctly handles partial alignments (q2t, local, etc.).

    Args:
        query: The query sequence (returned unchanged).
        templ: The template sequence.
        path: A list of (i, j, move) tuples, where i and j are 1-based coordinates.
              Example: [(3, 1, 'M'), (4, 2, 'M'), ...]
    """
    # 1. If there is no alignment path, the template is all gaps.
    if not path:
        return query, "-" * len(query)

    # 2. Calculate the number of gaps to place at the beginning (prefix) of the alignment.
    # The query coordinate (i) of the first element in the path is the start of the alignment.
    start_query_pos = path[0][0]  # 1-based index
    prefix_gaps = "-" * (start_query_pos - 1)

    # 3. Create the core part of the alignment path.
    core_aligned_templ = []
    last_query_pos = 0  # Track the last query position used in the alignment
    for i, j, move in path:
        if move == "M":  # Match/Mismatch
            core_aligned_templ.append(templ[j - 1].upper())
            last_query_pos = i
        elif move == "D":  # Deletion in Query -> Insertion in template
            core_aligned_templ.append(templ[j - 1].lower())
        elif move == "I":  # Insertion in Query -> Deletion in template
            core_aligned_templ.append("-")
            last_query_pos = i

    # 4. Calculate the number of gaps to place at the end (suffix) of the alignment.
    suffix_gaps = "-" * (len(query) - last_query_pos)

    # 5. Combine the prefix gaps, core alignment, and suffix gaps to create the final template.
    final_aligned_template = prefix_gaps + "".join(core_aligned_templ) + suffix_gaps

    return query, final_aligned_template


def format_a3m(query: str, templ: str, path: str) -> tuple[str, str]:
    """
    Generates a correct A3M-formatted template sequence using a standard alignment path ('M', 'I', 'D').

    - 'M': Match/Mismatch (consumes a character from both query and template)
    - 'I': Insertion in Query (consumes a character from query, adds '-' to template)
    - 'D': Deletion in Query (consumes a character from template, adds a lowercase character to template)
    """
    aligned_templ = []
    query_idx = 0
    templ_idx = 0

    for move in path:
        if move == "M":
            # Query and template are aligned (Match/Mismatch)
            aligned_templ.append(templ[templ_idx].upper())
            query_idx += 1
            templ_idx += 1
        elif move == "I":
            # Insertion in query -> gap (-) in template (Deletion in template)
            aligned_templ.append("-")
            query_idx += 1
        elif move == "D":
            # Deletion in query -> insertion (lowercase) in template (Insertion in template)
            aligned_templ.append(templ[templ_idx].lower())
            templ_idx += 1
        else:
            raise ValueError(f"Invalid move '{move}' in path")

    # Check if all characters have been consumed
    # assert query_idx == len(query)
    assert templ_idx == len(templ)

    # Return the original query and the aligned version of the template, as per A3M format
    return query, "".join(aligned_templ)


def aln_to_matches(aln1, aln2):
    if all("-" not in x for x in (aln1, aln2)):
        return []

    matches = []
    i, j = 0, 0
    for a, b in zip(aln1, aln2):
        if a != "-" and b != "-":
            matches.append((i, j))
            i += 1
            j += 1
        elif a != "-":
            i += 1
        elif b != "-":
            j += 1
        else:
            continue

    return matches


def convolve2d(image, kernel, padding_mode="valid", stride=1):
    """
    Performs 2D convolution of an image with a given kernel.
    Optimized for performance using NumPy's vectorized operations.

    Args:
        image (np.ndarray): The input 2D image (grayscale).
        kernel (np.ndarray): The 2D convolution kernel.
        padding_mode (str): The padding mode to use.
                            'valid': No padding. Output size will be smaller.
                            'same': Pads the image so that the output size is the same as the input.
                                    Pads with zeros.
                            'full': Pads the image so that every pixel of the input is covered
                                    by the kernel. Output size will be larger.
                            'reflect', 'symmetric', 'edge', 'wrap': Other NumPy padding modes.
                                    (Note: 'same' and 'full' specifically use 'constant' padding with 0s)
        stride (int): The stride of the convolution. Defaults to 1.

    Returns:
        np.ndarray: The convolved image.

    Raises:
        ValueError: If the padding_mode is not recognized or if stride is less than 1.
    """

    if stride < 1:
        raise ValueError("Stride must be at least 1.")

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding amounts based on mode
    if padding_mode == "valid":
        pad_height_top, pad_height_bottom = 0, 0
        pad_width_left, pad_width_right = 0, 0
    elif padding_mode == "same":
        # Calculate total padding needed
        total_pad_height = kernel_height - 1
        total_pad_width = kernel_width - 1

        # Distribute padding: prefer more padding on the bottom/right if odd
        pad_height_top = total_pad_height // 2
        pad_height_bottom = total_pad_height - pad_height_top
        pad_width_left = total_pad_width // 2
        pad_width_right = total_pad_width - pad_width_left
    elif padding_mode == "full":
        pad_height_top, pad_height_bottom = kernel_height - 1, kernel_height - 1
        pad_width_left, pad_width_right = kernel_width - 1, kernel_width - 1
    else:
        # For other NumPy padding modes, we apply padding to achieve a 'same'-like output size
        # if the user doesn't explicitly specify padding amounts.
        # This is a heuristic to make these modes behave reasonably for convolution.
        total_pad_height = kernel_height - 1
        total_pad_width = kernel_width - 1
        pad_height_top = total_pad_height // 2
        pad_height_bottom = total_pad_height - pad_height_top
        pad_width_left = total_pad_width // 2
        pad_width_right = total_pad_width - pad_width_left

    # Apply padding
    if padding_mode == "valid":
        padded_image = image
    elif padding_mode in ["same", "full"]:
        # For 'same' and 'full', we use constant padding with zeros
        padded_image = np.pad(
            image,
            ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)),
            mode="constant",
            constant_values=0,
        )
    else:
        # For other specified NumPy padding modes
        try:
            padded_image = np.pad(
                image,
                ((pad_height_top, pad_height_bottom), (pad_width_left, pad_width_right)),
                mode=padding_mode,
            )
        except ValueError as e:
            raise ValueError(f"Invalid padding_mode '{padding_mode}' or padding calculation error: {e}")

    padded_image_height, padded_image_width = padded_image.shape

    # Calculate output dimensions
    output_height = (padded_image_height - kernel_height) // stride + 1
    output_width = (padded_image_width - kernel_width) // stride + 1

    # Ensure output dimensions are non-negative
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Output dimensions are non-positive. Check image, kernel, padding, and stride values.")

    # Get strides of the padded image
    image_strides = padded_image.strides

    # Create a view of the padded image that represents all possible kernel-sized windows
    # This is the core of the vectorized approach.
    # The new shape will be (output_height, output_width, kernel_height, kernel_width)
    # The new strides will allow us to slide the window by 'stride' for the first two dimensions
    # and then access elements within the kernel for the last two dimensions.
    view_shape = (output_height, output_width, kernel_height, kernel_width)
    view_strides = (
        image_strides[0] * stride,  # Stride for moving down rows in the output
        image_strides[1] * stride,  # Stride for moving right columns in the output
        image_strides[0],  # Stride for moving down rows within the kernel window
        image_strides[1],  # Stride for moving right columns within the kernel window
    )

    # Use as_strided to create the view without copying data
    sub_matrices = np.lib.stride_tricks.as_strided(padded_image, shape=view_shape, strides=view_strides)

    # Perform element-wise multiplication with the kernel and sum along the kernel dimensions
    # np.einsum is very efficient for this type of operation (sum-product)
    # 'ijkl,kl->ij' means:
    # i,j: output dimensions
    # k,l: kernel dimensions
    # For each (i,j) in the output, sum over k,l of (sub_matrices[i,j,k,l] * kernel[k,l])
    output = np.einsum("ijkl,kl->ij", sub_matrices, kernel)

    return output


def setup_logging():
    """Configures basic logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def stack_and_mask_arrays(
    arrays: List[np.ndarray],
    max_length: int,
    padding_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pads and stacks a list of NumPy arrays of variable length to create a tensor and a mask.
    """
    if not arrays or arrays[0] is None:
        # Return empty tensors if data loading fails
        return torch.empty(0), torch.empty(0)

    batch_size = len(arrays)
    feature_dim = arrays[0].shape[1]

    stacked_tensor = torch.full((batch_size, max_length, feature_dim), padding_value, dtype=torch.float32)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

    for i, arr in enumerate(arrays):
        if arr is None:
            continue  # Skip samples that failed to load
        copy_len = min(arr.shape[0], max_length)
        stacked_tensor[i, :copy_len, :] = torch.from_numpy(arr[:copy_len, :])
        mask[i, :copy_len] = True

    return stacked_tensor, mask
