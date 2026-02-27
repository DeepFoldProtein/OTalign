#!/usr/bin/env python3
"""
Convert ECOD distribution files (FASTA + hierarchy) to CSV format.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def parse_hierarchy_file(hierarchy_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse ECOD hierarchy file to build mapping from assignment to names.

    Format: level<tab>id<tab>name<tab><tab>count
    Where level is A, X, H, T, or F

    Returns:
        Dictionary mapping assignment strings (e.g., "1.1.1.3") to hierarchy names
    """
    hierarchy = {}

    with open(hierarchy_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            level = parts[0]  # A, X, H, T, or F
            level_id = parts[1]  # e.g., "a.1", "1.1.1.3"
            name = parts[2]  # e.g., "cradle loop barrel"

            # Build hierarchical keys
            if level == "A":
                # Architecture level (e.g., "a.1" -> id "1")
                hierarchy[level_id] = {"A": name}
            elif level == "X":
                # X-group level (e.g., "1.1")
                hierarchy[level_id] = {"X": name}
            elif level == "H":
                # H-group level (e.g., "1.1.1")
                hierarchy[level_id] = {"H": name}
            elif level == "T":
                # T-group level (e.g., "1.1.1.2")
                hierarchy[level_id] = {"T": name}
            elif level == "F":
                # F-group level (e.g., "1.1.1.2.1" or just "1.1.1.1")
                hierarchy[level_id] = {"F": name}

    return hierarchy


def get_hierarchy_names(assignment: str, hierarchy: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """
    Get A, X, H, T, F names for a given assignment.

    Args:
        assignment: e.g., "1.1.1.3"
        hierarchy: Parsed hierarchy mapping

    Returns:
        Dictionary with keys 'A', 'X', 'H', 'T', 'F'
    """
    parts = assignment.split(".")
    result = {"A": "NO_X_NAME", "X": "NO_X_NAME", "H": "NO_H_NAME", "T": "NO_T_NAME", "F": "NO_F_NAME"}

    # Try to find each level
    for i in range(len(parts), 0, -1):
        level_id = ".".join(parts[:i])
        if level_id in hierarchy:
            level_data = hierarchy[level_id]
            result.update(level_data)

    return result


def parse_fasta(fasta_path: Path, max_entries: int = None) -> List[Tuple[str, str]]:
    """
    Parse FASTA file manually (without BioPython).

    Returns:
        List of (header, sequence) tuples
    """
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous sequence if exists
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq)))
                    if max_entries and len(sequences) >= max_entries:
                        break

                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget the last sequence
        if current_header is not None and (max_entries is None or len(sequences) < max_entries):
            sequences.append((current_header, "".join(current_seq)))

    return sequences


def parse_fasta_header_full(header: str) -> Tuple[str, str, str, str]:
    """
    Parse full ECOD FASTA header.

    Example: e2nmzA1 uid:0 range:A:1-99 assignment:1.1.1.3
    (Note: '>' already removed)

    Returns:
        (domain_id, uid, range_str, assignment)
    """
    parts = header.split()
    domain_id = parts[0]

    uid = ""
    range_str = ""
    assignment = ""

    for part in parts[1:]:
        if part.startswith("uid:"):
            uid = part.split(":", 1)[1]
        elif part.startswith("range:"):
            range_str = part.split(":", 1)[1]
        elif part.startswith("assignment:"):
            assignment = part.split(":", 1)[1]

    return domain_id, uid, range_str, assignment


def parse_fasta_header_f40(header: str) -> Tuple[str, str, str, str]:
    """
    Parse ECOD F40 FASTA header (e.g. ecod.v293.1.F40.fa).

    Example: 1|e1hvcA1|1.1.1.3
    (Note: '>' already removed)

    Returns:
        (domain_id, uid, range_str, assignment)
    """
    parts = header.split("|")
    if len(parts) >= 3:
        uid = parts[0].strip()
        domain_id = parts[1].strip()
        assignment = parts[2].strip()
        return domain_id, uid, "", assignment
    # Fallback: treat whole header as domain_id
    return header.strip(), "", "", ""


def convert_ecod_to_csv(fasta_path: Path, hierarchy_path: Path, output_csv: Path, max_entries: int = None, format: str = "full"):
    """
    Convert ECOD FASTA and hierarchy files to CSV format.

    Args:
        fasta_path: Path to ECOD FASTA file
        hierarchy_path: Path to ecod.v293.1.hierarchy.txt
        output_csv: Path to output CSV file
        max_entries: Maximum number of entries to process (None for all)
        format: "full" (uid/range/assignment headers) or "f40" (num|domain_id|assignment)
    """
    if format not in ("full", "f40"):
        raise ValueError("format must be 'full' or 'f40'")
    parse_header = parse_fasta_header_full if format == "full" else parse_fasta_header_f40

    print(f"Parsing hierarchy file: {hierarchy_path}")
    hierarchy = parse_hierarchy_file(hierarchy_path)
    print(f"Loaded {len(hierarchy)} hierarchy entries")

    print(f"\nParsing FASTA file: {fasta_path} (format={format})")
    print("This may take a few minutes...")

    sequences = parse_fasta(fasta_path, max_entries)
    print(f"Loaded {len(sequences)} sequences")

    # Write CSV
    print(f"\nWriting to {output_csv}...")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "description", "sequence"])

        for i, (header, seq) in enumerate(sequences):
            domain_id, uid, range_str, assignment = parse_header(header)

            # Get hierarchy names
            names = get_hierarchy_names(assignment, hierarchy)

            # Build description string in expected format
            description = f"ECOD_{uid}_{domain_id} | {assignment} | {range_str} | A: {names['A']}, X: {names['X']}, H: {names['H']}, T: {names['T']}, F: {names['F']}"

            writer.writerow([domain_id, description, seq])

            if (i + 1) % 10000 == 0:
                print(f"Wrote {i + 1} sequences...")

    print(f"Done! CSV file created with {len(sequences)} entries")

    # Print sample
    print("\nSample entries (first 3):")
    with open(output_csv, "r") as f:
        reader = csv.reader(f)
        header_row = next(reader)
        print(f"Columns: {', '.join(header_row)}")
        for i, row in enumerate(reader):
            if i >= 3:
                break
            print(f"\n{i + 1}. ID: {row[0]}")
            print(f"   Description: {row[1][:100]}...")
            print(f"   Sequence: {row[2][:50]}...")


def main():
    parser = argparse.ArgumentParser(description="Convert ECOD distribution files to CSV format")
    parser.add_argument("--fasta", type=Path, default=Path("/store/database/ecod/ecod.v293.1.fa"), help="Path to ECOD FASTA file")
    parser.add_argument("--hierarchy", type=Path, default=Path("/store/database/ecod/ecod.v293.1.hierarchy.txt"), help="Path to ECOD hierarchy file")
    parser.add_argument("--output", type=Path, default=Path("/store/database/ecod/ecod.csv"), help="Path to output CSV file")
    parser.add_argument("--max-entries", type=int, default=None, help="Maximum number of entries to process (for testing)")
    parser.add_argument("--format", type=str, choices=("full", "f40"), default="full", help="FASTA header format: full (ecod.v293.1.fa) or f40 (ecod.v293.1.F40.fa)")

    args = parser.parse_args()

    convert_ecod_to_csv(args.fasta, args.hierarchy, args.output, args.max_entries, format=args.format)


if __name__ == "__main__":
    main()
