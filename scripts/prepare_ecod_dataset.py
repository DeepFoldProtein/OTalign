"""
Prepare ECOD dataset for homolog benchmark (pLM-BLAST paper convention).

Uses H-group for query selection and labels: TP = same H-group, FP = different X-group,
Neutral = same X different H (excluded from PR/ROC).
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def download_ecod_data(output_dir: Path, ecod_version: str = "latest") -> Path:
    """
    Download ECOD domain list from EBI/RCSB.

    Args:
        output_dir: Directory to save downloaded data
        ecod_version: ECOD version (default: "latest")

    Returns:
        Path to downloaded ECOD file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ECOD distribution URL (example - adjust based on actual source)
    # The actual ECOD data can be obtained from:
    # http://prodata.swmed.edu/ecod/distributions/

    ecod_url = f"http://prodata.swmed.edu/ecod/distributions/ecod.{ecod_version}.domains.txt"
    output_file = output_dir / f"ecod_{ecod_version}_domains.txt"

    if output_file.exists():
        logging.info(f"ECOD data already exists at {output_file}")
        return output_file

    logging.info(f"Downloading ECOD data from {ecod_url}")
    try:
        response = requests.get(ecod_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(output_file, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading ECOD") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        logging.info(f"Downloaded ECOD data to {output_file}")
        return output_file

    except Exception as e:
        logging.error(f"Failed to download ECOD data: {e}")
        logging.info("Please manually download ECOD data from http://prodata.swmed.edu/ecod/")
        raise


def parse_ecod_hierarchy(description: str) -> Dict[str, str]:
    """
    Parse ECOD hierarchy from description string.

    Expected format: "ECOD_uid_domain | assignment | range | A: ..., X: ..., H: ..., T: ..., F: ... | ..."

    Args:
        description: ECOD description string

    Returns:
        Dictionary with A, X, H, T, F groups
    """
    hierarchy = {"A": "", "X": "", "H": "", "T": "", "F": ""}

    try:
        # Extract the hierarchy section
        if " | " not in description:
            return hierarchy

        parts = description.split(" | ")
        if len(parts) < 4:
            return hierarchy

        # The hierarchy is in the 4th part (index 3): "A: ..., X: ..., H: ..., T: ..., F: ..."
        hierarchy_str = parts[3] if len(parts) >= 4 else ""

        for key in ["A", "X", "H", "T", "F"]:
            if f"{key}: " in hierarchy_str:
                start = hierarchy_str.find(f"{key}: ") + len(f"{key}: ")
                end = hierarchy_str.find(", ", start)
                if end == -1:
                    hierarchy[key] = hierarchy_str[start:].strip()
                else:
                    hierarchy[key] = hierarchy_str[start:end].strip()

    except Exception as e:
        logging.warning(f"Failed to parse hierarchy from: {description[:100]}... Error: {e}")

    return hierarchy


def load_ecod_csv(csv_path: Path, max_sequences: Optional[int] = None) -> pd.DataFrame:
    """
    Load ECOD data from CSV file.

    Expected columns: id, description, sequence

    Args:
        csv_path: Path to ECOD CSV file
        max_sequences: Maximum number of sequences to load (for testing)

    Returns:
        DataFrame with ECOD data including parsed hierarchy
    """
    logging.info(f"Loading ECOD data from {csv_path}")
    df = pd.read_csv(csv_path)

    if max_sequences:
        df = df.head(max_sequences)
        logging.info(f"Limited to {max_sequences} sequences for testing")

    # Parse hierarchy from description
    logging.info("Parsing ECOD hierarchy...")
    hierarchies = []
    for desc in tqdm(df["description"], desc="Parsing hierarchy"):
        hierarchies.append(parse_ecod_hierarchy(desc))

    hierarchy_df = pd.DataFrame(hierarchies)
    df = pd.concat([df, hierarchy_df], axis=1)

    # Filter out entries with no valid H-group (paper: TP = same H-group)
    valid_mask = (df["H"] != "") & (df["H"] != "NO_H_NAME")
    df = df[valid_mask].reset_index(drop=True)

    logging.info(f"Loaded {len(df)} ECOD domains with valid H-groups")

    # Print statistics
    logging.info(f"  - Unique X-groups: {df['X'].nunique()}")
    logging.info(f"  - Unique H-groups: {df['H'].nunique()}")

    return df


def create_query_db_split(df: pd.DataFrame, num_queries: Optional[int] = 100, min_group_size: int = 5, seed: int = 42, all_queries: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ECOD data into query and database sets for benchmark.
    Uses H-group (paper: TP = same H-group, FP = different X-group).

    Strategy:
    - Select queries from diverse H-groups (with sufficient members)
    - Remove query sequences from database to avoid trivial matches

    Args:
        df: ECOD DataFrame (must have H, X columns)
        num_queries: Number of query sequences (ignored if all_queries=True)
        min_group_size: Minimum number of members in H-group to be considered
        seed: Random seed
        all_queries: If True, use one query per valid H-group (no cap)

    Returns:
        (query_df, db_df) tuple
    """
    import numpy as np

    np.random.seed(seed)

    # Filter H-groups with enough members
    h_group_counts = df["H"].value_counts()
    valid_h_groups = h_group_counts[h_group_counts >= min_group_size].index.tolist()

    logging.info(f"Found {len(valid_h_groups)} H-groups with >= {min_group_size} members")

    if all_queries:
        n_use = len(valid_h_groups)
        logging.info(f"Using all queries: one per H-group ({n_use} queries)")
    else:
        n_use = num_queries

    df_filtered = df[df["H"].isin(valid_h_groups)].copy()

    # Sample queries: one from each H-group, cycling through groups
    queries = []
    h_groups_cycle = valid_h_groups.copy()
    np.random.shuffle(h_groups_cycle)

    group_idx = 0
    while len(queries) < n_use and len(df_filtered) > 0:
        h_group = h_groups_cycle[group_idx % len(h_groups_cycle)]
        group_members = df_filtered[df_filtered["H"] == h_group]

        if len(group_members) > 0:
            query_idx = np.random.choice(group_members.index)
            queries.append(query_idx)
            df_filtered = df_filtered.drop(query_idx)

        group_idx += 1

        if group_idx > len(h_groups_cycle) * n_use:
            logging.warning("Could not sample enough queries, breaking early")
            break

    query_df = df.loc[queries].reset_index(drop=True)
    db_indices = df.index.difference(queries)
    db_df = df.loc[db_indices].reset_index(drop=True)

    logging.info(f"Created query set: {len(query_df)} sequences")
    logging.info(f"Created database set: {len(db_df)} sequences")

    query_h_counts = query_df["H"].value_counts()
    logging.info(f"  - Query H-groups: {len(query_h_counts)} unique groups")
    logging.info(f"  - Avg queries per H-group: {len(query_df) / len(query_h_counts):.2f}")

    return query_df, db_df


def save_fasta(df: pd.DataFrame, output_path: Path):
    """Save DataFrame to FASTA format."""
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['id']}\n")
            f.write(f"{row['sequence']}\n")
    logging.info(f"Saved {len(df)} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare ECOD dataset for homolog/analog benchmark")
    parser.add_argument("--ecod_csv", type=str, required=True, help="Path to ECOD CSV file (with columns: id, description, sequence)")
    parser.add_argument("--output_dir", type=str, default="data/ecod", help="Output directory for prepared dataset")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of query sequences to sample (ignored if --all_queries)")
    parser.add_argument("--all_queries", action="store_true", help="Use one query per valid H-group (no limit; recommended for full evaluation)")
    parser.add_argument("--min_group_size", type=int, default=5, help="Minimum H-group size for query selection (paper: >=5 members)")
    parser.add_argument("--max_sequences", type=int, default=None, help="Maximum total sequences to load (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load ECOD data
    ecod_csv = Path(args.ecod_csv)
    if not ecod_csv.exists():
        logging.error(f"ECOD CSV file not found: {ecod_csv}")
        logging.info("Please provide a valid ECOD CSV file with columns: id, description, sequence")
        sys.exit(1)

    df = load_ecod_csv(ecod_csv, max_sequences=args.max_sequences)

    # Create query/db split (H-group based, paper convention)
    query_df, db_df = create_query_db_split(df, num_queries=args.num_queries, min_group_size=args.min_group_size, seed=args.seed, all_queries=args.all_queries)

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full metadata
    query_df.to_csv(output_dir / "queries.csv", index=False)
    db_df.to_csv(output_dir / "database.csv", index=False)

    # Save FASTA files
    save_fasta(query_df, output_dir / "queries.fasta")
    save_fasta(db_df, output_dir / "database.fasta")

    # Save metadata JSON for easy loading
    import json

    metadata = {
        "total_sequences": len(df),
        "num_queries": len(query_df),
        "num_database": len(db_df),
        "unique_h_groups_total": df["H"].nunique(),
        "unique_h_groups_queries": query_df["H"].nunique(),
        "min_group_size": args.min_group_size,
        "seed": args.seed,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Dataset preparation complete! Files saved to {output_dir}")
    logging.info("Next steps:")
    logging.info("  1. Build pLM-BLAST database: python scripts/build_ecod_plmblast_db.py")
    logging.info("  2. Run benchmark: python -m benchmark run --dataset ecod_homolog")


if __name__ == "__main__":
    main()
