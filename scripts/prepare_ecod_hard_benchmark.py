"""
Prepare ECOD "Hard" Benchmark Dataset (pLM-BLAST Paper Style).

This script replicates the benchmark construction from the pLM-BLAST paper:
1. Filter domains by length (50-600 aa)
2. Use ECOD30 (30% seq-id clustered via MMSeqs2, matching the paper)
3. Filter H-groups with >= 5 members
4. Randomly select 300 H-groups
5. From each H-group, select 5 domains (preferring different T-groups for diversity)
6. Final dataset: 1,500 domains (300 H-groups x 5 domains)

Hierarchy parsing uses the NUMERICAL assignment field (2nd pipe-delimited part of
the description), e.g. "907.1.1.1" → X=907, H=907.1, T=907.1.1.
This is required because many entries have placeholder text names like "NO_H_NAME".

Ground truth for evaluation:
- True Positive: same H-group
- False Positive: different X-group
- Neutral (excluded from PR/ROC): same X-group but different H-group
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_ecod_hierarchy(description: str) -> Dict[str, str]:
    """
    Parse ECOD hierarchy from the NUMERICAL assignment in the description.

    The 2nd pipe-delimited field contains the numerical hierarchy, e.g.:
      "907.1.1.1" → X=907, H=907.1, T=907.1.1
      "64.1.1"    → X=64,  H=64.1,  T=64.1.1  (3-field format also valid)

    Using numerical IDs is required because many entries have placeholder
    text names like "NO_H_NAME" or "NO_X_NAME" in the 4th field.
    """
    hierarchy = {"X": "", "H": "", "T": ""}

    try:
        if " | " not in description:
            return hierarchy

        parts = description.split(" | ")
        if len(parts) < 2:
            return hierarchy

        num_str = parts[1].strip()  # e.g., "907.1.1.1" or "64.1.1"
        fields = num_str.split(".")

        if len(fields) < 3:
            return hierarchy

        hierarchy["X"] = fields[0]  # "907"
        hierarchy["H"] = f"{fields[0]}.{fields[1]}"  # "907.1"
        hierarchy["T"] = f"{fields[0]}.{fields[1]}.{fields[2]}"  # "907.1.1"

    except Exception as e:
        logging.warning(f"Failed to parse hierarchy: {e}")

    return hierarchy


def load_and_filter_ecod(csv_path: Path, min_length: int = 50, max_length: int = 600) -> pd.DataFrame:
    """
    Load ECOD data and apply initial filters.

    Args:
        csv_path: Path to ECOD CSV file
        min_length: Minimum sequence length (paper: 50)
        max_length: Maximum sequence length (paper: 600)

    Returns:
        Filtered DataFrame with parsed hierarchy
    """
    logging.info(f"Loading ECOD data from {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"  Total sequences: {len(df)}")

    # Add sequence length
    df["seq_length"] = df["sequence"].str.len()

    # Filter by length
    df_filtered = df[(df["seq_length"] >= min_length) & (df["seq_length"] <= max_length)].copy()
    logging.info(f"  After length filter ({min_length}-{max_length} aa): {len(df_filtered)}")

    # Parse hierarchy
    logging.info("Parsing ECOD hierarchy...")
    hierarchies = []
    for desc in tqdm(df_filtered["description"], desc="Parsing"):
        hierarchies.append(parse_ecod_hierarchy(desc))

    hierarchy_df = pd.DataFrame(hierarchies)
    df_filtered = pd.concat([df_filtered.reset_index(drop=True), hierarchy_df], axis=1)

    # Filter out entries with missing numerical hierarchy
    valid_mask = (df_filtered["H"] != "") & (df_filtered["X"] != "")
    df_filtered = df_filtered[valid_mask].reset_index(drop=True)
    logging.info(f"  After hierarchy filter: {len(df_filtered)}")

    # Statistics
    logging.info(f"  Unique X-groups: {df_filtered['X'].nunique()}")
    logging.info(f"  Unique H-groups: {df_filtered['H'].nunique()}")
    logging.info(f"  Unique T-groups: {df_filtered['T'].nunique()}")

    return df_filtered


def select_hard_benchmark_domains(
    df: pd.DataFrame, num_h_groups: int = 300, domains_per_h_group: int = 5, min_h_group_size: int = 5, prefer_diverse_t_groups: bool = True, seed: int = 42
) -> pd.DataFrame:
    """
    Select domains for the "Hard" benchmark following paper methodology.

    Args:
        df: Filtered ECOD DataFrame
        num_h_groups: Number of H-groups to select (paper: 300)
        domains_per_h_group: Domains per H-group (paper: 5)
        min_h_group_size: Minimum H-group size (paper: >= 5)
        prefer_diverse_t_groups: Prefer different T-groups within H-group
        seed: Random seed

    Returns:
        DataFrame with selected domains (num_h_groups * domains_per_h_group)
    """
    random.seed(seed)

    # Find H-groups with enough members
    h_group_counts = df["H"].value_counts()
    eligible_h_groups = h_group_counts[h_group_counts >= min_h_group_size].index.tolist()

    logging.info(f"Found {len(eligible_h_groups)} H-groups with >= {min_h_group_size} members")

    if len(eligible_h_groups) < num_h_groups:
        logging.warning(f"Only {len(eligible_h_groups)} eligible H-groups available (requested {num_h_groups}). Using all available.")
        num_h_groups = len(eligible_h_groups)

    # Randomly select H-groups
    selected_h_groups = random.sample(eligible_h_groups, num_h_groups)
    logging.info(f"Selected {len(selected_h_groups)} H-groups")

    # Select domains from each H-group
    selected_domains = []

    for h_group in tqdm(selected_h_groups, desc="Selecting domains"):
        h_group_df = df[df["H"] == h_group].copy()

        if prefer_diverse_t_groups:
            # Group by T-group and try to select from different T-groups
            domains_by_t = defaultdict(list)

            for idx, row in h_group_df.iterrows():
                domains_by_t[row["T"]].append(idx)

            # Round-robin selection from different T-groups
            selected_indices = []
            t_group_list = list(domains_by_t.keys())
            random.shuffle(t_group_list)

            t_idx = 0
            while len(selected_indices) < domains_per_h_group and any(domains_by_t.values()):
                t_group = t_group_list[t_idx % len(t_group_list)]
                if domains_by_t[t_group]:
                    idx = random.choice(domains_by_t[t_group])
                    domains_by_t[t_group].remove(idx)
                    selected_indices.append(idx)
                t_idx += 1

                # Break if we've cycled through all T-groups with no candidates
                if t_idx > len(t_group_list) * domains_per_h_group:
                    break

            selected_domains.extend(selected_indices[:domains_per_h_group])
        else:
            # Simple random selection
            indices = h_group_df.index.tolist()
            random.shuffle(indices)
            selected_domains.extend(indices[:domains_per_h_group])

    result_df = df.loc[selected_domains].reset_index(drop=True)

    # Verify statistics
    logging.info(f"Selected {len(result_df)} domains total")
    logging.info(f"  - Unique H-groups: {result_df['H'].nunique()}")
    logging.info(f"  - Unique T-groups: {result_df['T'].nunique()}")
    logging.info(f"  - Unique X-groups: {result_df['X'].nunique()}")
    logging.info(f"  - Avg domains per H-group: {len(result_df) / result_df['H'].nunique():.2f}")

    # Length statistics
    logging.info(f"  - Min length: {result_df['seq_length'].min()}")
    logging.info(f"  - Max length: {result_df['seq_length'].max()}")
    logging.info(f"  - Mean length: {result_df['seq_length'].mean():.1f}")

    return result_df


def save_benchmark_dataset(df: pd.DataFrame, output_dir: Path, name: str = "hard_benchmark"):
    """
    Save the benchmark dataset in multiple formats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full CSV with all metadata
    csv_path = output_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV: {csv_path}")

    # Save FASTA
    fasta_path = output_dir / f"{name}.fasta"
    with open(fasta_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['id']}\n{row['sequence']}\n")
    logging.info(f"Saved FASTA: {fasta_path}")

    # Save metadata JSON
    metadata = {
        "name": name,
        "description": "ECOD Hard Benchmark (pLM-BLAST paper style)",
        "num_domains": len(df),
        "num_h_groups": df["H"].nunique(),
        "num_t_groups": df["T"].nunique(),
        "num_x_groups": df["X"].nunique(),
        "domains_per_h_group": len(df) / df["H"].nunique() if df["H"].nunique() > 0 else 0,
        "length_range": [int(df["seq_length"].min()), int(df["seq_length"].max())],
        "length_mean": float(df["seq_length"].mean()),
        "evaluation_criteria": {"true_positive": "same H-group", "false_positive": "different X-group", "neutral": "same X-group, different H-group"},
        "h_group_distribution": df["H"].value_counts().to_dict(),
        "x_group_distribution": df["X"].value_counts().to_dict(),
    }

    metadata_path = output_dir / f"{name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata: {metadata_path}")

    return csv_path, fasta_path, metadata_path


def analyze_expected_comparisons(df: pd.DataFrame):
    """
    Analyze expected comparisons for all-vs-all benchmark.
    """
    n = len(df)
    total_pairs = n * (n - 1) // 2  # Upper triangle excluding diagonal

    # Count expected TPs (same H-group pairs)
    h_group_counts = df["H"].value_counts()
    expected_tp = sum(c * (c - 1) // 2 for c in h_group_counts)

    # Count expected neutrals (same X, different H)
    # For each X-group, count pairs that are same X but different H
    expected_neutral = 0
    for x_group in df["X"].unique():
        x_df = df[df["X"] == x_group]
        x_count = len(x_df)
        x_total_pairs = x_count * (x_count - 1) // 2

        # Same H-group pairs within this X-group
        h_counts_in_x = x_df["H"].value_counts()
        same_h_pairs = sum(c * (c - 1) // 2 for c in h_counts_in_x)

        # Same X, different H = total X pairs - same H pairs
        expected_neutral += x_total_pairs - same_h_pairs

    expected_fp = total_pairs - expected_tp - expected_neutral

    logging.info("\nExpected comparison statistics (all-vs-all):")
    logging.info(f"  Total domain pairs: {total_pairs:,}")
    logging.info(f"  Expected True Positives (same H): {expected_tp:,} ({100 * expected_tp / total_pairs:.2f}%)")
    logging.info(f"  Expected Neutrals (same X, diff H): {expected_neutral:,} ({100 * expected_neutral / total_pairs:.2f}%)")
    logging.info(f"  Expected False Positives (diff X): {expected_fp:,} ({100 * expected_fp / total_pairs:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ECOD 'Hard' Benchmark Dataset (pLM-BLAST paper style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Standard paper configuration (300 H-groups x 5 domains = 1,500 domains)
  python prepare_ecod_hard_benchmark.py \\
      --ecod_csv /store/database/ecod/plmblast_ecod30/ECOD30.csv \\
      --output_dir data/ecod30_hard

  # Smaller test set
  python prepare_ecod_hard_benchmark.py \\
      --ecod_csv /store/database/ecod/plmblast_ecod30/ECOD30.csv \\
      --output_dir data/ecod30_hard_mini --num_h_groups 30 --domains_per_h_group 3
        """,
    )

    parser.add_argument("--ecod_csv", type=str, required=True, help="Path to ECOD CSV file (id, description, sequence)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for benchmark dataset")
    parser.add_argument("--num_h_groups", type=int, default=300, help="Number of H-groups to select (paper: 300)")
    parser.add_argument("--domains_per_h_group", type=int, default=5, help="Domains to select per H-group (paper: 5)")
    parser.add_argument("--min_h_group_size", type=int, default=5, help="Minimum H-group size (paper: >= 5)")
    parser.add_argument("--min_length", type=int, default=50, help="Minimum sequence length (paper: 50)")
    parser.add_argument("--max_length", type=int, default=600, help="Maximum sequence length (paper: 600)")
    parser.add_argument("--no_diverse_t_groups", action="store_true", help="Disable T-group diversity selection (paper prefers diverse T-groups)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Load and filter
    df = load_and_filter_ecod(Path(args.ecod_csv), min_length=args.min_length, max_length=args.max_length)

    # Select domains
    benchmark_df = select_hard_benchmark_domains(
        df, num_h_groups=args.num_h_groups, domains_per_h_group=args.domains_per_h_group, min_h_group_size=args.min_h_group_size, prefer_diverse_t_groups=not args.no_diverse_t_groups, seed=args.seed
    )

    # Analyze expected comparisons
    analyze_expected_comparisons(benchmark_df)

    # Save
    save_benchmark_dataset(benchmark_df, Path(args.output_dir), name="hard_benchmark")

    logging.info("\nBenchmark dataset prepared successfully!")
    logging.info(f"Total comparisons for all-vs-all: {len(benchmark_df) * (len(benchmark_df) - 1) // 2:,}")


if __name__ == "__main__":
    main()
