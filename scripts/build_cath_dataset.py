#!/usr/bin/env python3
import argparse
import json
import logging
import multiprocessing as mp
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm


def gapped_to_data(aln1: str, aln2: str) -> Tuple[str, str, List[Tuple[int, int]], float]:
    """
    Convert gapped aligned strings into:
      - ungapped seq1, seq2
      - index pairs in 0-based ungapped coordinates
      - percent identity over aligned (non-gap) positions
    """
    if len(aln1) != len(aln2):
        raise ValueError("Aligned sequences must have the same length.")
    # Build ungapped sequences
    ung1 = [a for a in aln1 if a != "-"]
    ung2 = [b for b in aln2 if b != "-"]
    seq1 = "".join(ung1)
    seq2 = "".join(ung2)

    # Walk alignment to build index pairs and identity
    i = j = 0
    pairs: List[Tuple[int, int]] = []
    matches = 0
    aligned = 0
    for a, b in zip(aln1, aln2):
        a_is = a != "-"
        b_is = b != "-"
        if a_is and b_is:
            pairs.append((i, j))
            aligned += 1
            if a.upper() == b.upper():
                matches += 1
            i += 1
            j += 1
        elif a_is and not b_is:
            i += 1
        elif not a_is and b_is:
            j += 1
        else:
            # gap-gap; no index advance
            pass
    pid = (matches / aligned * 100.0) if aligned > 0 else 0.0
    return seq1, seq2, pairs, pid


def parse_tmalign_output(output: str) -> Dict[str, float]:
    """Parses the stdout of TMalign to extract scores."""
    scores = {}

    # For lines like: "Aligned length= 94, RMSD=   3.26, Seq_ID=n_identical/n_aligned= 0.309"
    aligned_info_re = re.compile(r"Aligned length=\s*(\d+),\s*RMSD=\s*([0-9\.]+),\s*Seq_ID=.*=\s*([0-9\.]+)")

    # For lines like: "TM-score= 0.54199 (normalized by length of Structure_1: L=135, d0=4.32)"
    tm_score_re = re.compile(r"TM-score=\s*([0-9\.]+)\s*\(normalized by length of Structure_(\d):.*\)")

    for line in output.splitlines():
        aligned_match = aligned_info_re.match(line.strip())
        if aligned_match:
            scores["aligned_length"] = int(aligned_match.group(1))
            scores["rmsd"] = float(aligned_match.group(2))
            scores["frac_identity"] = float(aligned_match.group(3))

        tm_match = tm_score_re.match(line.strip())
        if tm_match:
            score = float(tm_match.group(1))
            struct_idx = int(tm_match.group(2))
            scores[f"tm_score_{struct_idx}"] = score

    # A third TM-score (normalized by average length) might be present
    avg_tm_re = re.compile(r"TM-score=\s*([0-9\.]+)\s*\(if normalized by average length of chains\)")
    for line in output.splitlines():
        avg_match = avg_tm_re.match(line.strip())
        if avg_match:
            scores["tm_score_avg"] = float(avg_match.group(1))

    return scores


# --- 1. CATH Data Parsing ---


def parse_domain_id_list(path: Path) -> set[str]:
    """Parses a file containing a list of domain IDs, one per line."""
    with open(path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def parse_cath_domain_list(path: Path) -> Dict[str, Dict]:
    """
    Parses the CATH domain list file (cath-domain-list-v4_4_0.txt).
    Returns a dictionary mapping domain ID to its metadata.
    """
    domains = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            domain_id = parts[0]
            # Superfamily ID is constructed from columns C, A, T, H (cols 2-5)
            superfamily_id = ".".join(parts[1:5])
            try:
                # The 11th column (index 10) is the domain length.
                length = int(parts[10])
                domains[domain_id] = {
                    "superfamily_id": superfamily_id,
                    "length": length,
                }
            except (ValueError, IndexError):
                logging.warning(f"Could not parse line, skipping: {' '.join(parts)}")
                continue
    return domains


def log_superfamily_stats(superfamilies: Dict[str, List[str]]):
    """Logs statistics about the superfamily distribution."""
    num_superfamilies = len(superfamilies)
    if num_superfamilies == 0:
        logging.info("No superfamilies to report stats on.")
        return

    sizes = [len(v) for v in superfamilies.values()]
    single_member_count = sum(1 for s in sizes if s == 1)

    logging.info("--- Superfamily Statistics ---")
    logging.info(f"Total superfamilies: {num_superfamilies}")
    logging.info(f"Superfamilies with a single member: {single_member_count}")
    logging.info(f"Superfamilies with >1 member: {num_superfamilies - single_member_count}")

    size_bins = {
        "2-5": 0,
        "6-10": 0,
        "11-50": 0,
        "51+": 0,
    }
    for s in sizes:
        if 2 <= s <= 5:
            size_bins["2-5"] += 1
        elif 6 <= s <= 10:
            size_bins["6-10"] += 1
        elif 11 <= s <= 50:
            size_bins["11-50"] += 1
        elif s > 50:
            size_bins["51+"] += 1

    logging.info("Distribution of multi-member superfamily sizes:")
    for k, v in size_bins.items():
        logging.info(f"  - Size {k}: {v}")
    logging.info("------------------------------")


def group_domains_by_superfamily(domains: Dict[str, Dict]) -> Dict[str, List[str]]:
    """
    Groups domain IDs by their superfamily ID.
    """
    superfamilies = defaultdict(list)
    for domain_id, meta in domains.items():
        superfamilies[meta["superfamily_id"]].append(domain_id)
    return superfamilies


# --- 2. Data Filtering ---


def filter_domains_by_length(domains: Dict[str, Dict], min_len: int, max_len: int) -> Dict[str, Dict]:
    """Filters domains based on sequence length."""
    return {domain_id: meta for domain_id, meta in domains.items() if min_len <= meta["length"] <= max_len}


# --- 3. Pair Generation ---


def generate_pairs(
    superfamilies: Dict[str, List[str]],
    num_pos: int,
    num_neg: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Generates positive and negative pairs for alignment."""
    positive_pairs = set()
    negative_pairs = set()

    # --- Generate Positive Pairs (same superfamily) ---
    superfamily_keys = [k for k, v in superfamilies.items() if len(v) > 1]
    if not superfamily_keys:
        logging.warning("No superfamilies with more than one member found after filtering. Cannot generate positive pairs.")
    else:
        pbar_pos = tqdm(total=num_pos, desc="Generating positive pairs")
        while len(positive_pairs) < num_pos:
            sf_key = random.choice(superfamily_keys)
            d1, d2 = random.sample(superfamilies[sf_key], 2)
            pair = tuple(sorted((d1, d2)))
            if pair not in positive_pairs:
                positive_pairs.add(pair)
                pbar_pos.update(1)
        pbar_pos.close()

    # --- Generate Negative Pairs (different superfamily) ---
    all_superfamily_keys = list(superfamilies.keys())
    pbar_neg = tqdm(total=num_neg, desc="Generating negative pairs")
    while len(negative_pairs) < num_neg:
        sf_key1, sf_key2 = random.sample(all_superfamily_keys, 2)
        if sf_key1 == sf_key2:
            continue

        d1 = random.choice(superfamilies[sf_key1])
        d2 = random.choice(superfamilies[sf_key2])
        pair = tuple(sorted((d1, d2)))

        # Ensure it's not accidentally a positive pair (though highly unlikely)
        if pair not in positive_pairs and pair not in negative_pairs:
            negative_pairs.add(pair)
            pbar_neg.update(1)
    pbar_neg.close()

    return list(positive_pairs), list(negative_pairs)


# --- 4. Ground Truth Alignment (TM-align) ---


def run_tmalign_for_pair(task: Tuple) -> Optional[Dict]:
    """Worker function to run TM-align on a single pair."""
    pair, label, domain_meta, args = task
    d1, d2 = pair
    meta1 = domain_meta[d1]
    meta2 = domain_meta[d2]

    pdb_dir = Path(args["pdb_root"])

    # CATH PDB files are named by domain ID without an extension in a flat directory.
    pdb1_path = pdb_dir / d1
    pdb2_path = pdb_dir / d2

    if not pdb1_path.exists() or not pdb2_path.exists():
        logging.warning(f"Missing PDB for pair {d1}-{d2}. Searched for {pdb1_path} and {pdb2_path}")
        return None

    try:
        cmd = [
            args["tmalign_bin"],
            str(pdb1_path),
            str(pdb2_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        tm_scores = parse_tmalign_output(result.stdout)

        # For negative samples, we might filter them
        if label == "negative" and args["neg_tm_cutoff"] > 0:
            tm_score_1 = tm_scores.get("tm_score_2", 0.0)  # TM-score is normalized by the 2nd protein length
            if tm_score_1 >= args["neg_tm_cutoff"]:
                return None  # Discard this 'easy' negative sample

        # Extract the alignment from the output
        lines = result.stdout.splitlines()
        alignment_lines = []
        try:
            # Find the start of the alignment block
            start_index = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('(":" denotes'):
                    start_index = i + 1
                    break

            if start_index != -1 and len(lines) >= start_index + 3:
                # The alignment is the three lines following the header
                seq1_aligned = lines[start_index].strip()
                match_string = lines[start_index + 1].strip()
                seq2_aligned = lines[start_index + 2].strip()
                alignment_lines = [seq1_aligned, match_string, seq2_aligned]

        except IndexError:
            pass  # Will be caught by the length check below

        if len(alignment_lines) < 3:
            logging.warning(f"Could not parse alignment from TM-align output for pair {d1}-{d2}. Skipping.")
            return None

        seq1_aligned, _, seq2_aligned = alignment_lines

        # Convert gapped alignment to ungapped sequences and mappings
        seq1, seq2, ref_pairs, pid = gapped_to_data(seq1_aligned, seq2_aligned)

        superfamily_id = meta1["superfamily_id"]
        group_id = superfamily_id if label == "positive" else f"negative_pair_{d1}_{d2}"
        pair_id = f"{group_id}:{d1}-{d2}"

        cath_labels = [f"superfamily:{superfamily_id}"] if label == "positive" else []

        meta = {
            "tm_score_1": tm_scores.get("tm_score_1"),
            "tm_score_2": tm_scores.get("tm_score_2"),
            "tm_score_avg": tm_scores.get("tm_score_avg"),
            "rmsd": tm_scores.get("rmsd"),
            "length1": meta1["length"],
            "length2": meta2["length"],
        }

        return {
            "pair_id": pair_id,
            "group_id": group_id,
            "set_name": "cath",
            "label": label,
            "seq1_id": d1,
            "seq2_id": d2,
            "seq1": seq1,
            "seq2": seq2,
            "ref_alignment": ref_pairs,
            "percent_identity": pid,
            "cath_labels": cath_labels,
            "meta": json.dumps(meta),
        }

    except subprocess.CalledProcessError as e:
        logging.error(f"TM-align failed for pair {d1}-{d2}: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred for pair {d1}-{d2}: {e}")
        return None


# --- 5. Main Orchestration ---


def main():
    """Main function to orchestrate the dataset generation pipeline."""
    ap = argparse.ArgumentParser(description="Build a CATH-based fine-tuning dataset for protein structure alignment.")

    # Input/Output paths
    ap.add_argument("--cath_domain_list", type=str, required=True, help="Path to the CATH domain list file (e.g., cath-domain-list-v4_4_0.txt).")
    ap.add_argument(
        "--domain_list", type=str, default=None, help="Path to a file containing a list of domain IDs to filter against (e.g., a non-redundant subset). If not provided, no pre-filtering is done."
    )
    ap.add_argument("--pdb_root", type=str, required=True, help="Path to the directory with CATH PDB files.")
    ap.add_argument("--output_dir", type=str, default="work/cath_dataset", help="Directory to save the output JSONL file and logs.")
    ap.add_argument("--tmalign_bin", type=str, default="TMalign", help="Path to the TMalign executable.")

    # Filtering and Sampling
    ap.add_argument("--min_len", type=int, default=30, help="Minimum domain length to include.")
    ap.add_argument("--max_len", type=int, default=1000, help="Maximum domain length to include.")
    ap.add_argument("--num_pos_pairs", type=int, default=10000, help="Number of positive pairs (same superfamily) to generate.")
    ap.add_argument("--num_neg_pairs", type=int, default=10000, help="Number of negative pairs (different superfamily) to generate.")
    ap.add_argument("--neg_tm_cutoff", type=float, default=0.2, help="TM-score cutoff for 'hard' negative samples. Pairs above this are discarded.")
    ap.add_argument("--neg_oversample_factor", type=float, default=1.5, help="Factor to oversample negative pairs to ensure enough pairs pass the TM-score cutoff.")

    # Execution
    ap.add_argument("--workers", type=int, default=mp.cpu_count(), help="Number of parallel workers for TM-align.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = ap.parse_args()
    args_dict = vars(args)

    # --- Setup ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "build_cath_dataset.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logging.info(f"Starting CATH dataset generation with args: {args_dict}")

    random.seed(args.seed)

    # --- 1. Parse CATH data ---
    logging.info(f"Parsing CATH domain list from {args.cath_domain_list}...")
    all_domains = parse_cath_domain_list(Path(args.cath_domain_list))
    logging.info(f"Found {len(all_domains)} total domains from domain list.")

    # --- 2. Filter domains ---
    # First, filter by available PDB files
    pdb_root = Path(args.pdb_root)
    logging.info(f"Scanning for available PDB files in {pdb_root}...")
    available_pdbs = {p.name for p in pdb_root.glob("*") if p.is_file()}
    logging.info(f"Found {len(available_pdbs)} available PDB files.")

    domains_with_pdb = {domain_id: meta for domain_id, meta in all_domains.items() if domain_id in available_pdbs}
    logging.info(f"{len(domains_with_pdb)} domains have a corresponding PDB file.")

    # Optional: Filter by a provided domain list (e.g., S40 non-redundant list)
    if args.domain_list:
        domain_list_path = Path(args.domain_list)
        if domain_list_path.exists():
            logging.info(f"Filtering domains based on list: {domain_list_path}")
            filter_ids = parse_domain_id_list(domain_list_path)
            logging.info(f"Found {len(filter_ids)} IDs in the domain filter list.")

            # Intersect the domains that have PDBs with the filter list
            pre_filtered_domains = {domain_id: meta for domain_id, meta in domains_with_pdb.items() if domain_id in filter_ids}

            logging.info(f"{len(pre_filtered_domains)} domains remaining after intersecting with filter list.")

            # For debugging, check if any domains in the result are NOT in the filter list
            unfiltered_leaks = {domain_id for domain_id in pre_filtered_domains if domain_id not in filter_ids}
            if unfiltered_leaks:
                logging.error(f"Logic error: {len(unfiltered_leaks)} domains leaked through the filter. Example: {list(unfiltered_leaks)[:5]}")

        else:
            logging.error(f"Domain list file not found at {domain_list_path}. Exiting.")
            return
    else:
        pre_filtered_domains = domains_with_pdb

    logging.info(f"Filtering domains by length (min: {args.min_len}, max: {args.max_len})...")
    filtered_domains = filter_domains_by_length(pre_filtered_domains, args.min_len, args.max_len)
    logging.info(f"{len(filtered_domains)} domains remaining after length filtering.")

    if not filtered_domains:
        logging.error("No domains remaining after filtering. Cannot proceed. Please check your filtering criteria.")
        return

    superfamilies = group_domains_by_superfamily(filtered_domains)
    log_superfamily_stats(superfamilies)

    # --- 3. Generate pairs ---
    num_neg_to_generate = int(args.num_neg_pairs * args.neg_oversample_factor)
    logging.info(f"Generating {args.num_pos_pairs} positive pairs.")
    logging.info(f"Generating {num_neg_to_generate} negative pairs (oversampling by {args.neg_oversample_factor}x to account for TM-score filtering)...")

    positive_pairs, negative_pairs = generate_pairs(superfamilies, args.num_pos_pairs, num_neg_to_generate)
    logging.info(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} candidate negative pairs.")

    # --- 4. Run TM-align in parallel ---
    pos_tasks = [(p, "positive", filtered_domains, args_dict) for p in positive_pairs]
    neg_tasks = [(p, "negative", filtered_domains, args_dict) for p in negative_pairs]
    tasks = pos_tasks + neg_tasks

    logging.info(f"Running TM-align on {len(tasks)} total pairs using {args.workers} workers...")

    pos_results = []
    neg_results = []
    with mp.Pool(processes=args.workers) as pool:
        with tqdm(total=len(tasks), desc="Running TM-align") as pbar:
            for result in pool.imap_unordered(run_tmalign_for_pair, tasks):
                if result:
                    if result["label"] == "positive":
                        pos_results.append(result)
                    else:
                        neg_results.append(result)
                pbar.update(1)

    logging.info(f"Successfully processed {len(pos_results)} positive pairs and {len(neg_results)} negative pairs.")

    # Trim negative pairs to the desired number
    if len(neg_results) < args.num_neg_pairs:
        logging.warning(f"Generated only {len(neg_results)} negative pairs after filtering, which is less than the target of {args.num_neg_pairs}. Consider increasing --neg_oversample_factor.")
    else:
        random.shuffle(neg_results)
        neg_results = neg_results[: args.num_neg_pairs]
        logging.info(f"Trimmed negative pairs to the desired {len(neg_results)}.")

    results = pos_results + neg_results
    random.shuffle(results)  # Shuffle the final dataset

    # --- 5. Save results and Log Final Statistics ---
    output_jsonl_path = output_dir / "dataset.jsonl"
    logging.info(f"Saving final dataset to {output_jsonl_path}...")
    with open(output_jsonl_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Final statistics
    final_pos_count = len(pos_results)
    final_neg_count = len(neg_results)

    pos_superfamilies = set()
    for r in results:
        if r["label"] == "positive":
            d1_id = r["seq1_id"]
            if d1_id in filtered_domains:
                pos_superfamilies.add(filtered_domains[d1_id]["superfamily_id"])

    logging.info("--- Final Dataset Statistics ---")
    logging.info(f"Total pairs in final dataset: {len(results)}")
    logging.info(f"  - Positive pairs: {final_pos_count}")
    logging.info(f"  - Negative pairs: {final_neg_count}")
    logging.info(f"Unique superfamilies represented in positive pairs: {len(pos_superfamilies)}")
    logging.info("---------------------------------")

    # Save superfamily composition
    superfamily_composition = defaultdict(list)
    included_domains = {r["seq1_id"] for r in results} | {r["seq2_id"] for r in results}
    for domain_id in included_domains:
        if domain_id in filtered_domains:
            sf_id = filtered_domains[domain_id]["superfamily_id"]
            superfamily_composition[sf_id].append(domain_id)

    output_sf_comp_path = output_dir / "superfamily_composition.json"
    logging.info(f"Saving superfamily composition to {output_sf_comp_path}...")
    with open(output_sf_comp_path, "w") as f:
        json.dump(superfamily_composition, f, indent=2)

    logging.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
