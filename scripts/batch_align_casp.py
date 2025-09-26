"""
Batch alignment script for CASP15 targets using OTalign.

This script processes the filtered_fasta_list.txt file and runs OTalign alignment
for each target by combining vdb_ankh and vdb_esm1b FASTA files.

For each target (e.g., T1154):
- Input: combines casp15_test1/T1154/msas/vdb_ankh_faiss/plm_vdb_raw.fasta
         and casp15_test1/T1154/msas/vdb_esm1b_faiss/plm_vdb_raw.fasta
- Output: casp15_test1/T1154/msas/vdb_ankh_esm1b_faiss_otalign/otalign.a3m
"""

import argparse
import logging
import os
import sys
from typing import List, Tuple

from otalign.align.fasta_aligner import FastaAligner


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    return logging.getLogger(__name__)


def parse_filtered_fasta_list(list_file: str) -> List[Tuple[int, str, str, str]]:
    """
    Parse the filtered_fasta_list.txt file.

    Returns:
        List of tuples: (sequence_length, target_id, fasta_name, fasta_path)
    """
    targets = []
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            parts = line.split()
            if len(parts) >= 4:
                seq_length = int(parts[0])
                target_id = parts[1]
                fasta_name = parts[2]
                fasta_path = parts[3]
                targets.append((seq_length, target_id, fasta_name, fasta_path))
    return targets


def run_otalign(aligner: FastaAligner, input_fasta: str, output_a3m: str, logger: logging.Logger, verbose: bool = False) -> bool:
    """
    Run OTalign alignment on the input FASTA file using FastaAligner directly.

    Args:
        aligner: Pre-initialized FastaAligner instance
        input_fasta: Path to input FASTA file
        output_a3m: Path to output A3M file
        logger: Logger instance
        verbose: Whether to log verbose output

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Running OTalign alignment: {input_fasta} -> {output_a3m}")

        # Run alignment
        results = aligner.align_fasta(input_fasta, output_a3m)

        # Log results
        if verbose:
            logger.info(f"Query: {results['query_id']} (length: {results['query_length']})")
            logger.info(f"Targets: {results['num_targets']} sequences")
            if aligner.enable_filtering:
                logger.info(f"Accepted targets: {results['num_accepted_targets']}/{results['num_targets']}")
                logger.info(f"Filtered targets: {results['num_filtered_targets']}/{results['num_targets']}")

        return True

    except Exception as e:
        logger.error(f"Error running OTalign: {e}")
        if verbose:
            import traceback

            logger.debug(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch align CASP15 targets using OTalign", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input arguments
    parser.add_argument("filtered_list", type=str, help="Path to filtered_fasta_list.txt file")
    parser.add_argument("--casp-base", type=str, default="/gpfs/deepfold/users/baehanjin/work/casp15", help="Base directory for CASP15 data")

    # OTalign arguments
    parser.add_argument("--model", type=str, default="ProtT5_XL_UniRef50", help="Pre-trained language model to use")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cpu, cuda:0, etc.)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for embedding generation")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for embedding cache")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "fp32", "bf16"], help="Data type for embeddings")

    # Alignment parameters
    parser.add_argument("--epsilon", type=float, default=0.1, help="Sinkhorn regularization parameter")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Unbalanced OT regularization for query sequence")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Unbalanced OT regularization for target sequences")
    parser.add_argument("--num-iter", type=int, default=1000, help="Maximum number of Sinkhorn iterations")
    # Filtering options
    parser.add_argument("--enable-filtering", action="store_true", default=False, help="Enable logistic filtering of alignments")
    parser.add_argument("--filter-threshold", type=float, default=0.5, help="Logistic filtering threshold (0-1, higher = more stringent)")

    # Processing options
    parser.add_argument("--start-from", type=str, default=None, help="Start processing from this target ID (for resuming)")
    parser.add_argument("--max-targets", type=int, default=None, help="Maximum number of targets to process")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="Skip targets where output already exists")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print what would be done without actually running")

    # Output options
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--log-file", type=str, default=None, help="Log file path (default: stderr)")

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.verbose)
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    # Validate input file
    if not os.path.exists(args.filtered_list):
        logger.error(f"Filtered list file not found: {args.filtered_list}")
        sys.exit(1)

    # Parse the filtered list
    logger.info(f"Parsing filtered list: {args.filtered_list}")
    targets = parse_filtered_fasta_list(args.filtered_list)
    logger.info(f"Found {len(targets)} targets to process")

    if not targets:
        logger.error("No targets found in filtered list")
        sys.exit(1)

    # Create a single FastaAligner instance to reuse across all targets
    logger.info("Initializing FastaAligner...")
    aligner = FastaAligner(
        model_name=args.model,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        dtype=args.dtype,
        epsilon=args.epsilon,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        num_iter=args.num_iter,
        filter_threshold=args.filter_threshold,
        enable_filtering=args.enable_filtering,
    )
    logger.info(f"FastaAligner initialized with model: {args.model}, device: {args.device}")

    # Process targets
    processed_count = 0
    success_count = 0
    skip_count = 0
    start_processing = args.start_from is None

    for seq_length, target_id, fasta_name, fasta_path in targets:
        # Check if we should start processing from this target
        if not start_processing:
            if target_id == args.start_from:
                start_processing = True
            else:
                continue

        # Check max targets limit
        if args.max_targets and processed_count >= args.max_targets:
            logger.info(f"Reached maximum target limit ({args.max_targets})")
            break

        processed_count += 1
        logger.info(f"Processing target {processed_count}/{len(targets)}: {target_id} (length: {seq_length})")

        # Build file paths
        target_dir = os.path.join(args.casp_base, "casp15_test1", target_id)
        input_fasta = os.path.join(target_dir, "msas", "vdb_ak_esm1b_faiss", "plm_vdb_raw.fasta")
        output_dir = os.path.join(target_dir, "msas", "vdb_ak_esm1b_faiss_otalign_updated")
        output_a3m = os.path.join(output_dir, "otalign.a3m")

        # Check if output already exists
        if args.skip_existing and os.path.exists(output_a3m):
            logger.info(f"Skipping {target_id} - output already exists: {output_a3m}")
            skip_count += 1
            continue

        # Check if input file exists
        if not os.path.exists(input_fasta):
            logger.warning(f"Input file does not exist for {target_id}: {input_fasta}")
            continue

        if args.dry_run:
            logger.info(f"[DRY RUN] Would process {target_id}:")
            logger.info(f"  Input: {input_fasta}")
            logger.info(f"  Output: {output_a3m}")
            continue

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Run OTalign
        logger.info(f"Running OTalign for {target_id}")
        if run_otalign(aligner, input_fasta, output_a3m, logger, args.verbose):
            logger.info(f"Successfully processed {target_id}")
            success_count += 1
        else:
            logger.error(f"OTalign failed for {target_id}")

    # Print summary
    logger.info("\nProcessing complete!")
    logger.info(f"Targets processed: {processed_count}")
    logger.info(f"Successful alignments: {success_count}")
    logger.info(f"Skipped (existing): {skip_count}")
    logger.info(f"Failed alignments: {processed_count - success_count}")


if __name__ == "__main__":
    main()
