"""
Script to align sequences from a FASTA file using OTalign.

This script reads a FASTA file, uses the first sequence as a query,
and aligns all other sequences to it using OTalign. The output is
written in A3M format with proper gap and deletion handling.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from otalign.align.fasta_aligner import align_fasta_file


def main():
    parser = argparse.ArgumentParser(description="Align sequences from FASTA file using OTalign", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input/Output arguments
    parser.add_argument("fasta_file", type=str, help="Path to input FASTA file (first sequence will be used as query)")
    parser.add_argument("output_file", type=str, help="Path to output A3M file")

    # Model arguments
    parser.add_argument("--model", type=str, default="ESM2_6_8M", help="Pre-trained language model to use for embeddings")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for embedding cache")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for embedding generation")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "fp32", "bf16"], help="Data type for embeddings")

    # Alignment parameters
    parser.add_argument("--epsilon", type=float, default=0.1, help="Sinkhorn regularization parameter")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Unbalanced OT regularization for query sequence")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Unbalanced OT regularization for target sequences")
    parser.add_argument("--num-iter", type=int, default=1000, help="Maximum number of Sinkhorn iterations")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance for Sinkhorn algorithm")

    # Filtering options
    parser.add_argument("--enable-filtering", action="store_true", default=False, help="Enable logistic filtering of alignments")
    parser.add_argument("--filter-threshold", type=float, default=0.5, help="Logistic filtering threshold (0-1, higher = more stringent)")

    # Output options
    parser.add_argument("--results-json", type=str, default=None, help="Path to save alignment statistics as JSON")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress information")

    args = parser.parse_args()

    # Validate input file
    fasta_path = Path(args.fasta_file)
    if not fasta_path.exists():
        print(f"Error: FASTA file not found: {args.fasta_file}", file=sys.stderr)
        sys.exit(1)

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.verbose:
        print(f"Using device: {device}")
        print(f"Model: {args.model}")
        print(f"Input: {args.fasta_file}")
        print(f"Output: {args.output_file}")
        if args.cache_dir:
            print(f"Cache directory: {args.cache_dir}")
        print()

    try:
        # Perform alignment
        results = align_fasta_file(
            fasta_path=args.fasta_file,
            output_path=args.output_file,
            model_name=args.model,
            cache_dir=args.cache_dir,
            device=device,
            batch_size=args.batch_size,
            dtype=args.dtype,
            epsilon=args.epsilon,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            num_iter=args.num_iter,
            tol=args.tol,
            enable_filtering=args.enable_filtering,
            filter_threshold=args.filter_threshold,
        )

        # Print summary
        print("\nAlignment completed successfully!")
        print(f"Query: {results['query_id']} (length: {results['query_length']})")
        print(f"Targets: {results['num_targets']} sequences")
        print(f"Output written to: {results['output_file']}")

        # Print filtering information if enabled
        if args.enable_filtering:
            print(f"Filtering enabled (threshold: {args.filter_threshold})")
            print(f"Accepted targets: {results['num_accepted_targets']}/{results['num_targets']}")
            print(f"Filtered (rejected) targets: {results['num_filtered_targets']}/{results['num_targets']}")
            if results["accepted_output_file"]:
                print(f"Accepted sequences written to: {results['accepted_output_file']}")
            if results["filtered_output_file"]:
                print(f"Filtered (rejected) sequences written to: {results['filtered_output_file']}")

        if args.verbose and results["alignments"]:
            print("\nAlignment statistics:")
            for align_result in results["alignments"]:
                print(f"  {align_result['target_id']}:")
                print(f"    Cost: {align_result['cost']:.4f}")
                print(f"    Aligned pairs: {align_result['num_pairs']}")
                print(f"    Query coverage: {align_result['coverage_query']:.2%}")
                print(f"    Target coverage: {align_result['coverage_target']:.2%}")

        # Save detailed results if requested
        if args.results_json:
            with open(args.results_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Detailed results saved to: {args.results_json}")

    except Exception as e:
        print(f"Error during alignment: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
