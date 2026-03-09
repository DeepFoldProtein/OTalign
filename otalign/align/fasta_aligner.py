"""
FASTA-based multiple sequence alignment using OTalign.

This module provides functionality to align multiple sequences from a FASTA file,
using the first sequence as a query and aligning all other sequences to it.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from otalign.align.uot_alignment import hard_alignment_from_transport, uot_alignment_metrics_with_sinkhorn
from otalign.io.fasta_utils import read_fasta_sequences, write_a3m_msa
from otalign.models.embedding import get_embeddings_for_sequences


def logistic_filter_score(
    transport_cost: float, sinkhorn_divergence: float, w_transport: float = -23.150216560289554, w_sinkhorn: float = -25.691965923484595, bias: float = 34.804294222280376
) -> float:
    """
    Compute a logistic filtering score based on alignment quality metrics.

    Args:
        transport_cost: Transport cost from alignment
        sinkhorn_divergence: Sinkhorn divergence value
        w_transport: Weight for transport cost
        w_sinkhorn: Weight for sinkhorn divergence
        bias: Bias term from metrics
    Returns:
        Probability score between 0 and 1 (higher = better quality)
    """
    # Linear combination of metrics
    linear_score = w_transport * transport_cost + w_sinkhorn * sinkhorn_divergence + bias

    # Apply sigmoid to get probability
    return 1.0 / (1.0 + np.exp(-linear_score))


class FastaAligner:
    """
    Aligner for multiple sequences from FASTA files using OTalign.
    """

    def __init__(
        self,
        model_name: str = "ESM2_6_8M",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 4,
        dtype: str = "fp32",
        epsilon: float = 0.1,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        num_iter: int = 1000,
        filter_threshold: float = 0.5,
        enable_filtering: bool = True,
        # Logistic filtering parameters (calibrated on example values)
        filter_w_transport: float = -23.150216560289554,
        filter_w_sinkhorn: float = -25.691965923484595,
        filter_bias: float = 34.804294222280376,
    ):
        """
        Initialize the FASTA aligner.

        Args:
            model_name: Pre-trained language model to use for embeddings
            cache_dir: Directory for embedding cache
            device: Device to run computations on
            batch_size: Batch size for embedding generation
            dtype: Data type for embeddings ("fp16", "fp32", "bf16")
            epsilon: Sinkhorn regularization parameter
            lambda1: Unbalanced OT regularization for query
            lambda2: Unbalanced OT regularization for target
            num_iter: Maximum number of Sinkhorn iterations
            filter_threshold: Logistic filtering threshold (0-1, higher = more stringent)
            enable_filtering: Whether to enable logistic filtering
            filter_w_transport: Logistic weight for transport cost
            filter_w_sinkhorn: Logistic weight for sinkhorn divergence
            filter_bias: Bias term from metrics
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self.dtype = dtype
        self.epsilon = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.num_iter = num_iter
        self.filter_threshold = filter_threshold
        self.enable_filtering = enable_filtering
        self.filter_w_transport = filter_w_transport
        self.filter_w_sinkhorn = filter_w_sinkhorn
        self.filter_bias = filter_bias

    def align_fasta(self, fasta_path: str | Path, output_path: str | Path, **kwargs) -> Dict[str, Any]:
        """
        Align sequences from a FASTA file using the first sequence as query.

        Args:
            fasta_path: Path to input FASTA file
            output_path: Path to output A3M file
            **kwargs: Additional arguments for alignment

        Returns:
            Dictionary containing alignment results and statistics
        """
        # Read sequences from FASTA file
        sequences = read_fasta_sequences(fasta_path)

        if len(sequences) < 2:
            raise ValueError("FASTA file must contain at least 2 sequences")

        # First sequence is the query
        query_id, query_seq = sequences[0]
        targets = sequences[1:]

        print(f"Query: {query_id} (length: {len(query_seq)})")
        print(f"Targets: {len(targets)} sequences")

        # Get embeddings for all sequences
        all_sequences = [query_seq] + [seq for _, seq in targets]
        all_ids = [query_id] + [seq_id for seq_id, _ in targets]

        print("Generating embeddings...")
        embeddings = get_embeddings_for_sequences(
            sequences=all_sequences,
            seq_ids=all_ids,
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            device=self.device,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )

        query_embedding = embeddings[0]
        target_embeddings = embeddings[1:]

        # Align each target to the query
        target_alignments = []
        accepted_target_alignments = []  # sequences that pass filter
        filtered_target_alignments = []  # sequences that don't pass filter
        alignment_results = []

        print("Performing alignments...")
        for i, ((target_id, target_seq), target_embedding) in enumerate(zip(targets, target_embeddings)):
            print(f"  Aligning {target_id} ({i + 1}/{len(targets)})")

            # Use SinkhornUOT class like run_otalign_on_dataset.py for consistency
            from otalign.align.sinkhorn import SinkhornUOT

            # Add batch dimension and create masks (following run_otalign_on_dataset.py pattern)
            query_emb_batch = query_embedding.unsqueeze(0)  # [1, M, D]
            target_emb_batch = target_embedding.unsqueeze(0)  # [1, N, D]

            m, n = query_embedding.shape[0], target_embedding.shape[0]
            query_mask = torch.ones(1, m, dtype=torch.bool, device=query_embedding.device)
            target_mask = torch.ones(1, n, dtype=torch.bool, device=target_embedding.device)

            # Initialize aligner exactly like run_otalign_on_dataset.py
            aligner = SinkhornUOT()

            # Perform alignments (AB, AA, BB) exactly like run_otalign_on_dataset.py
            res_ab = aligner(query_emb_batch, target_emb_batch, query_mask, target_mask, self.num_iter, self.epsilon, self.lambda1, self.lambda2)
            res_aa = aligner(query_emb_batch, query_emb_batch, query_mask, query_mask, self.num_iter, self.epsilon, self.lambda1, self.lambda1)
            res_bb = aligner(target_emb_batch, target_emb_batch, target_mask, target_mask, self.num_iter, self.epsilon, self.lambda2, self.lambda2)

            # Extract individual results (remove batch dimension for compatibility)
            result = {
                "plan": res_ab["transport_plan"][0],  # [M, N]
                "scaling_u": res_ab["scaling_u"][0],  # [M] - keep original name for compatibility
                "scaling_v": res_ab["scaling_v"][0],  # [N] - keep original name for compatibility
                "cost": (res_ab["transport_plan"][0] * res_ab["cost_matrix"][0]).sum(),
            }

            # Calculate metrics with proper Sinkhorn divergence (exactly like run_otalign_on_dataset.py)
            metrics = uot_alignment_metrics_with_sinkhorn(
                a=res_ab["mu"],  # [1, M]
                b=res_ab["nu"],  # [1, N]
                cost_matrix=res_ab["cost_matrix"],  # [1, M, N]
                transport_plan=res_ab["transport_plan"],  # [1, M, N]
                mask_a=query_mask,  # [1, M]
                mask_b=target_mask,  # [1, N]
                plan_xx=res_aa["transport_plan"],  # [1, M, M]
                cost_xx=res_aa["cost_matrix"],  # [1, M, M]
                plan_yy=res_bb["transport_plan"],  # [1, N, N]
                cost_yy=res_bb["cost_matrix"],  # [1, N, N]
                reg=self.epsilon,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
            )

            # Extract scalar values from batch tensors
            transport_cost = float(metrics["transport_cost"][0])
            sinkhorn_divergence = float(metrics["sinkhorn_divergence"][0])

            # Calculate filtering score using configured weights
            filter_score = logistic_filter_score(
                transport_cost=transport_cost,
                sinkhorn_divergence=sinkhorn_divergence,
                w_transport=self.filter_w_transport,
                w_sinkhorn=self.filter_w_sinkhorn,
                bias=self.filter_bias,
            )

            # Handle NaN values in filter_score as a safety measure
            if not torch.isfinite(torch.tensor(filter_score)):
                print(f"Warning: Invalid filter_score for {target_id}, setting to 0.0")
                filter_score = 0.0

            # Extract alignment path exactly like run_otalign_on_dataset.py
            P_np = res_ab["transport_plan"][0].cpu().numpy()  # [M, N]
            f_np = res_ab["scaling_u"][0].log().cpu().numpy()  # [M]
            g_np = res_ab["scaling_v"][0].log().cpu().numpy()  # [N]

            hard_aln = hard_alignment_from_transport(
                P=P_np,
                f=f_np,
                g=g_np,
                mode="global",  # Using global alignment mode (same as run_otalign_on_dataset.py default)
            )

            # Extract only the match pairs from the alignment path (same as run_otalign_on_dataset.py)
            alignment_pairs = [[r - 1, c - 1] for r, c, op in hard_aln["path"] if op == "M"]

            # Always add to full alignments
            target_alignments.append((target_id, target_seq, alignment_pairs))

            # Separate sequences based on filter result
            print(f"Filter score: {filter_score}, Filter threshold: {self.filter_threshold}")
            passes_filter = filter_score >= self.filter_threshold if self.enable_filtering else True
            if passes_filter:
                accepted_target_alignments.append((target_id, target_seq, alignment_pairs))
            else:
                filtered_target_alignments.append((target_id, target_seq, alignment_pairs))

            alignment_results.append(
                {
                    "target_id": target_id,
                    "cost": float(result["cost"]),
                    "transport_cost": transport_cost,
                    "sinkhorn_divergence": sinkhorn_divergence,
                    "filter_score": filter_score,
                    "passes_filter": passes_filter,
                    "alignment_score": float(hard_aln["score"]),
                    "num_pairs": len(alignment_pairs),
                    "coverage_query": len(alignment_pairs) / len(query_seq),
                    "coverage_target": len(alignment_pairs) / len(target_seq),
                }
            )

        # Prepare filter parameters for metadata
        filter_params = {
            "w_transport": self.filter_w_transport,
            "w_sinkhorn": self.filter_w_sinkhorn,
            "bias": self.filter_bias,
            "filter_threshold": self.filter_threshold if self.enable_filtering else None,
        }

        # Write A3M outputs
        print(f"Writing full A3M output to {output_path}")
        write_a3m_msa(
            output_path=output_path,
            query_id=query_id,
            query_seq=query_seq,
            target_alignments=target_alignments,
            alignment_metadata=alignment_results,
            filter_params=filter_params,
        )

        # Write accepted and filtered A3M outputs if filtering is enabled
        accepted_output_path = None
        filtered_output_path = None

        if self.enable_filtering:
            output_path_obj = Path(output_path)
            base_dir = output_path_obj.parent
            base_name = output_path_obj.stem
            extension = output_path_obj.suffix

            # Write accepted sequences (those that pass filter)
            if accepted_target_alignments:
                accepted_output_path = base_dir / f"{base_name}_accepted{extension}"
                accepted_metadata = [metadata for metadata in alignment_results if metadata["passes_filter"]]

                print(f"Writing accepted A3M output to {accepted_output_path}")
                write_a3m_msa(
                    output_path=accepted_output_path,
                    query_id=query_id,
                    query_seq=query_seq,
                    target_alignments=accepted_target_alignments,
                    alignment_metadata=accepted_metadata,
                    filter_params=filter_params,
                )
            else:
                print("Warning: No alignments passed the filtering threshold")

            # Write filtered sequences (those that don't pass filter)
            if filtered_target_alignments:
                filtered_output_path = base_dir / f"{base_name}_filtered{extension}"
                filtered_metadata = [metadata for metadata in alignment_results if not metadata["passes_filter"]]

                print(f"Writing filtered (rejected) A3M output to {filtered_output_path}")
                write_a3m_msa(
                    output_path=filtered_output_path,
                    query_id=query_id,
                    query_seq=query_seq,
                    target_alignments=filtered_target_alignments,
                    alignment_metadata=filtered_metadata,
                    filter_params=filter_params,
                )
            else:
                print("Warning: All alignments passed the filtering threshold")

        return {
            "query_id": query_id,
            "query_length": len(query_seq),
            "num_targets": len(targets),
            "num_accepted_targets": len(accepted_target_alignments),
            "num_filtered_targets": len(filtered_target_alignments),
            "filter_threshold": self.filter_threshold if self.enable_filtering else None,
            "alignments": alignment_results,
            "output_file": str(output_path),
            "accepted_output_file": str(accepted_output_path) if accepted_output_path else None,
            "filtered_output_file": str(filtered_output_path) if filtered_output_path else None,
        }


def align_fasta_file(fasta_path: str | Path, output_path: str | Path, model_name: str = "ESM2_6_8M", cache_dir: Optional[str] = None, device: str = "cpu", **kwargs) -> Dict[str, Any]:
    """
    Convenience function to align sequences from a FASTA file.

    Args:
        fasta_path: Path to input FASTA file
        output_path: Path to output A3M file
        model_name: Pre-trained language model to use
        cache_dir: Directory for embedding cache
        device: Device to run computations on
        **kwargs: Additional arguments for FastaAligner and alignment

    Returns:
        Dictionary containing alignment results
    """
    # Separate FastaAligner kwargs from alignment kwargs
    aligner_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        in ["batch_size", "dtype", "epsilon", "lambda1", "lambda2", "num_iter", "filter_threshold", "enable_filtering", "filter_w_transport", "filter_w_sinkhorn", "filter_w_bias", "filter_intercept"]
    }

    alignment_kwargs = {k: v for k, v in kwargs.items() if k in ["num_iter", "tol", "damp", "gauge", "stabilize"]}

    aligner = FastaAligner(model_name=model_name, cache_dir=cache_dir, device=device, **aligner_kwargs)

    return aligner.align_fasta(fasta_path, output_path, **alignment_kwargs)
