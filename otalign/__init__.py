"""
OTalign: Optimal Transport for Protein Sequence Alignment

This package provides tools for protein sequence alignment using optimal transport.
"""

from otalign.align.fasta_aligner import FastaAligner, align_fasta_file
from otalign.io.fasta_utils import read_fasta_sequences, write_a3m_msa


__version__ = "0.1.0"

__all__ = [
    "FastaAligner",
    "align_fasta_file",
    "read_fasta_sequences",
    "write_a3m_msa",
]
