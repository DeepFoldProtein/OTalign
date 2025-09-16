from pathlib import Path

from .alignment_parser import (
    parse_a3m_alignment_text,
    parse_clustal_alignment_text,
    parse_fasta_alignment_text,
    parse_stockholm_alignment_text,
)


def read_text(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="ignore")


def read_fasta_alignment(path: str | Path):
    text = read_text(path)
    return parse_fasta_alignment_text(text)


def read_a3m_alignment(path: str | Path):
    text = read_text(path)
    return parse_a3m_alignment_text(text)


def read_stockholm_alignment(path: str | Path):
    text = read_text(path)
    return parse_stockholm_alignment_text(text)


def read_clustal_alignment(path: str | Path):
    text = read_text(path)
    return parse_clustal_alignment_text(text)
