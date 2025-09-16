from otalign.io.fasta_utils import wrap_fasta_sequence


def test_wrap_fasta_sequence_basic():
    s = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = wrap_fasta_sequence(s, width=5)
    lines = out.splitlines()
    assert lines == ["ABCDE", "FGHIJ", "KLMNO", "PQRST", "UVWXY", "Z"]
