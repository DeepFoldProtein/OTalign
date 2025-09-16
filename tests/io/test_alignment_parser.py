from otalign.io.alignment_parser import (
    convert_a3m_text_to_a2m,
    parse_clustal_alignment_text,
    parse_stockholm_alignment_text,
)


def test_convert_a3m_text_to_a2m():
    # All sequences must have the same number of match columns (uppercase or '-')
    # In this case, 4 match columns: (A,C,-,E), (D,C,-,F), (-,C,-,E)
    a3m_text = """
>seq1
AC-E
>seq2
aDC-F
>seq3
-bC-gE
"""
    expected = [
        {"id": "seq1", "seq": ".A.C-.E"},
        {"id": "seq2", "seq": "aD.C-.F"},
        {"id": "seq3", "seq": ".-bC-gE"},
    ]
    result = convert_a3m_text_to_a2m(a3m_text)
    assert result == expected


def test_convert_a3m_to_a2m_empty():
    assert convert_a3m_text_to_a2m("") == []


def test_parse_stockholm_alignment_text():
    sto_text = """
# STOCKHOLM 1.0
seq1    AC-DE
seq2    A-CDE
#=GC SS_cons ...
//
"""
    expected = [
        {"id": "seq1", "seq": "AC-DE"},
        {"id": "seq2", "seq": "A-CDE"},
    ]
    result = parse_stockholm_alignment_text(sto_text)
    assert result == expected


def test_parse_stockholm_alignment_text_multiblock():
    sto_text = """
# STOCKHOLM 1.0
seq1    AC-
seq2    A-C

seq1    DE
seq2    DE
//
"""
    expected = [
        {"id": "seq1", "seq": "AC-DE"},
        {"id": "seq2", "seq": "A-CDE"},
    ]
    result = parse_stockholm_alignment_text(sto_text)
    assert result == expected


def test_parse_clustal_alignment_text():
    clu_text = """
CLUSTAL W (1.82) multiple sequence alignment

seq1      AC-DE
seq2      A-CDE
          * ***

"""
    expected = [
        {"id": "seq1", "seq": "AC-DE"},
        {"id": "seq2", "seq": "A-CDE"},
    ]
    result = parse_clustal_alignment_text(clu_text)
    assert result == expected


def test_parse_clustal_alignment_text_multiblock():
    clu_text = """
CLUSTAL

seq1      AC-
seq2      A-C

seq1      DE
seq2      DE

"""
    expected = [
        {"id": "seq1", "seq": "AC-DE"},
        {"id": "seq2", "seq": "A-CDE"},
    ]
    result = parse_clustal_alignment_text(clu_text)
    assert result == expected
