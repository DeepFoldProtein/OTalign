import json
import subprocess
import sys
from pathlib import Path


def _run_cli(tmp_path: Path, fmt: str, body: str):
    inp = tmp_path / f"in.{fmt}"
    out = tmp_path / "out.jsonl"
    inp.write_text(body, encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "otalign.io.cli_convert",
        "--input",
        str(inp),
        "--output_jsonl",
        str(out),
    ]
    if fmt != "fasta":
        cmd += ["--format", fmt]
    subprocess.check_call(cmd)
    txt = out.read_text(encoding="utf-8").strip()
    rec = json.loads(txt)
    assert "ref_alignment" in rec and isinstance(rec["ref_alignment"], list)
    return rec


def test_cli_a3m(tmp_path):
    a3m = """>Q
bACDEFG--
>T
ACD-FGHI
"""
    rec = _run_cli(tmp_path, "a3m", a3m)
    assert rec["seq1_id"] == "Q" and rec["seq2_id"] == "T"
    assert rec["seq1"] == "BACDEFG"
    assert rec["seq2"] == "ACDFGHI"


def test_cli_fasta(tmp_path):
    fasta = """>A
K-TAA--GG
>B
KET-AAQGG
"""
    rec = _run_cli(tmp_path, "fasta", fasta)
    assert rec["seq1_id"] == "A" and rec["seq2_id"] == "B"
    assert rec["seq1"] == "KT AAGG".replace(" ", "")
    assert rec["seq2"] == "KETAAQGG"
