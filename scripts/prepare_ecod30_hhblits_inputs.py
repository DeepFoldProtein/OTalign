"""
Split the ECOD30 hard-benchmark CSV into per-domain FASTA files and write a
file list that the existing slurm_hhblits_hhmake_array.sh script can consume.

Produces:
  data/hhsuite/ecod30_hard/fasta/<id>.fasta   (one sequence per file)
  data/hhsuite/ecod30_hard/fasta.list         (absolute paths, one per line)

The slurm array script then writes:
  data/hhsuite/ecod30_hard/a3m/<id>.a3m
  data/hhsuite/ecod30_hard/hhm/<id>.hhm
"""

import argparse
import csv
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Tiny dotenv loader — does NOT override variables already in os.environ."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        os.environ.setdefault(k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="data/ecod30_hard/hard_benchmark.csv",
        help="Path to hard_benchmark.csv",
    )
    ap.add_argument(
        "--out_root",
        default="data/hhsuite/ecod30_hard",
        help="Output root directory",
    )
    args = ap.parse_args()

    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    out_root = Path(args.out_root)
    fasta_dir = out_root / "fasta"
    a3m_dir = out_root / "a3m"
    hhm_dir = out_root / "hhm"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    a3m_dir.mkdir(parents=True, exist_ok=True)
    hhm_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    print(f"Loaded {len(rows)} domains from {args.csv}")

    paths = []
    for r in rows:
        fp = fasta_dir / f"{r['id']}.fasta"
        if not fp.exists():
            fp.write_text(f">{r['id']}\n{r['sequence']}\n")
        paths.append(str(fp.resolve()))

    list_path = out_root / "fasta.list"
    list_path.write_text("\n".join(paths) + "\n")
    print(f"Wrote {len(paths)} FASTA files under {fasta_dir}")
    print(f"Wrote file list to {list_path}")
    print(f"Array size for slurm: 0-{len(paths) - 1}")
    print()
    hhdb = os.environ.get("HHDB", "<HHDB from .env>")
    sif = os.environ.get("HHSUITE_SIF", "<HHSUITE_SIF from .env>")
    print("Next, submit hhblits+hhmake via the chunked wrapper (reads .env):")
    print(f"  bash scripts/submit_ecod30_hhblits.sh   # uses HHDB={hhdb}, HHSUITE_SIF={sif}")


if __name__ == "__main__":
    main()
