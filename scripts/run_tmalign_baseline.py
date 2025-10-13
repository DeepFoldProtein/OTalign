import argparse
import json
import multiprocessing as mp
import re
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Optional, Tuple

from tqdm.auto import tqdm

from otalign.io.fasta_utils import reconstruct_alignment
from scripts.dataset_utils import iter_pairs_from_dataset


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


def get_pdb_paths(seq1_id: str, seq2_id: str, pdb_dir: Path) -> Optional[Tuple[Path, Path]]:
    """Constructs paths to PDB files assuming PDBs are named {seq_id}.pdb in pdb_dir."""
    try:
        pdb1_path = pdb_dir / f"{seq1_id}.pdb"
        pdb2_path = pdb_dir / f"{seq2_id}.pdb"

        if not pdb1_path.exists() or not pdb2_path.exists():
            return None

        return pdb1_path, pdb2_path
    except Exception as e:
        print(f"Error constructing PDB paths for {seq1_id}/{seq2_id}: {e}")
        return None


def _worker(task):
    (ex, args) = task
    pair_id = ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}")

    try:
        pdb_paths = get_pdb_paths(ex["seq1_id"], ex["seq2_id"], Path(args["pdb_dir"]))
        if not pdb_paths:
            return {"pair_id": pair_id, "error": "missing_pdb"}
        pdb1_path, pdb2_path = pdb_paths

        # Run TMalign for baseline structure alignment
        cmd = [
            args["tmalign_bin"],
            str(pdb1_path),
            str(pdb2_path),
        ]

        use_ref_ali = args.get("use_ref_alignment")
        fasta_dir = args.get("fasta_dir")
        guided_mode = False

        with ExitStack() as stack:
            fasta_path_to_use = None
            if use_ref_ali:
                if "ref_alignment" in ex and ex["ref_alignment"]:
                    if "seq1" not in ex or "seq2" not in ex:
                        return {"pair_id": pair_id, "error": "missing sequences for reference alignment"}

                    aligned_seq1, aligned_seq2 = reconstruct_alignment(ex["seq1"], ex["seq2"], ex["ref_alignment"])

                    tmp_fasta = stack.enter_context(tempfile.NamedTemporaryFile(mode="w", suffix=".fasta"))
                    tmp_fasta.write(f">{ex['seq1_id']}\n{aligned_seq1}\n")
                    tmp_fasta.write(f">{ex['seq2_id']}\n{aligned_seq2}\n")
                    tmp_fasta.flush()
                    fasta_path_to_use = tmp_fasta.name
                    guided_mode = True
            elif fasta_dir:
                fasta_path = Path(fasta_dir) / f"{pair_id}.fasta"
                if not fasta_path.exists():
                    return {"pair_id": pair_id, "error": f"missing_fasta: {fasta_path}"}
                fasta_path_to_use = str(fasta_path)
                guided_mode = True

            if fasta_path_to_use:
                cmd.extend(["-I", fasta_path_to_use])

            if args.get("extra_args"):
                cmd.extend(args["extra_args"])

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        tm_scores = parse_tmalign_output(result.stdout)

        rec = {
            "pair_id": pair_id,
            "seq1_id": ex["seq1_id"],
            "seq2_id": ex["seq2_id"],
            "metrics": tm_scores,
            "meta": {"tmalign_mode": "guided" if guided_mode else "baseline"},
        }
        return rec

    except subprocess.CalledProcessError as e:
        return {"pair_id": pair_id, "error": f"tmalign_failed: {e.stderr}"}
    except Exception as e:
        return {"pair_id": pair_id, "error": str(e)}


def main():
    ap = argparse.ArgumentParser(description="Run TMalign on a dataset to get baseline structural alignment scores.")
    ap.add_argument("--dataset", type=str, required=True, help="Path to the dataset (JSONL or HF identifier).")
    ap.add_argument("--pdb_dir", type=str, required=True, help="Root directory for PDB files (e.g., data/MALIDUP).")
    ap.add_argument("--fasta_dir", type=str, default=None, help="Optional directory containing *.fasta alignment files to guide TMalign with -I.")
    ap.add_argument("--use_ref_alignment", action="store_true", help="Use 'ref_alignment' from the dataset as input for TMalign.")
    ap.add_argument("--tmalign_bin", type=str, default="TMalign", help="Path to the TMalign executable.")
    ap.add_argument("--extra_args", nargs="*", default=[], help="Additional flags for TMalign.")
    ap.add_argument("--output", type=str, required=True, help="Path to the output JSONL file.")
    ap.add_argument("--workers", type=int, default=4, help="Number of worker processes.")
    args = ap.parse_args()

    pair_iterator = iter_pairs_from_dataset(args.dataset)

    args_dict = vars(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = ((ex, args_dict) for ex in pair_iterator)

    with out_path.open("w", encoding="utf-8") as fout:
        with mp.Pool(processes=args.workers) as pool:
            # Use tqdm to show progress
            results_iterator = pool.imap_unordered(_worker, tasks, chunksize=4)
            for rec in tqdm(results_iterator, desc="Running TMalign baseline"):
                if rec:
                    fout.write(json.dumps(rec) + "\n")

    print(f"[ok] wrote TMalign baseline scores -> {out_path}")


if __name__ == "__main__":
    main()
