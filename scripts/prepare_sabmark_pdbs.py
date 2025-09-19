import argparse
import glob
import json
import os
from pathlib import Path

from tqdm.auto import tqdm


def iter_jsonl(path: Path):
    """Yields JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description="Prepare MALIDUP PDBs by creating a flat directory of symlinks named by seq_id.")
    ap.add_argument(
        "--malidup_jsonl",
        type=str,
        required=True,
        help="Path to the MALIDUP JSONL file (e.g., from convert_malidup.py).",
    )
    ap.add_argument(
        "--pdb_root",
        type=str,
        required=True,
        help="Root directory of the original MALIDUP PDB files (e.g., data/MALIDUP).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the flattened symlinks (e.g., data/MALIDUP_flat).",
    )
    args = ap.parse_args()

    malidup_jsonl = Path(args.malidup_jsonl)
    pdb_root = Path(args.pdb_root)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating symlinks in {output_dir}...")

    created_count = 0
    skipped_count = 0

    for ex in tqdm(iter_jsonl(malidup_jsonl), desc="Processing pairs"):
        group_id = ex.get("group_id")
        pair_id = ex.get("pair_id")
        if not group_id or not pair_id:
            continue
        try:
            original_ali_path = Path(glob.glob(str(pdb_root / group_id / "*.manual.ali"))[0])
            aln1, aln2 = original_ali_path.read_text().upper().strip().splitlines()

            output_path = output_dir / f"{pair_id}.fasta"
            output_path.write_text(f">seq1\n{aln1}\n>seq2\n{aln2}")
        except Exception as e:
            print(f"An error occurred for seq_id '{group_id}': {e}")
            skipped_count += 1

        for seq_key in ["seq1_id", "seq2_id"]:
            seq_id = ex.get(seq_key)
            if not seq_id:
                continue

            try:
                # Expected seq_id format: 'group_id:domain_id'
                group_id, domain_id = seq_id.split(":")

                # Original PDB path: {pdb_root}/{group_id}/{domain_id}.pdb
                original_pdb_path = pdb_root / group_id / f"{domain_id}.pdb"

                # Symlink path: {output_dir}/{seq_id}.pdb
                symlink_path = output_dir / f"{seq_id}.pdb"

                if not original_pdb_path.exists():
                    print(f"Warning: Source PDB not found, skipping: {original_pdb_path}")
                    skipped_count += 1
                    continue

                # Create relative symlink for portability
                # os.symlink(src, dst)
                # We want the link at symlink_path to point to original_pdb_path
                # The source path needs to be relative to the symlink's location
                relative_original_path = os.path.relpath(original_pdb_path.resolve(), symlink_path.parent.resolve())

                if not symlink_path.exists():
                    os.symlink(relative_original_path, symlink_path)
                    created_count += 1
                else:
                    # If it exists but points to the wrong place, update it
                    if not symlink_path.is_symlink() or os.readlink(symlink_path) != relative_original_path:
                        os.remove(symlink_path)
                        os.symlink(relative_original_path, symlink_path)
                        created_count += 1

            except ValueError:
                print(f"Warning: Could not parse seq_id '{seq_id}', skipping.")
                skipped_count += 1
            except Exception as e:
                print(f"An error occurred for seq_id '{seq_id}': {e}")
                skipped_count += 1

    print(f"\n[ok] Done. Created {created_count} new symlinks. Skipped {skipped_count} entries.")
    print(f"Flat PDB directory ready at: {output_dir}")


if __name__ == "__main__":
    main()
