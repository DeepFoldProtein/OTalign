"""
Build pLM-BLAST embedding database from ECOD dataset.

This script generates per-residue embeddings for ECOD database sequences
using ProtT5 or other PLMs, compatible with pLM-BLAST search format.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Add plmblast to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLMBLAST_ROOT = PROJECT_ROOT / "third_party" / "plmblast"
if PLMBLAST_ROOT.exists():
    sys.path.insert(0, str(PLMBLAST_ROOT))
else:
    raise ImportError(f"pLM-BLAST not found at {PLMBLAST_ROOT}")


def build_database_embeddings(input_csv: Path, output_dir: Path, model_name: str = "ProtT5_XL_UniRef50", device: str = "cuda", batch_size: int = 0, use_cache: bool = True):
    """
    Build pLM-BLAST database from ECOD sequences.

    Args:
        input_csv: Path to ECOD CSV (with id, sequence columns)
        output_dir: Output directory for embeddings
        model_name: PLM model name
        device: Device to use (cuda/cpu)
        batch_size: Batch size (0 for adaptive)
        use_cache: Whether to use cached embeddings
    """
    from otalign.models import get_plm_model

    # Load data
    logging.info(f"Loading sequences from {input_csv}")
    df = pd.read_csv(input_csv)

    if "id" not in df.columns or "sequence" not in df.columns:
        raise ValueError("CSV must contain 'id' and 'sequence' columns")

    logging.info(f"Loaded {len(df)} sequences")

    # Setup model
    logging.info(f"Loading PLM model: {model_name}")
    plm = get_plm_model(model_name, device=device)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process sequences
    logging.info("Computing embeddings...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding sequences"):
        seq_id = row["id"]
        sequence = row["sequence"]

        # Output path for this embedding
        emb_file = output_dir / f"{seq_id}.emb.64"

        # Skip if already exists
        if use_cache and emb_file.exists():
            continue

        try:
            # Get embeddings [L, D]
            with torch.no_grad():
                embeddings = plm.embed([sequence])[0]  # [L, D]

            # Save as float32 (pLM-BLAST format)
            embeddings_np = embeddings.cpu().numpy().astype("float32")

            # Save in pLM-BLAST format (binary)
            with open(emb_file, "wb") as f:
                embeddings_np.tofile(f)

        except Exception as e:
            logging.error(f"Failed to process {seq_id}: {e}")
            continue

    # Create index file (CSV format expected by pLM-BLAST)
    index_file = output_dir / "index.csv"
    df[["id", "sequence"]].to_csv(index_file, index=False)

    logging.info(f"Database build complete: {output_dir}")
    logging.info(f"  - Total sequences: {len(df)}")
    logging.info(f"  - Index file: {index_file}")


def main():
    parser = argparse.ArgumentParser(description="Build pLM-BLAST embedding database from ECOD dataset")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file (database.csv from prepare_ecod_dataset.py)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for pLM-BLAST database")
    parser.add_argument("--model", type=str, default="ProtT5_XL_UniRef50", help="PLM model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size (0 for adaptive)")
    parser.add_argument("--no_cache", action="store_true", help="Recompute all embeddings even if they exist")

    args = parser.parse_args()

    input_csv = Path(args.input)
    if not input_csv.exists():
        logging.error(f"Input file not found: {input_csv}")
        sys.exit(1)

    output_dir = Path(args.output)

    build_database_embeddings(input_csv=input_csv, output_dir=output_dir, model_name=args.model, device=args.device, batch_size=args.batch_size, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
