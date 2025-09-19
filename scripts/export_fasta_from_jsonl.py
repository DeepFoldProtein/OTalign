import argparse
import json
from pathlib import Path
from typing import cast

from tqdm.auto import tqdm

from otalign.io.fasta_utils import reconstruct_alignment
from scripts.dataset_utils import iter_pairs_from_dataset


def load_sequences(dataset_path: str) -> dict[str, dict]:
    """
    Loads sequences from a dataset into a dictionary keyed by pair_id.
    """
    sequences = {}
    data_iterable = iter_pairs_from_dataset(dataset_path)

    for ex_raw in tqdm(data_iterable, desc="Loading sequences"):
        ex = cast(dict, ex_raw)
        pair_id = ex.get("pair_id", f"{ex['seq1_id']}-{ex['seq2_id']}")
        sequences[pair_id] = {
            "seq1": ex["seq1"],
            "seq2": ex["seq2"],
            "seq1_id": ex["seq1_id"],
            "seq2_id": ex["seq2_id"],
        }
    return sequences


def load_alignments(jsonl_path: str) -> dict[str, dict]:
    """
    Loads alignment results from a JSONL file into a dictionary keyed by pair_id.
    """
    alignments = {}
    print(f"Loading alignments from {jsonl_path}...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading alignments"):
            res = json.loads(line)
            pair_id = res["pair_id"]
            alignments[pair_id] = res
    return alignments


def main():
    parser = argparse.ArgumentParser(description="Generate FASTA files from OTAlign JSONL output and original dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the original dataset (JSONL file) or Hugging Face dataset identifier (e.g., 'user/dataset,config,split').",
    )
    parser.add_argument(
        "--alignments_jsonl",
        type=str,
        required=True,
        help="Path to the JSONL file containing alignment results from run_otalign_on_dataset.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output FASTA files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_sequences(args.dataset)
    alignments = load_alignments(args.alignments_jsonl)

    print("Generating FASTA files...")
    processed_count = 0
    for pair_id, alignment_result in tqdm(alignments.items(), desc="Generating FASTA"):
        if "error" in alignment_result:
            # print(f"Skipping {pair_id} due to error: {alignment_result['error']}")
            continue

        if pair_id not in sequences:
            # print(f"Warning: pair_id {pair_id} not found in the original dataset. Skipping.")
            continue

        seq_data = sequences[pair_id]
        seq1 = seq_data["seq1"]
        seq2 = seq_data["seq2"]
        seq1_id = seq_data["seq1_id"]
        seq2_id = seq_data["seq2_id"]

        pred_alignment = alignment_result.get("pred_alignment")
        if not pred_alignment:
            continue

        aligned_seq1, aligned_seq2 = reconstruct_alignment(seq1, seq2, pred_alignment)

        fasta_content = f">{seq1_id}\n{aligned_seq1}\n>{seq2_id}\n{aligned_seq2}\n"

        output_path = output_dir / f"{pair_id}.fasta"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(fasta_content)
        processed_count += 1

    print(f"\n[ok] {processed_count} FASTA files saved to {output_dir}")


if __name__ == "__main__":
    main()
