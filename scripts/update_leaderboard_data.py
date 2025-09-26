"""
Script to update the leaderboard data.ts file with evaluation results from CSV files.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


def read_csv_data(eval_dir: str) -> Dict[str, pd.DataFrame]:
    """Read all alignment metrics summary CSV files."""
    datasets = ["malidup", "malisam", "sabmark_sup", "sabmark_twi"]
    data = {}

    for dataset in datasets:
        csv_path = os.path.join(eval_dir, f"_{dataset}", "alignment_metrics_summary.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            data[dataset] = df
            print(f"Loaded {dataset}: {len(df)} rows")
        else:
            print(f"Warning: {csv_path} not found")

    return data


def extract_metrics(data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """Extract F1 and Recall metrics for each model and dataset."""
    results = {}

    for dataset, df in data.items():
        for _, row in df.iterrows():
            model = row["label"]
            metric = row["metric"]
            mean_value = row["mean"]
            if model not in results:
                results[model] = {}
            # Map dataset and metric to the expected field names
            if dataset == "malidup" and metric == "F1":
                results[model]["malidup_f1"] = round(mean_value, 4)
            elif dataset == "malisam" and metric == "F1":
                results[model]["malisam_f1"] = round(mean_value, 4)
            elif dataset == "sabmark_sup" and metric == "Recall":
                results[model]["sabmark_sup_recall"] = round(mean_value, 4)
            elif dataset == "sabmark_twi" and metric == "Recall":
                results[model]["sabmark_twi_recall"] = round(mean_value, 4)

    return results


def calculate_average(metrics: Dict[str, float]) -> Optional[float]:
    """Calculate average score from all four metrics: malidup_f1, malisam_f1, sabmark_sup_recall, sabmark_twi_recall."""
    scores = []

    if "malidup_f1" in metrics and metrics["malidup_f1"] is not None:
        scores.append(metrics["malidup_f1"])

    if "malisam_f1" in metrics and metrics["malisam_f1"] is not None:
        scores.append(metrics["malisam_f1"])

    if "sabmark_sup_recall" in metrics and metrics["sabmark_sup_recall"] is not None:
        scores.append(metrics["sabmark_sup_recall"])

    if "sabmark_twi_recall" in metrics and metrics["sabmark_twi_recall"] is not None:
        scores.append(metrics["sabmark_twi_recall"])

    if scores:
        return round(sum(scores) / len(scores), 4)

    return None


def create_leaderboard_entries(metrics: Dict[str, Dict[str, float]]) -> List[Dict]:
    """Create leaderboard entries from extracted metrics."""

    # Model mapping - keys match the exact labels from CSV data
    model_info = {
        "ProteinGLM_100B_INT4": {
            "model": "OTalign (ProteinGLM 100B INT4)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with ProteinGLM 100B INT4 embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "100B INT4",
        },
        "AnkhCL": {
            "model": "OTalign (AnkhCL)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with AnkhCL embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "1.15B",
        },
        "ESM1b_33_650M": {
            "model": "OTalign (ESM-1b)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with ESM-1b embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "650M",
        },
        "ESM2_12_35M": {
            "model": "OTalign (ESM-2 35M)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with ESM-2 35M embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "35M",
        },
        "ESM2_30_150M": {
            "model": "OTalign (ESM-2 150M)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with ESM-2 150M embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "150M",
        },
        "ESM2_33_650M": {
            "model": "OTalign (ESM-2 650M)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with ESM-2 650M embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "650M",
        },
        "ESM2_36_3B": {
            "model": "OTalign (ESM-2 3B)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with ESM-2 3B embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "3B",
        },
        "ESM2_6_8M": {
            "model": "OTalign (ESM-2 8M)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with ESM-2 8M embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "8M",
        },
        "ProtT5_XL_UniRef50": {
            "model": "OTalign (ProtT5-XL)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with ProtT5-XL embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "3B",
        },
        # Ankh variants
        "Ankh_base": {
            "model": "OTalign (Ankh Base)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with Ankh Base embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "450M",
        },
        "Ankh_large": {
            "model": "OTalign (Ankh Large)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with Ankh Large embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "1.15B",
        },
        "Ankh3_large": {
            "model": "OTalign (Ankh3 Large)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with Ankh3 Large embeddings",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "1.15B",
        },
        # LoRA fine-tuned models
        "ESM1b_LoRA_ft2_2": {
            "model": "OTalign (ESM-1b LoRA ft2_2)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with LoRA fine-tuned ESM-1b embeddings (ft2_2)",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "650M",
        },
        "ESM1b_LoRA_ft5_10": {
            "model": "OTalign (ESM-1b LoRA ft5_10)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with LoRA fine-tuned ESM-1b embeddings (ft5_10)",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "650M",
        },
        "ESM1b_LoRA_finetune_ot_head_1": {
            "model": "OTalign (ESM-1b LoRA OT Head)",
            "type": "OTalign",
            "description": "Optimal Transport alignment with LoRA fine-tuned ESM-1b and OT head",
            "organization": "DeepFold",
            "paper_url": "",
            "code_url": "https://github.com/DeepFoldProtein/OTalign",
            "parameters": "650M",
        },
        # PLM-based methods
        "DeepBLAST": {
            "model": "DeepBLAST (ProtT5-XL)",
            "type": "PLM-based",
            "description": "Deep learning protein sequence alignment using bidirectional LSTM",
            "organization": "flatironinstitute",
            "paper_url": "https://doi.org/10.1038/s41587-023-01917-2",
            "code_url": "https://github.com/flatironinstitute/deepblast",
            "parameters": "3B",
        },
        "PLMAlign_ProtT5_global": {
            "model": "PLMAlign (ProtT5-XL, Global)",
            "type": "PLM-based",
            "description": "Protein language model alignment with ProtT5-XL global alignment",
            "organization": "Shanfeng Zhu Lab",
            "paper_url": "https://doi.org/10.1038/s41467-024-46808-5",
            "code_url": "https://github.com/maovshao/PLMAlign",
            "parameters": "3B",
        },
        # Traditional methods
        "HHAlign": {
            "model": "HHAlign",
            "type": "Traditional",
            "description": "Profile-profile alignment with MSAs",
            "organization": "Söding Lab",
            "paper_url": "https://doi.org/10.1093/bioinformatics/bti125",
            "code_url": "https://github.com/soedinglab/hh-suite",
            "parameters": "N/A",
        },
        "NWAlign": {
            "model": "Needleman-Wunsch",
            "type": "Traditional",
            "description": "Dynamic programming with substitution matrices",
            "organization": "Zhang Lab",
            "paper_url": "https://doi.org/10.1016/0022-2836(70)90057-4",
            "code_url": "https://zhanggroup.org/NW-align/",
            "parameters": "N/A",
        },
    }

    entries = []

    for model_key, model_metrics in metrics.items():
        if model_key not in model_info:
            print(f"Warning: Unknown model {model_key}, skipping")
            continue
        info = model_info[model_key]
        # Calculate average F1
        avg = calculate_average(model_metrics)

        entry = {
            "rank": 0,  # Will be set later based on sorting
            "model": info["model"],
            "type": info["type"],
            "description": info["description"],
            "paper_url": info["paper_url"],
            "code_url": info["code_url"],
            "parameters": info["parameters"],
            "average": avg,
            "malidup_f1": model_metrics.get("malidup_f1"),
            "malisam_f1": model_metrics.get("malisam_f1"),
            "sabmark_sup_recall": model_metrics.get("sabmark_sup_recall"),
            "sabmark_twi_recall": model_metrics.get("sabmark_twi_recall"),
            "date_submitted": "2025-09-19",
            "organization": info["organization"],
        }
        entries.append(entry)

    # Sort by average F1 (descending), with None values at the end
    entries.sort(key=lambda x: x["average"] if x["average"] is not None else -1, reverse=True)

    # Assign ranks
    for i, entry in enumerate(entries):
        entry["rank"] = i + 1

    return entries


def generate_typescript_content(entries: List[Dict]) -> str:
    """Generate TypeScript content for data.ts file."""

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    content = """import { LeaderboardData } from "./types";

export const leaderboardData: LeaderboardData = {
  leaderboard_data: [
"""

    for entry in entries:
        content += f'''    {{
      rank: {entry["rank"]},
      model: "{entry["model"]}",
      type: "{entry["type"]}",
      description: "{entry["description"]}",
      paper_url: "{entry["paper_url"]}",
      code_url: "{entry["code_url"]}",
      parameters: "{entry["parameters"]}",
      average: {entry["average"] if entry["average"] is not None else "null"},
      malidup_f1: {entry["malidup_f1"] if entry["malidup_f1"] is not None else "null"},
      malisam_f1: {entry["malisam_f1"] if entry["malisam_f1"] is not None else "null"},
      sabmark_sup_recall: {entry["sabmark_sup_recall"] if entry["sabmark_sup_recall"] is not None else "null"},
      sabmark_twi_recall: {entry["sabmark_twi_recall"] if entry["sabmark_twi_recall"] is not None else "null"},
      date_submitted: "{entry["date_submitted"]}",
      organization: "{entry["organization"]}",
    }},
'''

    content += f'''  ],
  metadata: {{
    last_updated: "{timestamp}",
    total_models: {len(entries)},
    datasets: ["MALIDUP", "MALISAM", "SABmark"],
    metrics: ["F1 Score", "Recall"],
    version: "1.0.0",
  }},
}};
'''

    return content


def main():
    """Main function to update leaderboard data."""

    # Paths
    eval_dir = "/gpfs/deepfold/work/otalign/eval/results"
    output_file = "/store/deepfold/users/baehanjin/work/OTalign/nextjs-leaderboard/src/lib/data.ts"

    print("Reading CSV data...")
    csv_data = read_csv_data(eval_dir)

    print("Extracting metrics...")
    metrics = extract_metrics(csv_data)

    print("Creating leaderboard entries...")
    entries = create_leaderboard_entries(metrics)

    print("Generating TypeScript content...")
    ts_content = generate_typescript_content(entries)

    print(f"Writing to {output_file}...")
    with open(output_file, "w") as f:
        f.write(ts_content)

    print("Done!")

    # Print summary
    print("\nLeaderboard Summary:")
    print("=" * 50)
    for entry in entries:
        avg = entry["average"] if entry["average"] is not None else "N/A"
        print(f"{entry['rank']}. {entry['model']}: {avg}")


if __name__ == "__main__":
    main()
