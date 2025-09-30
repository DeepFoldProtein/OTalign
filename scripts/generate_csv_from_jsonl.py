"""
Script to generate CSV files from JSONL evaluation results.
Processes JSONL files organized in the directory structure:
/path/to/results/{dataset}/{model}/results.jsonl
Calculates alignment metrics summary for each dataset.
"""

import argparse
import glob
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def extract_model_name(filepath: str) -> str:
    """Extract model name from JSONL filepath."""
    # The filepath format is: /path/to/results/{dataset}/{model}/results.jsonl
    # Extract the model name from the parent directory
    model_dir = os.path.basename(os.path.dirname(filepath))

    # Clean up model names to match existing format
    model_mapping = {
        "otalign_esm1b": "ESM1b_33_650M",
        "otalign_esm2": "ESM2_33_650M",
        "otalign_esm2_6_8m": "ESM2_6_8M",
        "otalign_esm2_12_35m": "ESM2_12_35M",
        "otalign_esm2_30_150m": "ESM2_30_150M",
        "otalign_esm2_36_3b": "ESM2_36_3B",
        "otalign_ankh_base": "Ankh_base",
        "otalign_ankh_large": "Ankh_large",
        "otalign_ankh3_large": "Ankh3_large",
        "otalign_ankh3_xl": "Ankh3_xl",
        "otalign_ankhcl": "AnkhCL",
        "otalign_prott5": "ProtT5_XL_UniRef50",
        "otalign_proteinglm_100b_int4": "ProteinGLM_100B_INT4",
        "otalign_esm1b_lora_ft2_2": "ESM1b_LoRA_ft2_2",
        "otalign_esm1b_lora_ft5_10": "ESM1b_LoRA_ft5_10",
        "esm1b-lora-finetune-ot-head-1": "ESM1b_LoRA_finetune_ot_head_1",
        "plmalign_prott5_global": "PLMAlign_ProtT5_global",
        "plmalign_prott5_global_before": "PLMAlign_ProtT5_global_before",
        "hhalign": "HHAlign",
        "nwalign": "NWAlign",
        "deepblast": "DeepBLAST",
        "baseline": "Baseline",
    }

    return model_mapping.get(model_dir, model_dir)


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values."""
    if len(values) < 2:
        return np.nan, np.nan

    mean = np.mean(values)
    sem = stats.sem(values)

    # Calculate confidence interval
    h = sem * stats.t.ppf((1 + confidence) / 2.0, len(values) - 1)

    return mean - h, mean + h


def process_jsonl_file(filepath: str) -> Dict[str, List[float]]:
    """Process a single JSONL file and extract metrics."""
    metrics_data = {"precision": [], "recall": [], "f1": [], "jaccard": []}

    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    metrics = data.get("metrics", {})

                    # Extract metrics, filtering out NaN values
                    for metric in ["precision", "recall", "f1", "jaccard"]:
                        if metric in metrics:
                            value = metrics[metric]
                            # Skip NaN values (분모가 0인 경우 제외)
                            if not (math.isnan(value) if isinstance(value, float) else False):
                                metrics_data[metric].append(value)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {}

    return metrics_data


def calculate_summary_stats(values: List[float]) -> Dict[str, float]:
    """Calculate summary statistics for a list of values."""
    if not values:
        return {"mean": np.nan, "sem": np.nan, "std": np.nan, "ci_95_lower": np.nan, "ci_95_upper": np.nan}

    values_array = np.array(values)
    mean_val = np.mean(values_array)
    std_val = np.std(values_array, ddof=1)  # Sample standard deviation
    sem_val = stats.sem(values_array)
    ci_lower, ci_upper = calculate_confidence_interval(values)

    return {"mean": mean_val, "sem": sem_val, "std": std_val, "ci_95_lower": ci_lower, "ci_95_upper": ci_upper}


def process_dataset(results_dir: str, dataset: str) -> pd.DataFrame:
    """Process all JSONL files for a specific dataset."""

    # Build the dataset directory path
    dataset_dir = os.path.join(results_dir, dataset)

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return pd.DataFrame()

    # Find all results.jsonl files in model subdirectories
    pattern = os.path.join(dataset_dir, "*/results.jsonl")
    jsonl_files = glob.glob(pattern)

    print(f"Processing {dataset}:")
    print(f"  Dataset directory: {dataset_dir}")
    print(f"  Found {len(jsonl_files)} results.jsonl files")

    results = []
    processed_models = set()

    # Process all files
    for filepath in jsonl_files:
        model_name = extract_model_name(filepath)

        # Skip if we already processed this model
        if model_name in processed_models:
            continue

        model_dir = os.path.basename(os.path.dirname(filepath))
        print(f"  Processing {model_dir}/results.jsonl -> {model_name}")

        metrics_data = process_jsonl_file(filepath)

        if not metrics_data:
            continue

        # Calculate summary statistics for each metric
        for metric, values in metrics_data.items():
            if values:  # Only process if we have data
                stats_dict = calculate_summary_stats(values)

                result_row = {
                    "label": model_name,
                    "metric": metric.capitalize(),  # F1, Precision, Recall, Jaccard
                    **stats_dict,
                }
                results.append(result_row)

        processed_models.add(model_name)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by label and metric for consistency
    if not df.empty:
        df = df.sort_values(["label", "metric"]).reset_index(drop=True)

    return df


def main():
    """Main function to process all datasets."""
    parser = argparse.ArgumentParser(description="Generate CSV files from JSONL evaluation results")
    parser.add_argument("--results_dir", type=str, default="/gpfs/deepfold/work/otalign/eval/results", help="Directory containing results subdirectories")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for CSV files (defaults to results_dir)")

    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    # Datasets to process (based on the terminal output)
    datasets = ["malidup", "malisam", "sabmark_sup", "sabmark_sup_fp", "sabmark_twi", "sabmark_twi_fp"]

    print(f"Processing evaluation results from: {results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # Process the dataset
        df = process_dataset(results_dir, dataset)

        if df.empty:
            print(f"  No data found for {dataset}")
            continue

        # Create output directory
        dataset_output_dir = os.path.join(output_dir, f"_{dataset}")
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Save CSV file
        output_file = os.path.join(dataset_output_dir, "alignment_metrics_summary.csv")
        df.to_csv(output_file, index=False, float_format="%.6f")

        print(f"  Saved {len(df)} rows to {output_file}")
        print(f"  Models: {sorted(df['label'].unique())}")

    print("\n" + "=" * 60)
    print("Processing complete!")


if __name__ == "__main__":
    main()
