"""
Script to generate CSV files from JSONL evaluation results.
Processes JSONL files to calculate alignment metrics summary for each dataset.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def extract_model_name(filename: str) -> str:
    """Extract model name from JSONL filename."""
    basename = os.path.basename(filename)
    
    # Remove dataset suffix and .jsonl extension
    for dataset in ['_malidup', '_malisam', '_sabmark-sup', '_sabmark-twi']:
        if dataset in basename:
            basename = basename.replace(dataset, '')
            break
    
    # Remove .jsonl extension
    basename = basename.replace('.jsonl', '')
    
    # Handle special cases
    if basename.startswith('otalign-'):
        model_name = basename[8:]  # Remove 'otalign-' prefix
    elif basename.startswith('baseline'):
        model_name = 'baseline'
    elif basename == 'hhalign':
        model_name = 'hhalign'
    elif basename == 'nwalign':
        model_name = 'nwalign'
    else:
        model_name = basename
    
    # Clean up model names to match existing format
    model_mapping = {
        'ESM1b': 'ESM1b',
        'esm1b': 'ESM1b',
        'ESM2_12_35M': 'ESM2_12_35M',
        'ESM2_30_150M': 'ESM2_30_150M',
        'ESM2_33_650M': 'ESM2_33_650M',
        'ESM2_36_3B': 'ESM2_36_3B',
        'ESM2_6_8M': 'ESM2_6_8M',
        'AnkhCL': 'AnkhCL',
        'ProtT5_XL_UniRef50': 'ProtT5_XL_UniRef50',
        'hhalign': 'hhalign',
        'nwalign': 'nwalign',
        'baseline': 'baseline'
    }
    
    return model_mapping.get(model_name, model_name)


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values."""
    if len(values) < 2:
        return np.nan, np.nan
    
    mean = np.mean(values)
    sem = stats.sem(values)
    
    # Calculate confidence interval
    h = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
    
    return mean - h, mean + h


def process_jsonl_file(filepath: str) -> Dict[str, List[float]]:
    """Process a single JSONL file and extract metrics."""
    metrics_data = {
        'precision': [],
        'recall': [],
        'f1': [],
        'jaccard': []
    }
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    metrics = data.get('metrics', {})
                    
                    # Extract metrics
                    for metric in ['precision', 'recall', 'f1', 'jaccard']:
                        if metric in metrics:
                            metrics_data[metric].append(metrics[metric])
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {}
    
    return metrics_data


def calculate_summary_stats(values: List[float]) -> Dict[str, float]:
    """Calculate summary statistics for a list of values."""
    if not values:
        return {
            'mean': np.nan,
            'sem': np.nan,
            'std': np.nan,
            'ci_95_lower': np.nan,
            'ci_95_upper': np.nan
        }
    
    values_array = np.array(values)
    mean_val = np.mean(values_array)
    std_val = np.std(values_array, ddof=1)  # Sample standard deviation
    sem_val = stats.sem(values_array)
    ci_lower, ci_upper = calculate_confidence_interval(values)
    
    return {
        'mean': mean_val,
        'sem': sem_val,
        'std': std_val,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper
    }


def process_dataset(eval_dir: str, dataset: str) -> pd.DataFrame:
    """Process all JSONL files for a specific dataset."""
    
    # Find all JSONL files for this dataset
    pattern = os.path.join(eval_dir, f"*{dataset}.jsonl")
    jsonl_files = glob.glob(pattern)
    
    # Also check for files ending with _tm.jsonl (transmembrane variants)
    tm_pattern = os.path.join(eval_dir, f"*{dataset}_tm.jsonl")
    tm_files = glob.glob(tm_pattern)
    
    print(f"Processing {dataset}:")
    print(f"  Found {len(jsonl_files)} regular files")
    print(f"  Found {len(tm_files)} TM files")
    
    results = []
    processed_models = set()
    
    # Process regular files first
    for filepath in jsonl_files:
        model_name = extract_model_name(filepath)
        
        # Skip if we already processed this model
        if model_name in processed_models:
            continue
            
        print(f"  Processing {os.path.basename(filepath)} -> {model_name}")
        
        metrics_data = process_jsonl_file(filepath)
        
        if not metrics_data:
            continue
            
        # Calculate summary statistics for each metric
        for metric, values in metrics_data.items():
            if values:  # Only process if we have data
                stats_dict = calculate_summary_stats(values)
                
                result_row = {
                    'label': model_name,
                    'metric': metric.capitalize(),  # F1, Precision, Recall, Jaccard
                    **stats_dict
                }
                results.append(result_row)
        
        processed_models.add(model_name)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by label and metric for consistency
    if not df.empty:
        df = df.sort_values(['label', 'metric']).reset_index(drop=True)
    
    return df


def main():
    """Main function to process all datasets."""
    parser = argparse.ArgumentParser(description='Generate CSV files from JSONL evaluation results')
    parser.add_argument('--eval_dir', type=str, default='/gpfs/deepfold/work/otalign/eval',
                        help='Directory containing JSONL files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for CSV files (defaults to eval_dir)')
    
    args = parser.parse_args()
    
    eval_dir = args.eval_dir
    output_dir = args.output_dir or eval_dir
    
    # Datasets to process
    datasets = ['malidup', 'malisam', 'sabmark-sup', 'sabmark-twi']
    
    print(f"Processing evaluation results from: {eval_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        # Process the dataset
        df = process_dataset(eval_dir, dataset)
        
        if df.empty:
            print(f"  No data found for {dataset}")
            continue
        
        # Create output directory
        dataset_output_dir = os.path.join(output_dir, f"_{dataset}")
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Save CSV file
        output_file = os.path.join(dataset_output_dir, "alignment_metrics_summary.csv")
        df.to_csv(output_file, index=False, float_format='%.6f')
        
        print(f"  Saved {len(df)} rows to {output_file}")
        print(f"  Models: {sorted(df['label'].unique())}")
    
    print("\n" + "="*60)
    print("Processing complete!")


if __name__ == "__main__":
    main()
