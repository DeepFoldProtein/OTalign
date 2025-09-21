#!/bin/bash

# Summary generation script for benchmark results
# Analyzes benchmark results and generates summary tables and JSON files

set -e

# Configuration
SCRIPT_DIR="/gpfs/deepfold/users/baehanjin/work/OTalign"
OUTPUT_ROOT="benchmark_results"

# Default models and datasets (can be overridden by command line arguments)
DEFAULT_MODELS=("ESM2_6_8M" "ESM2_12_35M" "ESM2_30_150M" "ESM2_36_3B")
DEFAULT_DATASETS=(
    "DeepFoldProtein/SABmark-dataset sup"
    "DeepFoldProtein/SABmark-dataset twi" 
    "DeepFoldProtein/malidup-dataset all"
    "DeepFoldProtein/malisam-dataset all"
)

# Parse command line arguments
MODELS=()
DATASETS=()
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --datasets)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            HELP=true
            shift
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --models MODEL1 MODEL2 ...     Specify models to include in summary"
    echo "  --datasets DATASET1 DATASET2   Specify datasets to include in summary"
    echo "  --output-dir DIR                Specify output directory (default: benchmark_results)"
    echo "  --help, -h                      Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --models ESM2_6_8M ESM2_12_35M --output-dir my_results"
    exit 0
fi

# Use defaults if not specified
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=("${DEFAULT_MODELS[@]}")
fi

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

cd "$SCRIPT_DIR"

echo "========================================"
echo "Benchmark Results Summary Generator"
echo "========================================"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${#DATASETS[@]} configurations"
echo "Output directory: $OUTPUT_ROOT"
echo "========================================"

# Check if output directory exists
if [ ! -d "$OUTPUT_ROOT" ]; then
    echo "Error: Output directory '$OUTPUT_ROOT' does not exist."
    echo "Please run benchmarks first or specify correct output directory with --output-dir"
    exit 1
fi

# Collect and summarize results
echo "Collecting and analyzing results..."

python3 << EOF
import json
import os
import sys
from collections import defaultdict

results_dir = "$OUTPUT_ROOT"
models = "${MODELS[*]}".split()
dataset_configs = "${DATASETS[*]}".split(" ")

# Parse dataset configurations
datasets = []
i = 0
while i < len(dataset_configs):
    if dataset_configs[i].startswith("DeepFoldProtein/"):
        dataset = dataset_configs[i].split("/")[1]  # Extract dataset name without prefix
        name = dataset_configs[i+1]
        datasets.append((dataset, name))
        i += 2
    else:
        i += 1

print(f"Processing {len(models)} models across {len(datasets)} dataset configurations...")

# Collect all results
all_results = defaultdict(dict)
missing_results = []

for model in models:
    for dataset, name in datasets:
        filename = f"otalign_{model}_{dataset}_{name}.jsonl"
        filepath = os.path.join(results_dir, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    records = [json.loads(line) for line in f]
                
                # Calculate average metrics
                metrics = defaultdict(list)
                for record in records:
                    if 'metrics' in record:
                        for k, v in record['metrics'].items():
                            if v is not None:
                                metrics[k].append(v)
                
                avg_metrics = {k: sum(v)/len(v) if v else 0 for k, v in metrics.items()}
                all_results[model][f"{dataset}_{name}"] = avg_metrics
                
                print(f"✓ {model} on {dataset}/{name}: {len(records)} pairs, F1={avg_metrics.get('f1', 0):.4f}, Recall={avg_metrics.get('recall', 0):.4f}")
            except Exception as e:
                print(f"✗ Error processing {model} on {dataset}/{name}: {e}")
                missing_results.append(f"{model} on {dataset}/{name}")
        else:
            print(f"✗ Missing results for {model} on {dataset}/{name}")
            missing_results.append(f"{model} on {dataset}/{name}")

# Save detailed summary
summary_file = os.path.join(results_dir, "benchmark_summary.json")
with open(summary_file, 'w') as f:
    json.dump(dict(all_results), f, indent=2)

print(f"\n✓ Detailed results summary saved to {summary_file}")

# Save missing results log
if missing_results:
    missing_file = os.path.join(results_dir, "missing_results.log")
    with open(missing_file, 'w') as f:
        for missing in missing_results:
            f.write(missing + '\n')
    print(f"⚠ Missing results log saved to {missing_file}")

# Print formatted table
print("\n" + "="*80)
print("BENCHMARK RESULTS SUMMARY")
print("="*80)

# Dynamic header generation
header = f"{'Model':<15}"
for dataset, name in datasets:
    column_name = f"{dataset}({name})"
    header += f" {column_name:<12}"
print(header)
print("-" * 80)

for model in models:
    row_str = f"{model:<15}"
    for dataset, name in datasets:
        key = f"{dataset}_{name}"
        if key in all_results[model]:
            recall = all_results[model][key].get('recall', 0)
            row_str += f" {recall:.4f}{'':>6}"
        else:
            row_str += f" {'N/A':<12}"
    
    print(row_str)

print("="*80)

# Generate CSV summary for easy import into spreadsheets
csv_file = os.path.join(results_dir, "benchmark_summary.csv")
with open(csv_file, 'w') as f:
    # Write header
    header_line = "Model"
    for dataset, name in datasets:
        header_line += f",{dataset}_{name}_recall,{dataset}_{name}_f1"
    f.write(header_line + '\n')
    
    # Write data
    for model in models:
        row = model
        for dataset, name in datasets:
            key = f"{dataset}_{name}"
            if key in all_results[model]:
                recall = all_results[model][key].get('recall', 0)
                f1 = all_results[model][key].get('f1', 0)
                row += f",{recall:.4f},{f1:.4f}"
            else:
                row += ",N/A,N/A"
        f.write(row + '\n')

print(f"✓ CSV summary saved to {csv_file}")

if missing_results:
    print(f"\n⚠ Warning: {len(missing_results)} result files are missing.")
    print("Run the benchmark script first or check the output directory.")
    sys.exit(1)
else:
    print(f"\n✓ All results processed successfully!")
EOF

if [ $? -eq 0 ]; then
    echo "========================================"
    echo "Summary Generation Complete!"
    echo "========================================"
    echo "Results are available in: $OUTPUT_ROOT/"
    echo "- Detailed summary: $OUTPUT_ROOT/benchmark_summary.json"
    echo "- CSV format: $OUTPUT_ROOT/benchmark_summary.csv"
    echo "========================================"
else
    echo "Summary generation failed. Check the error messages above."
    exit 1
fi
