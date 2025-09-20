#!/bin/bash

# Benchmark script for all missing ESM2 models
# Runs benchmarks across all datasets (SABmark, MALIDUP, MALISAM) using 4 GPUs
# Requires caches to be built first (run build_all_caches.sh)

set -e

# Configuration
SCRIPT_DIR="/store/deepfold/users/baehanjin/work/OTalign"
CACHE_ROOT=".cache"
OUTPUT_ROOT="benchmark_results"
DTYPE="fp32"

# Missing ESM2 models to benchmark
# MODELS=("ESM2_6_8M" "ESM2_12_35M" "ESM2_30_150M" "ESM2_33_650M" "ESM2_36_3B")
# MODELS=("ESM2_6_8M")
# MODELS=("ESM2_6_8M" "ESM2_12_35M" "ESM2_30_150M" "ESM2_36_3B")
MODELS=("ProteinGLM_100B_INT4")





# Datasets and configurations
DATASETS=(
    # "DeepFoldProtein/SABmark-dataset,sup,test"
    "DeepFoldProtein/SABmark-dataset,twi,test" 
    # "DeepFoldProtein/malidup-dataset,all,test"
    # "DeepFoldProtein/malisam-dataset,all,test"
)

# Determine batch sizes based on model size and GPU memory
get_benchmark_batch_size() {
    local model=$1
    case $model in
        "ESM2_6_8M") echo 16 ;;
        "ESM2_12_35M") echo 16 ;;
        "ESM2_30_150M") echo 16 ;;
        "ESM2_33_650M") echo 16 ;;
        "ESM2_36_3B") echo 16 ;;
        "ProteinGLM_100B_INT4") echo 32 ;;
        *) echo 16 ;;
    esac
}

cd "$SCRIPT_DIR"
source .venv.syntax/bin/activate
mkdir -p "$OUTPUT_ROOT"

echo "========================================"
echo "ESM2 Model Benchmark Pipeline"
echo "========================================"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${#DATASETS[@]} configurations"
echo "Using 4 GPUs in parallel"
echo "========================================"

# Check if caches exist before starting
echo "Checking cache availability..."
python3 << EOF
import os
import sys
import glob
from pathlib import Path

cache_root = ".cache"
models = "${MODELS[*]}".split()
dataset_configs = "${DATASETS[*]}".split(" ")

# Parse dataset configurations
datasets = []
for config in dataset_configs:
    if config.startswith("DeepFoldProtein/"):
        parts = config.split(',')
        if len(parts) >= 3:
            dataset = parts[0]
            subset = parts[1]
            split = parts[2]
            datasets.append((dataset, subset, split))
        else:
            print(f"Warning: Invalid dataset config format: {config}")
            continue

# Model name mapping from script names to actual HF model names
model_mapping = {
    "ESM2_6_8M": "esm2_t6_8M_UR50D",
    "ESM2_12_35M": "esm2_t12_35M_UR50D", 
    "ESM2_30_150M": "esm2_t30_150M_UR50D",
    "ESM2_33_650M": "esm2_t33_650M_UR50D",
    "ESM2_36_3B": "esm2_t36_3B_UR50D",
    "ProteinGLM_100B_INT4": "proteinglm-100b-int4"
}

missing_caches = []

for model in models:
    actual_model_name = model_mapping.get(model, model)
    for dataset, subset, split in datasets:
        dataset_short = dataset.split('/')[-1]  # Extract dataset name without prefix
        # Look for cache directories matching the pattern
        pattern = f"{cache_root}/{dataset_short}__{actual_model_name}__fp32_*__v2_lmdb"
        matching_caches = glob.glob(pattern)
        
        if not matching_caches:
            missing_caches.append(f"{model} - {dataset}/{subset}/{split}")
        else:
            # Check if the cache directory has content
            cache_path = Path(matching_caches[0])
            if not cache_path.exists() or not any(cache_path.iterdir()):
                missing_caches.append(f"{model} - {dataset}/{subset}/{split}")

if missing_caches:
    print("✗ Missing caches detected:")
    for cache in missing_caches:
        print(f"  - {cache}")
    print("\nPlease run 'scripts/esm2_family/build_all_caches.sh' first to build missing caches.")
    sys.exit(1)
else:
    print("✓ All required caches are available!")
EOF

if [ $? -ne 0 ]; then
    echo "Exiting due to missing caches."
    exit 1
fi

# Function to find the correct cache directory for a model-dataset combination
find_cache_dir() {
    local model=$1
    local dataset=$2
    local subset=$3
    local split=$4
    
    # Model name mapping
    local actual_model_name
    case $model in
        "ESM2_6_8M") actual_model_name="esm2_t6_8M_UR50D" ;;
        "ESM2_12_35M") actual_model_name="esm2_t12_35M_UR50D" ;;
        "ESM2_30_150M") actual_model_name="esm2_t30_150M_UR50D" ;;
        "ESM2_33_650M") actual_model_name="esm2_t33_650M_UR50D" ;;
        "ESM2_36_3B") actual_model_name="esm2_t36_3B_UR50D" ;;
        "ProteinGLM_100B_INT4") actual_model_name="proteinglm-100b-int4" ;;
        *) actual_model_name="$model" ;;
    esac
    
    local dataset_short=$(echo "$dataset" | cut -d'/' -f2)
    local pattern="$CACHE_ROOT/${dataset_short}__${actual_model_name}__fp32_*__v2_lmdb"
    local cache_path=$(ls -d $pattern 2>/dev/null | head -1)
    
    echo "$cache_path"
}

# Function to run benchmark for a model-dataset combination
run_benchmark() {
    local model=$1
    local dataset=$2
    local subset=$3
    local split=$4
    local gpu=$5
    local batch_size=$6
    
    local dataset_short=$(echo "$dataset" | cut -d'/' -f2)
    local output_file="$OUTPUT_ROOT/otalign-${model}-${dataset_short}-${subset}.jsonl"
    local cache_dir=$(find_cache_dir "$model" "$dataset" "$subset" "$split")
    
    if [ -z "$cache_dir" ]; then
        echo "[GPU$gpu] ERROR: Cache directory not found for $model on $dataset/$subset/$split"
        return 1
    fi
    
    echo "[GPU$gpu] Running benchmark: $model on $dataset/$subset/$split"
    echo "[GPU$gpu] Using cache: $cache_dir"
    echo "[GPU$gpu] Running: python scripts/run_otalign_on_dataset.py --dataset $dataset,$subset,$split --model $model --cache_dir $cache_dir --align_batch_size $batch_size --device cuda:$gpu --output $output_file"
    python scripts/run_otalign_on_dataset.py \
        --dataset "$dataset,$subset,$split" \
        --model "$model" \
        --cache_dir "$cache_dir" \
        --align_batch_size "$batch_size" \
        --device "cuda:$gpu" \
        --output "$output_file" &
}



# Run benchmarks in parallel
echo "========================================"
echo "Running Benchmarks"
echo "========================================"

gpu_counter=0
job_count=0

for model in "${MODELS[@]}"; do
    benchmark_batch_size=$(get_benchmark_batch_size "$model")
    
    for dataset_config in "${DATASETS[@]}"; do
        IFS=',' read -r dataset subset split <<< "$dataset_config"
        
        gpu=$((gpu_counter % 4))
        run_benchmark "$model" "$dataset" "$subset" "$split" "$gpu" "$benchmark_batch_size"
        
        gpu_counter=$((gpu_counter + 1))
        job_count=$((job_count + 1))
        
        # Wait for every 4 jobs to avoid overloading
        if [ $((job_count % 4)) -eq 0 ]; then
            echo "Waiting for batch of 4 benchmarks to complete..."
            wait
        fi
    done
done

# Wait for any remaining benchmarks
echo "Waiting for remaining benchmarks to complete..."
wait

echo "✓ All benchmarks completed successfully!"

# Collect and summarize results
echo "========================================"
echo "Collecting Results"
echo "========================================"

python3 << EOF
import json
import os
from collections import defaultdict

results_dir = "benchmark_results"
models = "${MODELS[*]}".split()
dataset_configs = "${DATASETS[*]}".split(" ")

# Parse dataset configurations
datasets = []
for config in dataset_configs:
    if config.startswith("DeepFoldProtein/"):
        parts = config.split(',')
        if len(parts) >= 3:
            dataset = parts[0].split("/")[1]  # Extract dataset name without prefix
            subset = parts[1]
            split = parts[2]
            datasets.append((dataset, subset, split))
        else:
            print(f"Warning: Invalid dataset config format: {config}")
            continue

# Collect all results
all_results = defaultdict(dict)

for model in models:
    for dataset, subset, split in datasets:
        filename = f"otalign-{model}-{dataset}-{subset}.jsonl"
        filepath = os.path.join(results_dir, filename)
        
        if os.path.exists(filepath):
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
            all_results[model][f"{dataset}_{subset}"] = avg_metrics
            
            print(f"✓ {model} on {dataset}/{subset}: {len(records)} pairs, F1={avg_metrics.get('f1', 0):.4f}, Recall={avg_metrics.get('recall', 0):.4f}")
        else:
            print(f"✗ Missing results for {model} on {dataset}/{subset}")

# Save summary
summary_file = os.path.join(results_dir, "benchmark_summary.json")
with open(summary_file, 'w') as f:
    json.dump(dict(all_results), f, indent=2)

print(f"\n✓ Results summary saved to {summary_file}")

# Print formatted table
print("\n" + "="*80)
print("BENCHMARK RESULTS SUMMARY")
print("="*80)

# Dynamic header generation
header = f"{'Model':<15}"
for dataset, subset, split in datasets:
    column_name = f"{dataset}({subset})"
    header += f" {column_name:<12}"
print(header)
print("-" * 80)

for model in models:
    row_str = f"{model:<15}"
    for dataset, subset, split in datasets:
        key = f"{dataset}_{subset}"
        if key in all_results[model]:
            recall = all_results[model][key].get('recall', 0)
            row_str += f" {recall:.4f}{'':>6}"
        else:
            row_str += f" {'N/A':<12}"
    
    print(row_str)

print("="*80)
EOF

echo "========================================"
echo "Benchmark Pipeline Complete!"
echo "========================================"
echo "Results are saved in: $OUTPUT_ROOT/"
echo "Summary file: $OUTPUT_ROOT/benchmark_summary.json"
echo "========================================"
