#!/bin/bash

# Cache building script for all missing ESM2 models
# Builds embedding caches across all datasets (SABmark, MALIDUP, MALISAM) using 4 GPUs

set -e

# Configuration
SCRIPT_DIR="/store/deepfold/users/baehanjin/work/OTalign"
CACHE_ROOT=".cache"
DTYPE="fp32"

# Missing ESM2 models to build caches for
# MODELS=("ESM2_6_8M" "ESM2_12_35M" "ESM2_33_650M" "ESM2_30_150M" "ESM2_36_3B")
# MODELS=("ESM2_6_8M")
MODELS=("ProteinGLM_100B_INT4")




# Datasets and configurations (format: dataset,config,split)
DATASETS=(
    # "DeepFoldProtein/SABmark-dataset,sup,test"
    "DeepFoldProtein/SABmark-dataset,twi,test" 
    "DeepFoldProtein/malidup-dataset,all,test"
    "DeepFoldProtein/malisam-dataset,all,test"
)

# Determine batch sizes based on model size and GPU memory
get_cache_batch_size() {
    local model=$1
    case $model in
        "ESM2_6_8M") echo 16 ;;
        "ESM2_12_35M") echo 12 ;;
        "ESM2_30_150M") echo 8 ;;
        "ESM2_33_650M") echo 6 ;;
        "ESM2_36_3B") echo 4 ;;
        "ProteinGLM_100B_INT4") echo 4 ;;
        *) echo 4 ;;
    esac
}

cd "$SCRIPT_DIR"
source .venv.syntax/bin/activate
mkdir -p "$CACHE_ROOT"

echo "========================================"
echo "ESM2 Model Cache Building Pipeline"
echo "========================================"
echo "Models: ${MODELS[*]}"
echo "Datasets: ${#DATASETS[@]} configurations"
echo "Using 4 GPUs in parallel"
echo "========================================"

# Function to build cache for a model-dataset combination
build_cache() {
    local model=$1
    local dataset_spec=$2
    local gpu=$3
    local batch_size=$4
    
    echo "[GPU$gpu] Building cache: $model on $dataset_spec"
    python scripts/build_cache.py \
        --dataset "$dataset_spec" \
        --model "$model" \
        --output_root "$CACHE_ROOT" \
        --dtype "$DTYPE" \
        --batch_size "$batch_size" \
        --device "cuda:$gpu" \
        --shard_size 100 &
}



# Build all embedding caches in parallel
echo "========================================"
echo "Building Embedding Caches"
echo "========================================"

gpu_counter=0
job_count=0

for model in "${MODELS[@]}"; do
    cache_batch_size=$(get_cache_batch_size "$model")
    
    for dataset_spec in "${DATASETS[@]}"; do
        gpu=$((gpu_counter % 4))
        build_cache "$model" "$dataset_spec" "$gpu" "$cache_batch_size"
        
        gpu_counter=$((gpu_counter + 1))
        job_count=$((job_count + 1))
        
        # Wait for every 4 jobs to avoid overloading
        if [ $((job_count % 4)) -eq 0 ]; then
            echo "Waiting for batch of 4 cache builds to complete..."
            wait
        fi
    done
done

# Wait for any remaining cache builds
echo "Waiting for remaining cache builds to complete..."
wait

echo "✓ All embedding caches built successfully!"

# Check cache status
echo "========================================"
echo "Cache Build Status Summary"
echo "========================================"

python3 << EOF
import os
import glob
from pathlib import Path

cache_root = ".cache"
models = "${MODELS[*]}".split()
dataset_specs = "${DATASETS[*]}".split()

# Parse dataset configurations (format: dataset,config,split)
datasets = []
for spec in dataset_specs:
    parts = spec.split(",")
    if len(parts) >= 2:
        dataset = parts[0]
        config = parts[1]
        datasets.append((dataset, config))
    else:
        print(f"Warning: Invalid dataset spec format: {spec}")
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

total_caches = 0
built_caches = 0

print(f"{'Model':<15} {'Dataset':<25} {'Config':<10} {'Status':<10}")
print("-" * 70)

for model in models:
    actual_model_name = model_mapping.get(model, model)
    for dataset, config in datasets:
        total_caches += 1
        
        dataset_short = dataset.split('/')[-1]  # Extract dataset name without prefix
        # Look for cache directories matching the pattern
        pattern = f"{cache_root}/{dataset_short}-{config}__{actual_model_name}__fp32_*__v2_lmdb"
        matching_caches = glob.glob(pattern)
        
        if matching_caches:
            # Check if the cache directory has content
            cache_path = Path(matching_caches[0])
            if cache_path.exists() and any(cache_path.iterdir()):
                status = "✓ Built"
                built_caches += 1
            else:
                status = "✗ Missing"
        else:
            status = "✗ Missing"
        
        print(f"{model:<15} {dataset_short:<25} {config:<10} {status:<10}")

print("-" * 70)
print(f"Cache Summary: {built_caches}/{total_caches} built successfully")

if built_caches == total_caches:
    print("✓ All caches are ready for benchmarking!")
else:
    print(f"⚠ {total_caches - built_caches} caches are missing")
EOF

echo "========================================"
echo "Cache Building Complete!"
echo "========================================"
echo "Cache location: $CACHE_ROOT/"
echo "Run 'scripts/run_all_benchmarks.sh' to start benchmarking"
echo "========================================"
