# OTalign Benchmarking Suite

This directory contains a comprehensive, modular suite for benchmarking the performance of `OTalign` and other alignment tools across various datasets. It is designed for reproducibility, ease of use, and extensibility.

## Directory Structure

```text
benchmark/
├── runner.py                # Main CLI entry point for running benchmarks and plotting
├── config.yaml              # Central configuration for all benchmarks
└── modules/                 # Pluggable evaluation modules for each tool
    ├── __init__.py
    ├── base_evaluator.py    # Abstract base class for evaluators
    ├── otalign_evaluator.py # Evaluator for OTalign
    ├── hhalign_evaluator.py # Evaluator for HHalign
    └── ...                  # Other evaluators
.cache/                      # Caches for embeddings and intermediate results
out/
├── results/                 # Raw output data from benchmark runs
│   └── <dataset_name>/
│       └── <model_name>/
│           ├── results.jsonl
│           └── transport_plans/
│               └── <pair_id>.npz
└── plots/                   # Generated plots for analysis
    └── <dataset_name>/
        ├── all_metrics.png
        └── recall_vs_precision.png
```

## Core Concepts

### 1. Centralized Configuration (`config.yaml`)

All aspects of the benchmark are controlled by `config.yaml`. This file defines:

- **Models**: Specifies which models to test, their parameters (e.g., `epsilon`, `tau`), display names for plots, and colors.
- **Datasets**: Lists the HuggingFace dataset identifiers for `SABmark`, `Malidup`, and `Malisam`.
- **Tests**: Maps which models should be run on which datasets.
- **Paths**: Defines the locations for output directories, caches, and executables.

### 2. Modular Evaluators

Each alignment tool is handled by a dedicated evaluator class in the `modules/` directory. This design makes it simple to add new tools to the benchmark by creating a new evaluator that inherits from `base_evaluator.py`.

### 3. Caching and Smart Updates

The suite is designed to be efficient:

- **Result Caching**: It checks for existing result files before re-running a benchmark.
- **`--update` Flag**: When this flag is used, the runner will intelligently re-compute results only if the underlying data or configuration has changed, by checking modification times or checksums.
- **Raw Data Storage**: All raw outputs, including transport plans (`.npz` files), are stored in a structured `results/` directory to ensure full reproducibility.

### 4. Command-Line Interface (`runner.py`)

The `runner.py` script provides a simple yet powerful CLI to manage the benchmark process.

**Usage:**

- **Run all benchmarks defined in `config.yaml`:**

  ```bash
  python benchmark/runner.py run
  ```

- **Run benchmarks for a specific dataset:**

  ```bash
  python benchmark/runner.py run --dataset malidup
  ```

- **Force re-computation of all results:**

  ```bash
  python benchmark/runner.py run --update
  ```

- **Generate all plots:**

  ```bash
  python benchmark/runner.py plot
  ```

- **Generate plots for a specific dataset:**

  ```bash
  python benchmark/runner.py plot --dataset sabmark_twi
  ```

### 5. Plotting

The plotting functionality generates publication-quality figures based on the results. It automatically uses the labels and colors defined in `config.yaml` to create:

- A comprehensive plot showing all relevant metrics (F1, Precision, Recall, Jaccard).
- Individual plots for key metrics, such as Recall on SABmark, for more detailed analysis.
