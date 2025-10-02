# OTalign Benchmarking Suite

This directory contains a comprehensive, modular suite for benchmarking the performance of `OTalign` and other alignment tools across various datasets. It is designed for reproducibility, ease of use, and extensibility.

## Directory Structure

```text
benchmark/
├── __main__.py              # Main CLI entry point for the benchmark suite
├── runner.py                # Core logic for running benchmarks
├── plotter.py               # Core logic for generating plots
├── config.yaml              # Central configuration for all benchmarks
└── modules/                 # Pluggable evaluation modules for each tool
    ├── __init__.py
    ├── base_evaluator.py    # Abstract base class for evaluators
    ├── otalign_evaluator.py # Evaluator for OTalign
    ├── hhalign_evaluator.py # Evaluator for HHalign
    └── ...                  # Other evaluators
out/
├── cache/                   # Caches for embeddings and intermediate results
├── results/                 # Raw output data from benchmark runs
│   └── <dataset_name>/
│       └── <model_name>/
│           ├── results.jsonl
│           └── transport_plans/
│               └── <pair_id>.npz
└── plots/                   # Generated plots for analysis
    └── <dataset_name>/
        ├── all_metrics.png
        ├── sabmark_recall_bar.png
        └── sabmark_metrics_box.png
```

## Core Concepts

### 1. Centralized Configuration (`config.yaml`)

All aspects of the benchmark are controlled by `config.yaml`. This file defines:

- **Models**: Specifies which models to test, their parameters (e.g., `epsilon`, `tau`), display names for plots, and colors.
- **Datasets**: Lists the HuggingFace dataset identifiers for `SABmark`, `Malidup`, and `Malisam`.
- **Tests**: Maps which models should be run on which datasets.
- **Paths**: Defines the locations for output directories, caches, and executables.
- **Plotting**: Defines how to generate plots for each benchmark group, including titles, metrics, and plot types.
- **Global Plot Style**: A `plot_style` section can be added to the root of `config.yaml` to define global styles (e.g., font size, figure size, and legend) that apply to all plots.

### 2. Modular Evaluators

Each alignment tool is handled by a dedicated evaluator class in the `modules/` directory. This design makes it simple to add new tools to the benchmark by creating a new evaluator that inherits from `base_evaluator.py`.

### 3. Caching and Smart Updates

The suite is designed to be efficient:

- **Result Caching**: It checks for existing result files before re-running a benchmark.
- **`--update` Flag**: When this flag is used, the runner will intelligently re-compute results only if the underlying data or configuration has changed, by checking modification times or checksums.
- **Raw Data Storage**: All raw outputs, including transport plans (`.npz` files), are stored in a structured `results/` directory to ensure full reproducibility.

### 4. Command-Line Interface (`__main__.py`)

The benchmark suite is run through a command-line interface, invoked using `python -m benchmark`. This entry point is managed by `__main__.py`, which provides two main commands: `run` and `plot`.

**Usage:**

- **Run all benchmarks defined in `config.yaml`:**

  ```bash
  python -m benchmark run
  ```

- **Run benchmarks for a specific dataset:**

  ```bash
  python -m benchmark run --dataset malidup
  ```

- **Force re-computation of all results:**

  ```bash
  python -m benchmark run --update
  ```

- **Generate all plots:**

  ```bash
  python -m benchmark plot
  ```

- **Generate plots for a specific test group:**

  ```bash
  python -m benchmark plot --test sabmark
  ```

  This command will generate all plots defined under the `sabmark` key in the `plotting` section of `config.yaml`.

### 5. Plotting and Configuration

The plotting functionality is highly configurable via `config.yaml`, allowing for the creation of publication-quality figures. The generated plots are saved in `out/plots/<test_name>/`.

A global `plot_style` section can be added to the root of `config.yaml` to define styles (e.g., font size, figure size) that apply to all plots. The legend can also be controlled globally from here.

#### Plotting Configuration (`config.yaml`)

The `plotting` section is organized by **test groups**. For each group, you can define a dictionary of `plots` to generate, where each key is the output filename.

- **`plots`**: A dictionary of plot objects, where each key is the plot's filename (e.g., `sabmark_metrics_box`).
  - **`type`**: The plot type, typically `boxplot` or `barplot`.
  - **`title`**: The plot title.
  - **`metrics`**: A list of metrics to include.
  - **`legend`**: (Optional) Controls the legend. This can be set globally in `plot_style` or per-plot. The per-plot setting overrides the global one.
    - **As a string**: `"bottom"`, `"upper right"`, `"outside"`, `"none"`, etc.
    - **As a dictionary**:
      - `position`: A string for the location (e.g., `"upper right"`, `"outside"`).
      - `ncol`: The number of columns.

Here is an example demonstrating global and local legend settings:

```yaml
# Global style settings
plot_style:
  figsize: [10, 7]
  legend: "upper right" # Set the legend globally for all plots

# ... other sections ...

# 5. Plotting Configuration
plotting:
  sabmark:
    plots:
      sabmark_metrics_box:
        type: "boxplot"
        title: "SABmark Benchmark"
        metrics: ["recall", "precision"]
        # This plot will use the global legend setting ("upper right")

      sabmark_recall_bar:
        type: "barplot"
        title: "SABmark Recall Comparison"
        metrics: ["recall"]
        legend: "none" # Override the global setting for this specific plot
```
