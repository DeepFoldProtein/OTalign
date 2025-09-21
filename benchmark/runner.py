import argparse
import sys
from pathlib import Path

import yaml


# Add the project root to the Python path to allow importing from 'otalign'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from benchmark.modules import get_evaluator_class


def run_benchmarks(config, args):
    """Orchestrates the benchmark runs."""
    print("Starting benchmark runs...")

    tests_to_run = config["tests"]

    for test_name, test_config in tests_to_run.items():
        if args.dataset and test_name != args.dataset:
            continue

        for dataset_name in test_config["datasets"]:
            dataset_config = config["datasets"][dataset_name]
            dataset_config["name"] = dataset_name  # Add name to config for easy access

            for model_key in test_config["models"]:
                model_config = config["models"][model_key]
                model_config["_key"] = model_key  # Inject the key for the evaluator

                try:
                    EvaluatorClass = get_evaluator_class(model_config["tool"])
                    evaluator = EvaluatorClass(model_config, dataset_config, config["paths"], args)
                    evaluator.run()
                except ImportError as e:
                    print(f"Failed to get evaluator for tool '{model_config['tool']}'. Error: {e}")
                except Exception as e:
                    print(f"An error occurred while running the benchmark for {model_key} on {dataset_name}: {e}")


from benchmark.plotter import Plotter


def generate_plots(config, args):
    """Orchestrates plot generation."""
    plotter = Plotter(config, args)
    plotter.generate_all_plots()


def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description="OTalign Benchmark Suite Runner")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- 'run' command ---
    run_parser = subparsers.add_parser("run", help="Execute benchmark tests")
    run_parser.add_argument("--dataset", type=str, help="Run benchmarks only for a specific dataset (e.g., 'malidup')")
    run_parser.add_argument("--update", action="store_true", help="Force re-computation of results")
    run_parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on (e.g., 'cpu', 'cuda'). Auto-detects if not set.")
    run_parser.set_defaults(func=run_benchmarks)

    # --- 'plot' command ---
    plot_parser = subparsers.add_parser("plot", help="Generate plots from benchmark results")
    plot_parser.add_argument("--dataset", type=str, help="Generate plots only for a specific dataset")
    plot_parser.set_defaults(func=generate_plots)

    args = parser.parse_args()

    # Load the central configuration file
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Execute the function associated with the chosen command
    args.func(config, args)


if __name__ == "__main__":
    main()
