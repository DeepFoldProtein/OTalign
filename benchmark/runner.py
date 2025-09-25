import logging
import sys
from pathlib import Path


# Add the project root to the Python path to allow importing from 'otalign'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from benchmark.modules import get_evaluator_class


def run_benchmarks(config, args):
    """Orchestrates the benchmark runs."""
    logging.info("Starting benchmark runs...")

    tests_to_run = config["tests"]
    if args.test and args.test in tests_to_run:
        tests_to_run = {args.test: tests_to_run[args.test]}

    for test_name, test_config in tests_to_run.items():
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
                    logging.error(f"Failed to get evaluator for tool '{model_config['tool']}'. Error: {e}")
                except Exception as e:
                    logging.error(f"An error occurred while running the benchmark for {model_key} on {dataset_name}: {e.__class__.__name__}: {e}")


from benchmark.plotter import Plotter


def generate_plots(config, args):
    """Orchestrates plot generation."""
    plotter = Plotter(config, args)
    plotter.generate_all_plots()
