import argparse
import logging
import sys
from pathlib import Path

import yaml
from tqdm import tqdm


# Add the project root to the Python path to allow importing from 'otalign'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from benchmark.runner import generate_plots, run_benchmarks


# Tqdm-friendly logging handler
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


def setup_logging():
    """Sets up root logger to use TQDM-friendly handler."""
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    # Remove any existing handlers
    for handler in log.handlers[:]:
        log.removeHandler(handler)
    # Add our custom handler
    log.addHandler(TqdmLoggingHandler())


def main():
    """Main entry point for the benchmark runner."""
    setup_logging()
    parser = argparse.ArgumentParser(description="OTalign Benchmark Suite Runner")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- 'run' command ---
    run_parser = subparsers.add_parser("run", help="Execute benchmark tests")
    run_parser.add_argument("--dataset", type=str, help="Run benchmarks only for a specific dataset (e.g., 'malidup')")
    run_parser.add_argument("--update", action="store_true", help="Force re-computation of results")
    run_parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on (e.g., 'cpu', 'cuda'). Auto-detects if not set.")
    run_parser.add_argument("--workers", type=int, default=1, help="Number of workers for tools like NWalign and HHalign.")
    run_parser.set_defaults(func=run_benchmarks)

    # --- 'plot' command ---
    plot_parser = subparsers.add_parser("plot", help="Generate plots from benchmark results")
    plot_parser.add_argument("--test", type=str, help="Generate plots only for a specific test group (e.g., 'sabmark', 'finetune-sab')")
    plot_parser.add_argument("--plot-format", type=str, default="png", help="Output format for plots (e.g., 'png', 'svg', 'pdf')")
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
