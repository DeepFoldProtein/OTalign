import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


class Plotter:
    """Handles the generation of plots from benchmark results."""

    def __init__(self, config, cli_args):
        self.config = config
        self.cli_args = cli_args
        self.results_dir = Path(config["paths"]["results_dir"])
        self.plots_dir = Path(config["paths"]["plots_dir"])
        self.plots_dir.mkdir(exist_ok=True)

    def generate_all_plots(self):
        """Generates all plots defined in the configuration."""
        logging.info("Generating all plots...")
        tests_to_plot = self.config["tests"]
        if self.cli_args.dataset and self.cli_args.dataset in tests_to_plot:
            tests_to_plot = {self.cli_args.dataset: tests_to_plot[self.cli_args.dataset]}

        with tqdm(tests_to_plot.items(), desc="Generating Plot Groups") as pbar:
            for test_name, test_config in pbar:
                pbar.set_description(f"Plotting group: {test_name}")
                for dataset_name in test_config["datasets"]:
                    self.generate_plots_for_dataset(dataset_name, test_config["models"])

    def generate_plots_for_dataset(self, dataset_name, model_keys):
        """Generates plots for a single dataset."""
        logging.info(f"  - Generating plots for dataset: {dataset_name}")

        # Load data for all models for this dataset
        all_data = []
        for model_key in model_keys:
            model_config = self.config["models"][model_key]

            # The directory name is simply the model key
            model_name_for_dir = model_key
            results_file = self.results_dir / dataset_name / model_name_for_dir / "results.jsonl"

            if not results_file.exists() or results_file.stat().st_size == 0:
                logging.warning(f"    - Results file not found or is empty for {model_config['label']}. Skipping.")
                continue

            with open(results_file, "r") as f:
                for line in f:
                    res = json.loads(line)
                    # Also check for 'new_metrics' which some evaluators might produce
                    metrics_sources = [res.get("metrics", {}), res.get("new_metrics", {})]
                    for metrics_source in metrics_sources:
                        for metric_name, value in metrics_source.items():
                            all_data.append({"label": model_config["label"], "metric": metric_name, "value": value, "color": model_config["color"]})

        if not all_data:
            logging.warning(f"    - No data found for dataset {dataset_name}. Skipping plot generation.")
            return

        df = pd.DataFrame(all_data)
        df["metric"] = df["metric"].str.replace("_", " ").str.title()

        # Get plotting configuration for this dataset
        plot_config = self.config["plotting"].get(dataset_name.split("_")[0])  # e.g., 'sabmark' from 'sabmark_twilight'
        if not plot_config:
            logging.warning(f"    - No plotting configuration for {dataset_name}. Skipping.")
            return

        # Generate the main plot with all metrics
        self._create_plot(df, dataset_name, plot_config, "all_metrics")

        # Generate any custom plots
        if "custom_plots" in plot_config:
            for custom_plot_config in plot_config["custom_plots"]:
                self._create_plot(df, dataset_name, custom_plot_config, custom_plot_config["name"])

    def _create_plot(self, df, dataset_name, plot_config, plot_name):
        """Helper function to create a single plot."""

        metrics_to_plot = [m.capitalize() for m in plot_config["metrics"]]
        plot_df = df[df["metric"].isin(metrics_to_plot)]

        if plot_df.empty:
            return

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 7))

        plot_type = plot_config.get("type", "boxplot")

        # Get unique labels and their colors to create a palette
        palette = {row["label"]: row["color"] for _, row in plot_df[["label", "color"]].drop_duplicates().iterrows()}

        if plot_type == "boxplot":
            plot = sns.boxplot(data=plot_df, x="metric", y="value", hue="label", palette=palette)
        elif plot_type == "barplot" or plot_type == "bar":
            plot = sns.barplot(data=plot_df, x="metric", y="value", hue="label", palette=palette, capsize=0.1, errorbar=("ci", 95))
        else:
            logging.warning(f"    - Unknown plot type '{plot_type}'. Skipping.")
            return

        plot.set_title(plot_config.get("title", f"{dataset_name.replace('_', ' ').title()} Comparison"), fontsize=16)
        plot.set_xlabel("Metric", fontsize=12)
        plot.set_ylabel("Value", fontsize=12)
        plot.set_ylim(0, 1.05)
        plot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, title="Model")

        plt.tight_layout()

        output_dir = self.plots_dir / dataset_name
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{plot_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"    - Plot saved to {output_path}")
