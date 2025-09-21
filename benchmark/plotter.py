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
        self.plot_format = cli_args.plot_format if hasattr(cli_args, "plot_format") and cli_args.plot_format else "png"
        self.plots_dir.mkdir(exist_ok=True)

    def generate_all_plots(self):
        """Generates all plots defined in the configuration."""
        logging.info("Generating all plots...")

        # Determine which test groups to plot
        if self.cli_args.test:
            if self.cli_args.test not in self.config["plotting"]:
                logging.error(f"Test group '{self.cli_args.test}' not found in 'plotting' configuration. Available groups: {list(self.config['plotting'].keys())}")
                return
            tests_to_plot = {self.cli_args.test: self.config["tests"][self.cli_args.test]}
        else:
            tests_to_plot = self.config["tests"]

        with tqdm(tests_to_plot.items(), desc="Generating Plot Groups") as pbar:
            for test_name, test_config in pbar:
                pbar.set_description(f"Plotting group: {test_name}")
                if test_name in self.config["plotting"]:
                    self.generate_plots_for_test_group(test_name, test_config)
                else:
                    logging.warning(f"  - No plotting configuration found for test group '{test_name}'. Skipping.")

    def generate_plots_for_test_group(self, test_name, test_config):
        """Generates all defined plots for a specific test group."""
        logging.info(f"  - Generating plots for test group: {test_name}")

        # 1. Aggregate data from all datasets and models in this test group
        all_data = []
        for dataset_name in test_config["datasets"]:
            for model_key in test_config["models"]:
                model_config = self.config["models"][model_key]
                results_file = self.results_dir / dataset_name / model_key / "results.jsonl"

                if not results_file.exists() or results_file.stat().st_size == 0:
                    logging.warning(f"    - Results file not found or empty for {model_config['label']} on {dataset_name}. Skipping.")
                    continue

                with open(results_file, "r") as f:
                    for line in f:
                        res = json.loads(line)
                        metrics = res.get("metrics", {})
                        for metric_name, value in metrics.items():
                            all_data.append({"label": model_config["label"], "metric": metric_name, "value": value, "color": model_config["color"], "dataset": dataset_name})

        if not all_data:
            logging.warning(f"    - No data found for test group {test_name}. Skipping plot generation.")
            return

        df = pd.DataFrame(all_data)
        df["metric"] = df["metric"].str.replace("_", " ").str.title()

        # 2. Generate each plot defined in the plotting configuration for this test group
        plot_group_config = self.config["plotting"][test_name]
        for plot_config in plot_group_config.get("plots", []):
            self._create_plot(df, test_name, plot_config)

    def _create_plot(self, df, test_name, plot_config):
        """Helper function to create a single plot based on its configuration."""
        plot_name = plot_config["name"]
        logging.info(f"    - Creating plot: {plot_name}")

        metrics_to_plot = [m.title() for m in plot_config["metrics"]]
        plot_df = df[df["metric"].isin(metrics_to_plot)]

        if plot_df.empty:
            logging.warning(f"      - No data for metrics {metrics_to_plot} in test group {test_name}. Skipping plot '{plot_name}'.")
            return

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(12, 8))

        plot_type = plot_config.get("type", "boxplot")
        palette = {row["label"]: row["color"] for _, row in plot_df[["label", "color"]].drop_duplicates().iterrows()}

        if plot_type == "boxplot":
            plot = sns.boxplot(data=plot_df, x="metric", y="value", hue="label", palette=palette)
        elif plot_type == "barplot" or plot_type == "bar":
            plot = sns.barplot(data=plot_df, x="metric", y="value", hue="label", palette=palette, capsize=0.1, errorbar=("ci", 95))
        else:
            logging.warning(f"      - Unknown plot type '{plot_type}' for plot '{plot_name}'. Skipping.")
            return

        plot.set_title(plot_config.get("title", f"{test_name.replace('_', ' ').title()} Comparison"), fontsize=18)
        plot.set_xlabel("Metric", fontsize=14)
        plot.set_ylabel("Value", fontsize=14)
        plot.tick_params(axis="x", labelsize=12)
        plot.tick_params(axis="y", labelsize=12)
        plot.set_ylim(0, 1.05)

        # Adjust legend
        handles, labels = plot.get_legend_handles_labels()
        plot.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=min(len(palette), 4), title="Model", fontsize=12)

        plt.tight_layout(rect=(0, 0.05, 1, 1))

        # Save the plot in a directory named after the test group
        output_dir = self.plots_dir / test_name
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{plot_name}.{self.plot_format}"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"      - Plot saved to {output_path}")
