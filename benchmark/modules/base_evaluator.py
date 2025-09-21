import abc
import logging
from pathlib import Path


class BaseEvaluator(abc.ABC):
    """
    Abstract base class for a benchmark evaluator.

    Each subclass is responsible for evaluating a specific alignment tool
    (e.g., OTalign, HHalign) on a given dataset.
    """

    def __init__(self, model_config, dataset_config, global_paths, cli_args):
        """
        Initializes the evaluator.

        Args:
            model_config (dict): The configuration for the specific model/tool from config.yaml.
            dataset_config (dict): The configuration for the dataset from config.yaml.
            global_paths (dict): The global paths configuration from config.yaml.
            cli_args (argparse.Namespace): The command-line arguments.
        """
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.global_paths = global_paths
        self.cli_args = cli_args
        self.tool_name = model_config["tool"]
        self.model_name = self._get_model_name_for_dir()

        # Define output paths
        self.results_dir = Path(global_paths["results_dir"]) / self.dataset_config["name"] / self.model_name
        self.results_file = self.results_dir / "results.jsonl"
        self.transport_plan_dir = self.results_dir / "transport_plans"

        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.tool_name == "otalign":
            self.transport_plan_dir.mkdir(exist_ok=True)

    def should_run(self) -> bool:
        """
        Determines if the benchmark should be run based on the `--update` flag
        and the existence and content of result files.
        """
        if self.cli_args.update:
            return True

        if not self.results_file.exists() or self.results_file.stat().st_size == 0:
            return True

        logging.info(f"Results for {self.model_name} on {self.dataset_config['name']} already exist. Skipping.")
        return False

    def _get_model_name_for_dir(self) -> str:
        """
        Returns the name used for the model's results directory.
        This should be the unique key from the config.yaml.
        """
        return self.model_config["_key"]

    @abc.abstractmethod
    def run(self):
        """
        Executes the benchmark for the specific tool and dataset.
        This method must be implemented by all subclasses.
        """
        raise NotImplementedError
