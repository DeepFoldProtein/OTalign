import logging

from tqdm import tqdm

from benchmark.modules.base_evaluator import BaseEvaluator
from scripts.run_nwalign_on_dataset import run_nwalign_evaluation


class NwalignEvaluator(BaseEvaluator):
    """Evaluator for the NWalign tool."""

    def run(self):
        """
        Executes the NWalign benchmark by calling the run_nwalign_evaluation function.
        """
        if not self.should_run():
            return

        # Construct the dataset identifier
        dataset_id = self.dataset_config["id"]
        if "config" in self.dataset_config:
            dataset_id += f",{self.dataset_config['config']}"
        if "split" in self.dataset_config:
            dataset_id += f",{self.dataset_config['split']}"

        try:
            with tqdm(total=1, desc=f"Evaluating {self.model_config['label']}", leave=False) as pbar:
                run_nwalign_evaluation(
                    dataset=dataset_id,
                    nwalign_bin=self.global_paths["nwalign_bin"],
                    output=str(self.results_file),
                    glocal=self.model_config["params"].get("glocal", 0),
                    workers=self.cli_args.workers,
                    pbar=pbar,
                )
        except Exception as e:
            logging.error(f"An unexpected error occurred while running the NWalign evaluation for {self.model_config['label']}: {e}")
