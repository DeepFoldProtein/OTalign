import logging

from tqdm import tqdm

from benchmark.modules.base_evaluator import BaseEvaluator
from scripts.run_deepblast_on_dataset import run_deepblast_evaluation


class DeepblastEvaluator(BaseEvaluator):
    """Evaluator for the DeepBLAST tool."""

    def run(self):
        """
        Executes the DeepBLAST benchmark by calling the run_deepblast_evaluation function.
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
                run_deepblast_evaluation(
                    dataset=dataset_id,
                    deepblast_ckpt=self.global_paths["deepblast_checkpoint"],
                    output=str(self.results_file),
                    alignment_mode=self.model_config["params"].get("alignment_mode", "smith-waterman"),
                    pbar=pbar,
                )
        except Exception as e:
            logging.error(f"An unexpected error occurred while running the NWalign evaluation for {self.model_config['label']}: {e}")
