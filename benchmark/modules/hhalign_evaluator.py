import logging

from tqdm import tqdm

from benchmark.modules.base_evaluator import BaseEvaluator
from scripts.run_hhalign_on_dataset import run_hhalign_evaluation


class HhalignEvaluator(BaseEvaluator):
    """Evaluator for the HHalign tool."""

    def run(self):
        """
        Executes the HHalign benchmark by calling the run_hhalign_evaluation function.
        """
        if not self.should_run():
            return

        # Construct the dataset identifier
        dataset_id = self.dataset_config["id"]
        if "config" in self.dataset_config:
            dataset_id += f",{self.dataset_config['config']}"
        if "split" in self.dataset_config:
            dataset_id += f",{self.dataset_config['split']}"

        hhm_dir = self.dataset_config.get("hhm_dir")
        if not hhm_dir:
            logging.warning(f"  - 'hhm_dir' not specified for dataset '{self.dataset_config['name']}'. Skipping HHalign run.")
            return

        try:
            with tqdm(total=1, desc=f"Evaluating {self.model_config['label']}", leave=False) as pbar:
                run_hhalign_evaluation(
                    dataset=dataset_id,
                    hhm_dir=hhm_dir,
                    hhalign_bin=self.global_paths["hhalign_bin"],
                    output=str(self.results_file),
                    mode=self.model_config["params"].get("mode", "local"),
                    workers=self.cli_args.workers,
                    pbar=pbar,
                )
        except Exception as e:
            logging.error(f"An unexpected error occurred while running the HHalign evaluation for {self.model_config['label']}: {e}")
