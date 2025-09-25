import logging
from pathlib import Path

from tqdm import tqdm

from benchmark.modules.base_evaluator import BaseEvaluator
from scripts.build_cache import build_cache
from scripts.run_plmalign_on_dataset import run_plmalign_evaluation


class PlmalignEvaluator(BaseEvaluator):
    """Evaluator for the PLMAlign tool."""

    def _get_model_name_for_dir(self) -> str:
        """
        Returns the directory name for the model, using the model key from the config.
        """
        return self.model_config["_key"]

    def run(self):
        """
        Executes the PLMAlign benchmark by calling the run_plmalign_evaluation function.
        """
        if not self.should_run():
            return

        dataset_id = self.dataset_config["id"]
        if "config" in self.dataset_config:
            dataset_id += f",{self.dataset_config['config']}"
        if "split" in self.dataset_config:
            dataset_id += f",{self.dataset_config['split']}"

        base_cache_dir = Path(self.global_paths["cache_dir"]) / self.dataset_config["name"] / self.model_name

        if not base_cache_dir.exists() or not any(base_cache_dir.iterdir()):
            logging.info(f"Cache not found for {self.model_name} on {self.dataset_config['name']}. Building cache...")
            self._build_cache_internal(base_cache_dir, dataset_id)

        try:
            actual_cache_dir = next(d for d in base_cache_dir.iterdir() if d.is_dir())
            logging.info(f"Found actual cache directory: {actual_cache_dir}")
        except StopIteration:
            logging.error(f"Cache directory '{base_cache_dir}' is empty or contains no subdirectories after build attempt.")
            raise FileNotFoundError(f"Cache not built correctly in {base_cache_dir}")

        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

        try:
            with tqdm(total=1, desc=f"Evaluating {self.model_config['label']}", leave=False) as pbar:
                run_plmalign_evaluation(
                    dataset=dataset_id,
                    output=str(self.results_file),
                    alignment_mode=self.model_config["params"].get("alignment_mode", "global"),
                    plm_model=self.model_config["params"].get("plm_model", "ProtT5_XL_UniRef50"),
                    cache_dir=str(actual_cache_dir),
                    device=device,
                    pbar=pbar,
                )
        except Exception as e:
            logging.error(f"An unexpected error occurred while running the PLMAlign evaluation for {self.model_config['label']}: {e}")

    def _build_cache_internal(self, cache_dir, dataset_id):
        """Builds the embedding cache using the imported build_cache function."""
        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        model_path = self.model_config.get("checkpoint_path")
        if not model_path:
            model_path = self.model_config["plm"]
        base_model = self.model_config.get("base_model")

        try:
            logging.info(f"  - Building cache with model: {model_path}")
            with tqdm(total=1, desc=f"Building cache for {self.model_config['label']}", leave=False) as pbar:
                build_cache(
                    dataset=dataset_id,
                    model=model_path,
                    output_root=str(cache_dir),
                    base_model_for_checkpoint=base_model,
                    device=device,
                    no_tqdm=False,  # Enable tqdm
                    pbar=pbar,  # Pass the progress bar
                )
            logging.info("Cache build complete.")
        except Exception as e:
            logging.error(f"Error building cache for {self.model_config['label']}: {e}")
            raise e
