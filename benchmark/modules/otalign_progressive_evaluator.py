import logging
from pathlib import Path

from tqdm import tqdm

from benchmark.modules.base_evaluator import BaseEvaluator
from scripts.build_cache import build_cache
from scripts.run_otalign_progressive_on_dataset import run_otalign_evaluation


class OtalignProgressiveEvaluator(BaseEvaluator):
    """
    Evaluator for the 'otalign-progressive' tool.
    """

    def _get_cache_model_name(self) -> str:
        """
        Returns the directory name for the cache, based on the underlying model.
        This allows sharing the cache between different methods using the same model.
        """
        # The cache is determined by the PLM, not the specific OTalign method config.
        # If a fine-tuned checkpoint is used, it gets its own cache.
        # Otherwise, we use the base PLM name.
        model_path = self.model_config.get("checkpoint_path")
        if not model_path:
            model_path = self.model_config["plm"]

        # Create a filesystem-friendly name from the model path/name.
        # This is a simple way to handle HF model IDs like 'facebook/esm2'.
        return model_path.replace("/", "__")

    def _get_model_name_for_dir(self) -> str:
        """
        Returns the directory name for the model, using the model key from the config.
        """
        return self.model_config["_key"]

    def __init__(self, model_config, dataset_config, global_paths, cli_args):
        super().__init__(model_config, dataset_config, global_paths, cli_args)
        self.tool_name = "otalign-progressive"

    def run(self):
        """
        Executes the OTAlign progressive benchmark.
        """
        if not self.should_run():
            return

        dataset_id = self.dataset_config["id"]
        if "config" in self.dataset_config:
            dataset_id += f",{self.dataset_config['config']}"
        if "split" in self.dataset_config:
            dataset_id += f",{self.dataset_config['split']}"

        # Prepare arguments for the evaluation function
        params = self.model_config.get("params", {})

        # Determine the model path or name
        cache_model_name = self._get_cache_model_name()
        base_cache_dir = Path(self.global_paths["cache_dir"]) / self.dataset_config["name"] / cache_model_name

        if not base_cache_dir.exists() or not any(base_cache_dir.iterdir()):
            logging.info(f"Cache not found for model '{cache_model_name}' on {self.dataset_config['name']}. Building cache...")
            self._build_cache_internal(base_cache_dir, dataset_id)

        try:
            actual_cache_dir = next(d for d in base_cache_dir.iterdir() if d.is_dir())
            logging.info(f"Found actual cache directory: {actual_cache_dir}")
        except StopIteration:
            logging.error(f"Cache directory '{base_cache_dir}' is empty or contains no subdirectories after build attempt.")
            raise FileNotFoundError(f"Cache not built correctly in {base_cache_dir}")

        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

        model_path = self.model_config.get("checkpoint_path")
        if not model_path:
            model_path = self.model_config["plm"]
        base_model = self.model_config.get("base_model")

        with tqdm(total=1, desc=f"Aligning {self.dataset_config['name']} with {self.model_name}") as pbar:
            try:
                run_otalign_evaluation(
                    dataset=dataset_id,
                    model=model_path,
                    output=str(self.results_file),
                    base_model_for_checkpoint=base_model,
                    cache_dir=str(actual_cache_dir),
                    device=device,
                    dtype=params.get("dtype", "fp32"),
                    reg_init=params.get("reg_init", 1.0),
                    reg_final=params.get("reg_final", 0.01),
                    reg_steps=params.get("reg_steps", 5),
                    reg_m=params.get("reg_m", 5.0),
                    dp_mu=params.get("dp_mu", 8.0),
                    num_iter=params.get("num_iter", 50000),
                    eval_band_width=self.dataset_config.get("eval_band_width", 0),
                    align_batch_size=params.get("align_batch_size", 16),
                    export_fasta_dir=None,  # Not needed for benchmark runs
                    no_tqdm=True,  # Disable inner tqdm
                    pbar=pbar,
                )
                logging.info(f"Successfully ran OTAlign-Progressive for {self.model_name} on {self.dataset_config['name']}.")
            except Exception as e:
                logging.error(f"Error running OTAlign-Progressive for {self.model_name} on {self.dataset_config['name']}: {e}", exc_info=True)

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
