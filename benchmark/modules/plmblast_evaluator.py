"""Evaluator for the pLM-BLAST tool (https://github.com/labstructbioinf/pLM-BLAST)."""

import logging
from pathlib import Path

from tqdm import tqdm

from benchmark.modules.base_evaluator import BaseEvaluator
from scripts.build_cache import build_cache


class PlmblastEvaluator(BaseEvaluator):
    """Evaluator for pLM-BLAST; uses same ProtT5 cache as PLMAlign when available."""

    def _get_model_name_for_dir(self) -> str:
        return self.model_config["_key"]

    def run(self):
        if not self.should_run():
            return

        from scripts.run_plmblast_on_dataset import run_plmblast_evaluation

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
            logging.error("Cache directory '%s' is empty or contains no subdirectories after build.", base_cache_dir)
            raise FileNotFoundError(f"Cache not built correctly in {base_cache_dir}") from None

        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        params = self.model_config.get("params", {}).copy()
        global_alignment = params.pop("global_alignment", True)

        try:
            with tqdm(total=1, desc=f"Evaluating {self.model_config['label']}", leave=False) as pbar:
                run_plmblast_evaluation(
                    dataset=dataset_id,
                    output=str(self.results_file),
                    cache_dir=str(actual_cache_dir),
                    global_alignment=global_alignment,
                    device=device,
                    pbar=pbar,
                    **params,
                )
        except Exception as e:
            logging.error("pLM-BLAST evaluation failed for %s: %s", self.model_config["label"], e)
            raise

    def _build_cache_internal(self, cache_dir: Path, dataset_id: str):
        """Build ProtT5 embedding cache (same as PLMAlign) for pLM-BLAST."""
        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        model_path = self.model_config.get("checkpoint_path") or self.model_config.get("plm", "ProtT5_XL_UniRef50")
        base_model = self.model_config.get("base_model")

        logging.info("  - Building cache with model: %s", model_path)
        with tqdm(total=1, desc=f"Building cache for {self.model_config['label']}", leave=False) as pbar:
            build_cache(
                dataset=dataset_id,
                model=model_path,
                output_root=str(cache_dir),
                base_model_for_checkpoint=base_model,
                device=device,
                no_tqdm=False,
                pbar=pbar,
            )
        logging.info("Cache build complete.")
