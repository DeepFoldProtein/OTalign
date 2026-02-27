"""Evaluator for DEDAL (https://github.com/DeepFoldProtein/dedal-fork)."""

import logging

from benchmark.modules.base_evaluator import BaseEvaluator


class DedalEvaluator(BaseEvaluator):
    """Evaluator for DEDAL; uses TensorFlow Hub model, no embedding cache."""

    def _get_model_name_for_dir(self) -> str:
        return self.model_config["_key"]

    def run(self):
        if not self.should_run():
            return

        from scripts.run_dedal_on_dataset import run_dedal_evaluation

        dataset_id = self.dataset_config["id"]
        if "config" in self.dataset_config:
            dataset_id += f",{self.dataset_config['config']}"
        if "split" in self.dataset_config:
            dataset_id += f",{self.dataset_config['split']}"

        params = self.model_config.get("params", {})
        model_url = self.model_config.get("model_url", "https://tfhub.dev/google/dedal/3")
        max_length = params.get("max_length", 512)
        use_gpu = params.get("use_gpu", True)  # Default: use GPU (faster)

        try:
            # Let run_dedal_evaluation use its own progress bar (correct total, visible updates)
            run_dedal_evaluation(
                dataset=dataset_id,
                output=str(self.results_file),
                model_url=model_url,
                max_length=max_length,
                use_gpu=use_gpu,
                pbar=None,
            )
        except Exception as e:
            logging.error("DEDAL evaluation failed for %s: %s", self.model_config["label"], e)
            raise
