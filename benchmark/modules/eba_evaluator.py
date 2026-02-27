"""Evaluator for EBA (Embedding-based alignment, https://github.com/DeepFoldProtein/EBA)."""

import logging

from benchmark.modules.base_evaluator import BaseEvaluator


class EbaEvaluator(BaseEvaluator):
    """Evaluator for EBA; uses a PLM (ProtT5/ESMb1) for embeddings and DTW alignment."""

    def _get_model_name_for_dir(self) -> str:
        return self.model_config["_key"]

    def run(self):
        if not self.should_run():
            return

        from scripts.run_eba_on_dataset import run_eba_evaluation

        dataset_id = self.dataset_config["id"]
        if "config" in self.dataset_config:
            dataset_id += f",{self.dataset_config['config']}"
        if "split" in self.dataset_config:
            dataset_id += f",{self.dataset_config['split']}"

        params = self.model_config.get("params", {})
        plm = params.get("plm", "ProtT5")
        device = params.get("device")
        l_sim = params.get("l", 1.0)

        try:
            run_eba_evaluation(
                dataset=dataset_id,
                output=str(self.results_file),
                plm=plm,
                device=device,
                l_sim=l_sim,
                pbar=None,
            )
        except Exception as e:
            logging.error("EBA evaluation failed for %s: %s", self.model_config["label"], e)
            raise
