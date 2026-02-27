"""
Evaluator for ECOD homolog detection benchmark using pLM-BLAST.

This evaluator computes ROC/PR curves for homolog detection instead of
alignment-level metrics.
"""

import json
import logging
from pathlib import Path

from tqdm import tqdm

from benchmark.modules.base_evaluator import BaseEvaluator


class EcodHomologEvaluator(BaseEvaluator):
    """
    Evaluator for ECOD homolog detection benchmark.

    This differs from standard alignment evaluators as it:
    1. Performs database search (query vs all DB entries)
    2. Computes ROC/PR curves based on ECOD hierarchy labels
    3. Reports AUC metrics instead of alignment precision/recall
    """

    def _get_model_name_for_dir(self) -> str:
        return self.model_config["_key"]

    def run(self):
        if not self.should_run():
            return

        ecod_data_dir = Path(self.dataset_config.get("data_dir", "data/ecod"))

        # Detect hard benchmark format (all-vs-all with hard_benchmark.csv)
        hard_benchmark_csv = ecod_data_dir / "hard_benchmark.csv"
        if hard_benchmark_csv.exists():
            self._run_hard_benchmark(ecod_data_dir)
            return

        # Standard query-vs-database format
        queries_csv = ecod_data_dir / "queries.csv"
        db_csv = ecod_data_dir / "database.csv"

        if not queries_csv.exists() or not db_csv.exists():
            logging.error("ECOD dataset not prepared. Run: python scripts/prepare_ecod_dataset.py")
            logging.error(f"  Expected files: {queries_csv}, {db_csv}")
            raise FileNotFoundError(f"ECOD dataset not found in {ecod_data_dir}")

        # Build embeddings if needed
        base_cache_dir = Path(self.global_paths["cache_dir"]) / self.dataset_config["name"] / self.model_name
        query_emb_dir = base_cache_dir / "queries"
        db_emb_dir = base_cache_dir / "database"

        if not query_emb_dir.exists() or not db_emb_dir.exists():
            logging.info("Building embeddings for ECOD dataset...")
            self._build_embeddings(ecod_data_dir, base_cache_dir)

        # Run search
        try:
            logging.info(f"Running ECOD homolog detection for {self.model_config['label']}")
            self._run_search(queries_csv=queries_csv, query_emb_dir=query_emb_dir, db_csv=db_csv, db_emb_dir=db_emb_dir)

            # Compute ROC/PR curves
            logging.info("Computing ROC/PR metrics...")
            self._compute_metrics()

        except Exception as e:
            logging.error(f"ECOD evaluation failed for {self.model_config['label']}: {e}")
            raise

    def _build_embeddings(self, data_dir: Path, cache_dir: Path):
        """Build embeddings for queries and database."""
        from scripts.build_ecod_plmblast_db import build_database_embeddings

        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        model_name = self.model_config.get("plm", "ProtT5_XL_UniRef50")

        query_emb_dir = cache_dir / "queries"
        db_emb_dir = cache_dir / "database"

        # Build query embeddings
        if not query_emb_dir.exists() or not any(query_emb_dir.iterdir()):
            logging.info("Building query embeddings...")
            with tqdm(total=1, desc="Building query embeddings", leave=False) as pbar:
                build_database_embeddings(input_csv=data_dir / "queries.csv", output_dir=query_emb_dir, model_name=model_name, device=device, use_cache=True)
                pbar.update(1)

        # Build database embeddings
        if not db_emb_dir.exists() or not any(db_emb_dir.iterdir()):
            logging.info("Building database embeddings...")
            with tqdm(total=1, desc="Building database embeddings", leave=False) as pbar:
                build_database_embeddings(input_csv=data_dir / "database.csv", output_dir=db_emb_dir, model_name=model_name, device=device, use_cache=True)
                pbar.update(1)

    def _run_search(self, queries_csv: Path, query_emb_dir: Path, db_csv: Path, db_emb_dir: Path):
        """Run pLM-BLAST search for all queries."""
        import pandas as pd
        from scripts.run_ecod_plmblast_search import (
            compute_labels,
            load_query_embeddings,
            run_plmblast_search,
        )

        # Load metadata
        query_df = pd.read_csv(queries_csv)
        db_df = pd.read_csv(db_csv)

        # Get parameters (labels: TP=same H, FP=diff X, Neutral excluded)
        params = self.model_config.get("params", {})
        global_alignment = params.get("global_alignment", True)

        logging.info(f"Processing {len(query_df)} queries against {len(db_df)} database entries")
        logging.info("  - Labels: TP=same H-group, FP=different X-group, Neutral excluded")

        # Load query embeddings
        query_embeddings = load_query_embeddings(queries_csv, query_emb_dir)
        logging.info(f"Loaded {len(query_embeddings)} query embeddings")

        # Process each query
        all_results = []

        with open(self.results_file, "w") as f_out:
            for _, query_row in tqdm(query_df.iterrows(), total=len(query_df), desc=f"Search {self.model_config['label']}"):
                query_id = query_row["id"]

                if query_id not in query_embeddings:
                    logging.warning(f"No embedding for {query_id}, skipping")
                    continue

                query_emb = query_embeddings[query_id]

                try:
                    # Run search
                    results = run_plmblast_search(query_emb, db_emb_dir, global_alignment=global_alignment)

                    if len(results) == 0:
                        logging.warning(f"No results for query {query_id}")
                        continue

                    # Add labels (TP=same H, FP=diff X, Neutral excluded)
                    results_labeled = compute_labels(query_row, db_df, results)

                    result_record = {
                        "query_id": query_id,
                        "query_group": query_row["H"],
                        "num_hits": len(results_labeled),
                        "num_positives": int((results_labeled["label"] == 1).sum()),
                        "hits": results_labeled.to_dict("records"),
                    }

                    f_out.write(json.dumps(result_record) + "\n")
                    f_out.flush()

                    all_results.append(result_record)

                except Exception as e:
                    logging.error(f"Failed query {query_id}: {e}")
                    continue

        logging.info(f"Search complete: {len(all_results)} queries processed")

    def _compute_metrics(self):
        """Compute and save ROC/PR metrics."""
        from scripts.evaluate_ecod_results import (
            compute_per_query_metrics,
            compute_roc_pr_curves,
            load_search_results,
            plot_roc_pr_curves,
        )

        # Load results
        scores, labels, _ = load_search_results(self.results_file)

        if len(scores) == 0:
            logging.error("No results to evaluate")
            return

        # Compute metrics
        metrics = compute_roc_pr_curves(scores, labels)

        # Save metrics
        metrics_file = self.results_dir / "roc_pr_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    "roc_auc": float(metrics.roc_auc),
                    "pr_auc": float(metrics.pr_auc),
                    "num_positives": int(metrics.num_positives),
                    "num_negatives": int(metrics.num_negatives),
                    "total_predictions": int(metrics.total_predictions),
                },
                f,
                indent=2,
            )

        # Plot curves
        plot_dir = self.results_dir / "plots"
        plot_roc_pr_curves(metrics, plot_dir, title_prefix=f"ECOD - {self.model_config['label']}")

        # Per-query metrics
        per_query_df = compute_per_query_metrics(self.results_file, self.results_dir / "per_query_metrics.csv")

        # Log summary
        logging.info(f"ROC AUC: {metrics.roc_auc:.4f}")
        logging.info(f"PR AUC: {metrics.pr_auc:.4f}")

        valid = per_query_df.dropna(subset=["roc_auc"])
        if len(valid) > 0:
            logging.info(f"Per-query mean ROC AUC: {valid['roc_auc'].mean():.4f}")
            logging.info(f"Per-query mean PR AUC: {valid['pr_auc'].mean():.4f}")

    def _run_hard_benchmark(self, data_dir: Path):
        """Run hard benchmark (all-vs-all), dispatching to pLM-BLAST, OTalign, EBA, or DEDAL."""
        params = self.model_config.get("params", {})

        if "dedal_score_method" in params:
            self._run_hard_benchmark_dedal(data_dir)
        elif "eba_score_method" in params:
            self._run_hard_benchmark_eba(data_dir)
        elif "score_method" in params:
            self._run_hard_benchmark_otalign(data_dir)
        else:
            self._run_hard_benchmark_plmblast(data_dir)

    def _run_hard_benchmark_plmblast(self, data_dir: Path):
        """Run hard benchmark with pLM-BLAST."""
        from scripts.run_hard_benchmark_plmblast import (
            assign_labels,
            compute_embeddings,
            compute_metrics,
            load_benchmark_data,
            run_pairwise_search,
        )

        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        params = self.model_config.get("params", {})
        mode = "global" if params.get("global_alignment", True) else "local"
        num_workers = getattr(self.cli_args, "workers", 1)

        logging.info(f"Running Hard benchmark (pLM-BLAST) for {self.model_config['label']}")

        df = load_benchmark_data(data_dir)
        embeddings = compute_embeddings(df, self.results_dir, device=device)
        results = run_pairwise_search(embeddings, df["id"].tolist(), mode=mode, num_workers=num_workers)
        result_df = assign_labels(results, df)

        # Save results
        results_csv = self.results_dir / "search_results.csv"
        result_df.to_csv(results_csv, index=False)
        logging.info(f"Saved results to {results_csv}")

        # Compute and save metrics
        metrics = compute_metrics(result_df, exclude_neutral=True)
        metrics["mode"] = mode
        metrics["num_domains"] = len(df)
        metrics["num_pairs"] = len(results)

        metrics_file = self.results_dir / "roc_pr_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logging.info(f"PR AUC: {metrics['pr_auc']:.4f}")

    def _run_hard_benchmark_otalign(self, data_dir: Path):
        """Run hard benchmark with OTalign."""
        from scripts.run_hard_benchmark_otalign import (
            assign_labels,
            compute_embeddings,
            compute_metrics,
            load_benchmark_data,
            score_pairs_otalign,
        )

        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        params = self.model_config.get("params", {})
        model_name = self.model_config.get("plm", "Ankh_Large")
        score_method = params.get("score_method", "mean_cosine")
        reg = params.get("reg", 0.1)
        lambda1 = params.get("lambda1", 1.0)
        lambda2 = params.get("lambda2", 1.0)
        num_iter = params.get("num_iter", 1000)
        batch_size = params.get("align_batch_size", 16)

        logging.info(f"Running Hard benchmark (OTalign, {score_method}) for {self.model_config['label']}")

        df = load_benchmark_data(data_dir)
        embeddings = compute_embeddings(
            df,
            self.results_dir,
            model_name=model_name,
            device=device,
        )
        results = score_pairs_otalign(
            embeddings,
            df["id"].tolist(),
            score_method=score_method,
            device=device,
            batch_size=batch_size,
            reg=reg,
            lambda1=lambda1,
            lambda2=lambda2,
            num_iter=num_iter,
        )
        result_df = assign_labels(results, df)

        # Save results
        results_csv = self.results_dir / "search_results.csv"
        result_df.to_csv(results_csv, index=False)
        logging.info(f"Saved results to {results_csv}")

        # Compute and save metrics
        metrics = compute_metrics(result_df, exclude_neutral=True)
        metrics["score_method"] = score_method
        metrics["model_name"] = model_name
        metrics["num_domains"] = len(df)
        metrics["num_pairs"] = len(results)

        metrics_file = self.results_dir / "roc_pr_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logging.info(f"PR AUC: {metrics['pr_auc']:.4f}")

    def _run_hard_benchmark_eba(self, data_dir: Path):
        """Run hard benchmark with EBA."""
        from scripts.run_hard_benchmark_eba import (
            assign_labels,
            compute_embeddings,
            compute_metrics,
            load_benchmark_data,
            score_pairs_eba,
        )

        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        params = self.model_config.get("params", {})
        plm = self.model_config.get("plm", "ProtT5")
        score_method = params.get("eba_score_method", "EBA_min")
        l_sim = params.get("l_sim", 1.0)

        logging.info(f"Running Hard benchmark (EBA, {score_method}) for {self.model_config['label']}")

        df = load_benchmark_data(data_dir)
        embeddings = compute_embeddings(
            df,
            self.results_dir,
            plm=plm,
            device=device,
        )
        results = score_pairs_eba(
            embeddings,
            df["id"].tolist(),
            score_method=score_method,
            l_sim=l_sim,
        )
        result_df = assign_labels(results, df)

        # Save results
        results_csv = self.results_dir / "search_results.csv"
        result_df.to_csv(results_csv, index=False)
        logging.info(f"Saved results to {results_csv}")

        # Compute and save metrics
        metrics = compute_metrics(result_df, exclude_neutral=True)
        metrics["score_method"] = score_method
        metrics["plm"] = plm
        metrics["num_domains"] = len(df)
        metrics["num_pairs"] = len(results)

        metrics_file = self.results_dir / "roc_pr_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logging.info(f"PR AUC: {metrics['pr_auc']:.4f}")

    def _run_hard_benchmark_dedal(self, data_dir: Path):
        """Run hard benchmark with DEDAL."""
        from scripts.run_hard_benchmark_dedal import (
            assign_labels,
            compute_metrics,
            load_benchmark_data,
            load_dedal_model,
            score_pairs_dedal,
        )

        params = self.model_config.get("params", {})
        score_method = params.get("dedal_score_method", "sw_scores")
        model_url = params.get("model_url", "https://tfhub.dev/google/dedal/3")
        max_length = params.get("max_length", 512)
        use_gpu = params.get("use_gpu", False)

        logging.info(f"Running Hard benchmark (DEDAL, {score_method}) for {self.model_config['label']}")

        df = load_benchmark_data(data_dir)
        model = load_dedal_model(model_url, use_gpu=use_gpu)
        results = score_pairs_dedal(
            model,
            df,
            score_method=score_method,
            max_length=max_length,
        )
        result_df = assign_labels(results, df)

        # Save results
        results_csv = self.results_dir / "search_results.csv"
        result_df.to_csv(results_csv, index=False)
        logging.info(f"Saved results to {results_csv}")

        # Compute and save metrics
        metrics = compute_metrics(result_df, exclude_neutral=True)
        metrics["score_method"] = score_method
        metrics["model_url"] = model_url
        metrics["num_domains"] = len(df)
        metrics["num_pairs"] = len(results)

        metrics_file = self.results_dir / "roc_pr_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logging.info(f"PR AUC: {metrics['pr_auc']:.4f}")
