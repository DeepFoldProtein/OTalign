import subprocess
import time
from pathlib import Path

from benchmark.modules.base_evaluator import BaseEvaluator


class OtalignEvaluator(BaseEvaluator):
    """Evaluator for the OTalign tool."""

    def _get_model_name_for_dir(self) -> str:
        """
        Returns the directory name for the model, using the model key from the config.
        """
        return self.model_config["_key"]

    def run(self):
        """
        Executes the OTalign benchmark by calling the existing run_otalign_on_dataset.py script.
        """
        if not self.should_run():
            return

        self._log_start()
        start_time = time.time()

        # Construct the dataset identifier for the script
        dataset_id = self.dataset_config["id"]
        if "config" in self.dataset_config:
            dataset_id += f",{self.dataset_config['config']}"
        if "split" in self.dataset_config:
            dataset_id += f",{self.dataset_config['split']}"

        # Define a base cache directory for this model and dataset
        base_cache_dir = Path(self.global_paths["cache_dir"]) / self.dataset_config["name"] / self.model_name

        # Check if the cache exists, if not, build it
        if not base_cache_dir.exists() or not any(base_cache_dir.iterdir()):
            print(f"Cache not found for {self.model_name} on {self.dataset_config['name']}. Building cache...")
            self._build_cache(base_cache_dir, dataset_id)

        # The build script creates a subdirectory; we need to find it.
        try:
            actual_cache_dir = next(d for d in base_cache_dir.iterdir() if d.is_dir())
            print(f"Found actual cache directory: {actual_cache_dir}")
        except StopIteration:
            print(f"Error: Cache directory '{base_cache_dir}' is empty or contains no subdirectories after build attempt.")
            raise FileNotFoundError(f"Cache not built correctly in {base_cache_dir}")

        # Path to the script to be executed
        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_otalign_on_dataset.py"

        # Build the command-line arguments
        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

        cmd = [
            "python",
            str(script_path),
            "--dataset",
            dataset_id,
            "--output",
            str(self.results_file),
            "--cache_dir",
            str(actual_cache_dir),
            "--no-tqdm",
            "--save_transport_plan_dir",
            str(self.transport_plan_dir),
            "--device",
            device,
        ]

        # Handle trained checkpoints vs. base models
        if "checkpoint_path" in self.model_config:
            cmd.extend(
                [
                    "--model",
                    self.model_config["checkpoint_path"],
                    "--base_model_for_checkpoint",
                    self.model_config["base_model"],
                ]
            )
        else:
            cmd.extend(
                [
                    "--model",
                    self.model_config["plm"],
                ]
            )

        # Add OT-specific parameters from config
        cmd.extend(
            [
                "--reg",
                str(self.model_config["params"].get("epsilon", 0.1)),
                "--lambda1",
                str(self.model_config["params"].get("tau", 1.0)),  # Pass tau directly as lambda1
                "--lambda2",
                str(self.model_config["params"].get("tau", 1.0)),  # Pass tau directly as lambda2
            ]
        )

        # Add align_batch_size if specified
        if "align_batch_size" in self.model_config["params"]:
            cmd.extend(["--align_batch_size", str(self.model_config["params"]["align_batch_size"])])

        try:
            # Execute the script
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(process.stdout)
            if process.stderr:
                print("--- STDERR ---")
                print(process.stderr)

        except subprocess.CalledProcessError as e:
            print(f"Error running OTalign for {self.model_config['label']}.")
            print(f"  - Command: {' '.join(cmd)}")
            print(f"  - Return Code: {e.returncode}")
            print(f"  - STDOUT: {e.stdout}")
            print(f"  - STDERR: {e.stderr}")
            # Optionally, re-raise or handle the error
            # raise e

        self._log_end(start_time)

    def _build_cache(self, cache_dir, dataset_id):
        """Builds the embedding cache using scripts/build_cache.py."""
        build_script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "build_cache.py"
        device = self.cli_args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        cmd = [
            "python",
            str(build_script_path),
            "--dataset",
            dataset_id,
            "--output_root",
            str(cache_dir),
            "--no-tqdm",
            "--device",
            device,
        ]

        if "checkpoint_path" in self.model_config:
            cmd.extend(
                [
                    "--model",
                    self.model_config["checkpoint_path"],
                    "--base_model_for_checkpoint",
                    self.model_config["base_model"],
                ]
            )
        else:
            cmd.extend(
                [
                    "--model",
                    self.model_config["plm"],
                ]
            )

        try:
            print(f"  - Running cache build command: {' '.join(cmd)}")
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if process.stdout:
                print(process.stdout)
            if process.stderr:
                print("--- STDERR (Cache Build) ---")
                print(process.stderr)
            print("Cache build complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error building cache for {self.model_config['label']}.")
            print(f"  - Command: {' '.join(cmd)}")
            print(f"  - Return Code: {e.returncode}")
            print(f"  - STDOUT: {e.stdout}")
            print(f"  - STDERR: {e.stderr}")
            raise e
