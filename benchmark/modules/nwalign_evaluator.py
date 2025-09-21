import subprocess
import time
from pathlib import Path

from benchmark.modules.base_evaluator import BaseEvaluator


class NwalignEvaluator(BaseEvaluator):
    """Evaluator for the NWalign tool."""

    def run(self):
        """
        Executes the NWalign benchmark by calling the existing run_nwalign_on_dataset.py script.
        """
        if not self.should_run():
            return

        self._log_start()
        start_time = time.time()

        # Construct the dataset identifier
        dataset_id = self.dataset_config["id"]
        if "config" in self.dataset_config:
            dataset_id += f",{self.dataset_config['config']}"
        if "split" in self.dataset_config:
            dataset_id += f",{self.dataset_config['split']}"

        script_path = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_nwalign_on_dataset.py"

        cmd = [
            "python",
            str(script_path),
            "--dataset",
            dataset_id,
            "--nwalign_bin",
            self.global_paths["nwalign_bin"],
            "--output",
            str(self.results_file),
            "--glocal",
            str(self.model_config["params"].get("glocal", 0)),
            "--workers",
            str(self.cli_args.workers),
        ]

        # Use Popen for better stream handling to avoid deadlocks
        cmd.insert(1, "-u")  # Add unbuffered flag for python

        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, bufsize=0) as process:
                # Use communicate to read stdout and stderr, preventing pipe buffer deadlocks
                stdout, stderr = process.communicate(timeout=3600)  # Generous timeout

                if stdout:
                    print(stdout.decode("utf-8", errors="ignore"))
                if stderr:
                    print("--- STDERR ---")
                    print(stderr.decode("utf-8", errors="ignore"))

                if process.returncode != 0:
                    # Manually create an error message similar to CalledProcessError
                    print(f"Error running NWalign for {self.model_config['label']}.")
                    print(f"  - Command: {' '.join(cmd)}")
                    print(f"  - Return Code: {process.returncode}")

        except Exception as e:
            print(f"An unexpected error occurred while running the NWalign subprocess for {self.model_config['label']}.")
            print(f"  - Command: {' '.join(cmd)}")
            print(f"  - Error: {e}")

        self._log_end(start_time)
