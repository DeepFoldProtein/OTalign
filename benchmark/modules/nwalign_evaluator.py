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
        ]

        try:
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(process.stdout)
            if process.stderr:
                print("--- STDERR ---")
                print(process.stderr)

        except subprocess.CalledProcessError as e:
            print(f"Error running NWalign for {self.model_config['label']}.")
            print(f"  - Command: {' '.join(cmd)}")
            print(f"  - Return Code: {e.returncode}")
            print(f"  - STDOUT: {e.stdout}")
            print(f"  - STDERR: {e.stderr}")

        self._log_end(start_time)
