# pLM-BLAST benchmark test environment

Separate uv environment for running [pLM-BLAST](https://github.com/DeepFoldProtein/pLM-BLAST) benchmarks.
The submodule points to the **DeepFoldProtein/pLM-BLAST** fork; changes (e.g. `coords_to_match_pairs`) should be committed and pushed to that fork.
Installs both the main project (OTalign) and pLM-BLAST dependencies (mkl, numba, biopython, etc.).

**Important:** `benchmark` and `third_party/plmblast` are relative to the project root, so always run commands **from the project root**.

## Installation

From the project root (`OTalign/`):

```bash
uv sync --directory .uv/plmblast
```

(First run may take a while to download Python 3.12+ and packages. If `pip list` shows numba, mkl, biopython, etc., the installation is correct.)

## Usage

All commands should be run from the **project root**.

### Run benchmark (pLM-BLAST only)

This environment runs **pLM-BLAST (plmblast_prott5) only**. External executables like NWalign/HHalign are not used.

```bash
.uv/plmblast/.venv/bin/python -m benchmark run --tests malidup --models plmblast_prott5
```

Or use the helper script (automatically applies `--models plmblast_prott5`):

```bash
.uv/plmblast/run_benchmark.sh malidup
```

### Activate the virtual environment

```bash
source .uv/plmblast/.venv/bin/activate
# Make sure the project root is the current working directory
python -m benchmark run --tests malidup
```

### Visualize results (plots)

After running the benchmark, visualize malidup results (F1/recall/precision boxplot, F1 barplot):

```bash
.uv/plmblast/.venv/bin/python -m benchmark plot --tests malidup --plot-format svg
# or
.uv/plmblast/plot_benchmark.sh malidup
```

Output files: `out/plots/malidup/metrics_box.svg`, `out/plots/malidup/malidup_f1_bar.svg` (and corresponding .csv files).

### Quick pLM-BLAST sanity check

```bash
.uv/plmblast/.venv/bin/python -c "
import sys
sys.path.insert(0, 'third_party/plmblast')
import alntools as aln
e = aln.Extractor(enh=False, norm=False, bfactor='global')
print('pLM-BLAST OK')
"
```

## Pushing changes to the submodule (fork)

After modifying `third_party/plmblast`, push to the fork:

```bash
cd third_party/plmblast
git add -A && git commit -m "Describe changes"
git push origin main
```

Then update the submodule reference in the **OTalign root**:

```bash
git add third_party/plmblast .gitmodules
git commit -m "Update plmblast submodule"
```

To pull upstream (labstructbioinf) updates: `cd third_party/plmblast && git fetch upstream && git merge upstream/main`

## Dependencies

- `pyproject.toml`: OTalign (editable) + pLM-BLAST packages (biopython, mkl, numba, etc.)
- Python: 3.12+ (matches OTalign requirements)
