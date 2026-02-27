# EBA benchmark test environment

[EBA](https://github.com/DeepFoldProtein/EBA) (Embedding-based alignment) benchmark environment.
Uses PLM (ProtT5/ESM1b) + DTW alignment.

**Before running:** Initialize the submodule and sync this environment, then run all commands **from the project root**.

---

## Installation (one-time)

```bash
git submodule update --init third_party/eba
uv sync --directory .uv/eba
```

---

## Commands

### Run eba_prott5 only (malidup)

```bash
.uv/eba/.venv/bin/python -m benchmark run --tests malidup --models eba_prott5 --update
```

### Run eba_prott5 on all 4 tests (MALIDUP, MALISAM, SABmark-sup, SABmark-twi) + plot

```bash
# 1) Run all four benchmarks
.uv/eba/.venv/bin/python -m benchmark run --tests malidup malisam sabmark-sup sabmark-twi --models eba_prott5 --update

# 2) Generate plots (metrics_box, barplot, etc.)
.uv/eba/.venv/bin/python -m benchmark plot --tests malidup malisam sabmark-sup sabmark-twi
```

- **run**: Results are saved to `out/results/<test>/eba_prott5/results.jsonl`.
- **plot**: Plots are generated under `out/plots/` for each test.
- `--update`: Recompute and overwrite existing results.

### One-liner (all 4 tests run + plot)

```bash
.uv/eba/.venv/bin/python -m benchmark run --tests malidup malisam sabmark-sup sabmark-twi --models eba_prott5 --update && .uv/eba/.venv/bin/python -m benchmark plot --tests malidup malisam sabmark-sup sabmark-twi
```

---

If installation fails, EBA dependencies (fair-esm, numba, etc.) may conflict with the main environment. It is recommended to use this directory (`.uv/eba`) only.
