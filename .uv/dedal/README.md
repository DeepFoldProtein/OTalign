# DEDAL benchmark test environment

Isolated [DEDAL](https://github.com/DeepFoldProtein/dedal-fork) benchmark environment.
DEDAL uses TensorFlow / TensorFlow Hub, so it is kept separate from the main OTalign (PyTorch) environment.

**Important:** `benchmark` and `third_party/dedal` are relative to the project root, so run all commands **from the project root**.

This environment uses **Python 3.11** (`>=3.11,<3.12`). DEDAL is an older codebase that has `pkg_resources`/`distutils` issues on Python 3.12+, so it is pinned to 3.11. OTalign dependencies (scipy, etc.) require 3.11 or above.

**GPU/CPU selection**:
- Default is **GPU** (`use_gpu: true` in config). Fast, but some GPUs (RTX Ada, etc.) may hit `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
- If you get GPU errors, set `params.use_gpu: false` in `configs/benchmark_config.yaml` under the `dedal` model to run on CPU (stable but slower).

## Installation

From the project root (`OTalign/`):

```bash
uv sync --directory .uv/dedal
```

## Usage

### Run benchmark (DEDAL only, malidup)

```bash
.uv/dedal/.venv/bin/python -m benchmark run --tests malidup --models dedal
# or
.uv/dedal/run_benchmark.sh malidup
```

### Submodule

DEDAL source code lives in `third_party/dedal` (DeepFoldProtein/dedal-fork submodule).
To make changes, commit and push to the fork.

## Dependencies

- TensorFlow, tensorflow-hub (DEDAL model: https://tfhub.dev/google/dedal/3)
- OTalign (editable), datasets, gin-config, etc.
