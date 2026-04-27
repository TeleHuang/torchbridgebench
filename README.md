# TorchBridgeBench

Unified Benchmark System for PyTorch to MindSpore/Ascend compatibility layers.

## Features

- **Hierarchical Benchmark**: Operator, Module, Model layers.
- **Multiple Backends**: Evaluates `torch`, `torch-npu`, `torch4ms`, `mindtorch`, and `mindnlp_patch`.
- **Unified Output**: Standardized JSON reports covering Compatibility, Correctness, Performance, and Usability.

## Architecture

- `adapters/`: Backend specific implementations to adapt the benchmark tests to different compatibility layers.
- `schema/`: JSON report schema definitions.
- `suites/`: Benchmark workloads separated by layer (operator, module, model).
- `cli.py`: Unified runner CLI.
- `preflight.py`: Environment check script.

## Usage

### Preflight Check

```bash
/root/autodl-tmp/mindnlp/.venv-torch4ms/bin/python preflight.py
```

### Run Benchmark

```bash
/root/autodl-tmp/mindnlp/.venv-torch4ms/bin/python cli.py --backend <backend_name> --output <output.json>
```

Available backends: `torch`, `torch-npu`, `torch4ms`, `mindtorch`, `mindnlp_patch`.

## Critical Runbook

### Approved Python Environments

- `/root/autodl-tmp/mindnlp/.venv-torch4ms`
- `/root/autodl-tmp/mindnlp/.venv-torch4ms-clean`
- `/root/autodl-tmp/mindnlp/.venv-torch4ms-legacy`

Do not use system Python for benchmark runs.

### Backend → Environment Matrix

- `torch`, `torch-npu`, `mindnlp_patch` → `/root/autodl-tmp/mindnlp/.venv-torch4ms`
- `torch4ms` → `/root/autodl-tmp/mindnlp/.venv-torch4ms-legacy`
- `mindtorch` → `/root/autodl-tmp/mindnlp/.venv-torch4ms-clean`

Why this mapping matters:

- `.venv-torch4ms` contains `torch_npu`, so it is required for `torch-npu` runs.
- The same venv must not be used for `torch4ms`, because `torch_npu` pre-registers the `privateuse1` backend as `npu`, which makes `torch4ms` import fail.
- On this server, `.venv-torch4ms-clean` carries `mindspore==2.3.0`, while `.venv-torch4ms-legacy` carries `mindspore==2.6.0`; current `torch4ms` DEV backward validation passes in the legacy env and is not the recommended benchmark path in the clean env.

### Experiment Dependency Notes

- `ascend-torch4ms/experiments/resnet_torchax_cifar/*` requires `torchvision`
- `ascend-torch4ms/experiments/mobilenet_torchax_cifar/*` requires `torchvision`
- `ascend-torch4ms/experiments/yolo_ultralytics_smoke/*` requires `ultralytics`

Current observed server state:

- `.venv-torch4ms-legacy` can import `torch4ms`, but does not provide `torchvision`
- `.venv-torch4ms` provides `torchvision`, but importing `torch4ms` there collides with `torch_npu`
- `ultralytics` is currently absent from the approved benchmark venvs

The repo regression wrapper keeps dependency and runtime limitations explicit:

- ResNet and MobileNet experiment smoke entries run the torch4ms path with a CPU MindSpore target, avoiding known Ascend BatchNorm / MaxPoolWithArgmaxV2 limits in manual CI.
- YOLO is marked `SKIP` when `ultralytics` is absent, so missing optional dependencies do not count as torch4ms regressions.
- `test_train_resnet_compare.py` remains a strict known-failing check for torch4ms gradient/update parity on larger torchvision models.

### Repo Regression Suite

`repo_training_regression` is now part of the benchmark and wraps high-value `ascend-torch4ms` scripts:

- `test_backward.py`
- `test_train_cnn.py`
- `test_train_transformer.py`
- `test_train_resnet_compare.py`
- ResNet / MobileNet / YOLO experiment smoke entries

This suite only runs for backend `torch4ms`; other backends show `N/A`.
Skipped repo-regression cases are excluded from pass-rate denominators and shown as `SKIP` in the Markdown report.

### Mandatory Pre-Run Checks

Run before each benchmark execution:

```bash
PY=/root/autodl-tmp/mindnlp/.venv-torch4ms/bin/python
$PY -c "import sys, importlib.util as u; print(sys.executable); print('mindspore', bool(u.find_spec('mindspore'))); print('torch_npu', bool(u.find_spec('torch_npu')))"
```

For `torch4ms`, use:

```bash
PY=/root/autodl-tmp/mindnlp/.venv-torch4ms-legacy/bin/python
$PY -c "import sys, importlib.util as u; print(sys.executable); print('torch4ms', bool(u.find_spec('torch4ms'))); print('torch_npu', bool(u.find_spec('torch_npu')))"
```

### Known Failure Signatures and Root Causes

- `No module named 'mindspore'`
  - Root cause: wrong interpreter selected.
  - Action: switch to one of the approved venvs above and rerun.
- `No module named 'torch_npu'`
  - Root cause: `torch-npu` benchmark was launched from `.venv-torch4ms-clean` or `.venv-torch4ms-legacy`.
  - Action: rerun `torch-npu` with `/root/autodl-tmp/mindnlp/.venv-torch4ms/bin/python`.
- `torch.register_privateuse1_backend() has already been set! Current backend: npu`
  - Root cause: `torch4ms` run executed in an environment/session where `torch_npu` backend is already registered.
  - Action: use `/root/autodl-tmp/mindnlp/.venv-torch4ms-legacy/bin/python` for `torch4ms` and avoid mixing `torch-npu` setup in the same runtime context.
- Sudden score collapse (for example backend drops to `0 / N`)
  - Root cause: environment/config regression rather than model/operator logic.
  - Action: stop and record the failure signature, root cause, and rerun steps before continuing.

### Post-Run Discipline

For critical failures, always document:

- failure signature,
- root cause,
- exact fix/recovery command,
- report file used for verification.
