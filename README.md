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
python preflight.py
```

### Run Benchmark

```bash
python cli.py --backend <backend_name> --output artifacts/reports/<output.json>
```

Available backends: `torch`, `torch-npu`, `torch4ms`, `mindtorch`, `mindnlp_patch`.

Filter flags can be combined for targeted runs:

```bash
# Run one suite.
python cli.py --backend torch4ms --suite module_smoke --output artifacts/reports/module_smoke.json

# Run one test by simple name.
python cli.py --backend torch4ms --suite module_smoke --test test_batchnorm2d_module --output artifacts/reports/batchnorm.json

# Run one test by qualified suite/test selector.
python cli.py --backend torch4ms --test module_smoke/test_avgpool2d_module --output artifacts/reports/avgpool.json

# Run one layer.
python cli.py --backend torch4ms --layer end2end --output artifacts/reports/end2end.json
```

`--suite`, `--test`, and `--layer` can be repeated or passed as comma-separated lists.

## Critical Runbook

### Current Torch4MS Baseline

The current active `torch4ms` source tree is:

```bash
/root/autodl-tmp/ascend-torch4ms-ms272-stable
```

Do not use `/root/autodl-tmp/ascend-torch4ms` for current benchmark conclusions.

Latest verified `torch4ms` results:

- MS2.8.0 + CANN8.5 NPU: `41 / 41` passed, `0` skipped.
- Report JSON: `artifacts/reports/report_torch4ms_ms280_cann85_npu_20260430_135800.json`
- Report Markdown: `artifacts/reports/benchmark_report_torch4ms_ms280_cann85_npu_20260430_135800.md`

### Approved Torch4MS Environments

MS2.8.0 clean CPU benchmark:

```bash
source /root/autodl-tmp/activate_torch4ms_ms280_clean.sh
```

MS2.8.0 + CANN8.5 Ascend NPU benchmark:

```bash
source /root/autodl-tmp/activate_torch4ms_ms280_cann85.sh
export TORCH4MS_REPO_ROOT=/root/autodl-tmp/ascend-torch4ms-ms272-stable
export TORCH4MS_DEVICE_TARGET=Ascend
export TORCH4MS_USE_MS_GRAPH_MODE=0
```

MS2.7.2 CPU baseline for strict backward and small training scripts:

```bash
source /root/autodl-tmp/activate_torch4ms_ms272_stable.sh
```

Do not use system Python for benchmark runs.

### Torch4MS NPU Full Run

Use this command for the current NPU acceptance run:

```bash
source /root/autodl-tmp/activate_torch4ms_ms280_cann85.sh
cd /root/autodl-tmp/torchbridgebench
TORCH4MS_REPO_ROOT=/root/autodl-tmp/ascend-torch4ms-ms272-stable \
TORCH4MS_DEVICE_TARGET=Ascend \
TORCH4MS_USE_MS_GRAPH_MODE=0 \
  python cli.py --backend torch4ms --output artifacts/reports/report_torch4ms_ms280_cann85_npu.json
```

Generate a Markdown report from a JSON output:

```bash
python report_generator.py \
  --input-glob artifacts/reports/report_torch4ms_ms280_cann85_npu.json \
  --output artifacts/reports/benchmark_report_torch4ms_ms280_cann85_npu.md
```

For a targeted NPU rerun:

```bash
TORCH4MS_REPO_ROOT=/root/autodl-tmp/ascend-torch4ms-ms272-stable \
TORCH4MS_DEVICE_TARGET=Ascend \
TORCH4MS_USE_MS_GRAPH_MODE=0 \
  python cli.py --backend torch4ms --test module_smoke/test_batchnorm2d_module --output artifacts/reports/batchnorm_npu.json
```

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

Run before each `torch4ms` benchmark execution:

```bash
python -c "import sys, mindspore, torch, torchvision, torch4ms; print(sys.executable); print('mindspore', mindspore.__version__); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('torch4ms', torch4ms.__file__)"
```

### Known Failure Signatures and Root Causes

- `No module named 'mindspore'`
  - Root cause: wrong interpreter selected.
  - Action: source the correct activation script above and rerun.
- `No module named 'torch_npu'`
  - Root cause: `torch-npu` benchmark was launched from an environment without `torch_npu`.
  - Action: switch to the NPU PyTorch environment intended for `torch-npu` runs.
- `torch.register_privateuse1_backend() has already been set! Current backend: npu`
  - Root cause: `torch4ms` run executed in an environment/session where `torch_npu` backend is already registered.
  - Action: use a clean process and do not mix `torch-npu` setup with `torch4ms` setup in the same runtime.
- `Launch kernel failed, name:Default/AvgPool3D-op0`
  - Root cause: Ascend `AvgPool2D` lowering through `AvgPool3D/AvgPool3DD` with unsupported float32 input.
  - Action: verify the `torch4ms` source includes the Ascend `avg_pool2d` float32-to-float16 fallback.
- Sudden score collapse (for example backend drops to `0 / N`)
  - Root cause: environment/config regression rather than model/operator logic.
  - Action: stop and record the failure signature, root cause, and rerun steps before continuing.

### Post-Run Discipline

Benchmark output files must stay out of the repository root. Write JSON and
Markdown reports under `artifacts/reports/`; write backend runtime dumps under
`artifacts/` or another ignored runtime directory. Only curated example reports
intended for documentation should be copied into a tracked docs/example path.

For critical failures, always document:

- failure signature,
- root cause,
- exact fix/recovery command,
- report file used for verification.
