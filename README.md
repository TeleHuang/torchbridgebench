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
python cli.py --backend <backend_name> --output <output.json>
```

Available backends: `torch`, `torch-npu`, `torch4ms`, `mindtorch`, `mindnlp_patch`.
