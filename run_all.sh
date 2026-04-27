#!/bin/bash

# Define paths to virtual environments
export VENV_DIR="/root/autodl-tmp/mindnlp/.venv-torch4ms"
export CLEAN_VENV_DIR="/root/autodl-tmp/mindnlp/.venv-torch4ms-clean"
export LEGACY_VENV_DIR="/root/autodl-tmp/mindnlp/.venv-torch4ms-legacy"

cd /root/autodl-tmp/torchbridgebench

# Run preflight checks
$VENV_DIR/bin/python preflight.py

echo "==============================="
echo "Environment matrix:"
echo "  torch / torch-npu / mindnlp_patch -> $VENV_DIR"
echo "  torch4ms (DEV benchmark path)     -> $LEGACY_VENV_DIR"
echo "  mindtorch                         -> $CLEAN_VENV_DIR"

echo "==============================="
echo "Running torch-npu benchmark..."
$VENV_DIR/bin/python cli.py --backend torch-npu --output report_torch_npu.json

echo "==============================="
echo "Running torch4ms benchmark..."
$LEGACY_VENV_DIR/bin/python cli.py --backend torch4ms --output report_torch4ms.json

echo "==============================="
echo "Running mindtorch benchmark..."
$CLEAN_VENV_DIR/bin/python cli.py --backend mindtorch --output report_mindtorch.json

echo "==============================="
echo "Running mindnlp_patch benchmark..."
$VENV_DIR/bin/python cli.py --backend mindnlp_patch --output report_mindnlp_patch.json

echo "==============================="
echo "All benchmarks completed!"
