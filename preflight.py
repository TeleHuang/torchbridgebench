import sys
import subprocess
import platform
import importlib

def check_python_version():
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {platform.python_version()}")

def check_module(name):
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'present')}")
    except Exception as exc:
        print(f"{name}: unavailable ({exc})")

def check_mindspore():
    check_module("mindspore")

def check_torch_npu():
    try:
        import torch
        import torch_npu
        print(f"PyTorch Version: {torch.__version__}")
        try:
            print(f"Torch-NPU Version: {torch_npu.__version__}")
        except AttributeError:
            pass
        print(f"NPU Available: {torch.npu.is_available()}")
    except ImportError:
        print("Torch or Torch-NPU is not installed.")
    except AttributeError:
        pass

def check_npu_smi():
    try:
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
        if result.returncode == 0:
            print("npu-smi is available.")
            print(result.stdout.strip().split('\n')[0])
            print(result.stdout.strip().split('\n')[2])
        else:
            print("npu-smi returned an error.")
    except FileNotFoundError:
        print("npu-smi is not found in PATH.")

def run_preflight():
    print("=== Preflight Environment Check ===")
    check_python_version()
    check_npu_smi()
    check_mindspore()
    check_torch_npu()
    check_module("torch4ms")
    check_module("mindtorch_v2")
    check_module("mindnlp")
    check_module("torchvision")
    check_module("ultralytics")
    print("===================================")

if __name__ == "__main__":
    run_preflight()
