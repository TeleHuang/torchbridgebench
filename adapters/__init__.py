from .torch_adapter import TorchAdapter
from .torch_npu_adapter import TorchNPUAdapter
from .torch4ms_adapter import Torch4MSAdapter
from .mindtorch_adapter import MindTorchAdapter
from .mindnlp_patch_adapter import MindNLPPatchAdapter

ADAPTERS = {
    "torch": TorchAdapter,
    "torch-npu": TorchNPUAdapter,
    "torch4ms": Torch4MSAdapter,
    "mindtorch": MindTorchAdapter,
    "mindnlp_patch": MindNLPPatchAdapter,
}

def get_adapter(name: str):
    if name not in ADAPTERS:
        raise ValueError(f"Adapter {name} not found. Available adapters: {list(ADAPTERS.keys())}")
    return ADAPTERS[name]()
