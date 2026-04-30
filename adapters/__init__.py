import importlib

ADAPTERS = {
    "torch": ("adapters.torch_adapter", "TorchAdapter"),
    "torch-npu": ("adapters.torch_npu_adapter", "TorchNPUAdapter"),
    "torch4ms": ("adapters.torch4ms_adapter", "Torch4MSAdapter"),
    "mindtorch": ("adapters.mindtorch_adapter", "MindTorchAdapter"),
    "mindnlp_patch": ("adapters.mindnlp_patch_adapter", "MindNLPPatchAdapter"),
}

def get_adapter(name: str):
    if name not in ADAPTERS:
        raise ValueError(f"Adapter {name} not found. Available adapters: {list(ADAPTERS.keys())}")
    module_name, class_name = ADAPTERS[name]
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return adapter_cls()
