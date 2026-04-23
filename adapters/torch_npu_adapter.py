from .base import BaseAdapter

class TorchNPUAdapter(BaseAdapter):
    @property
    def name(self):
        return "torch-npu"

    def setup(self):
        try:
            import torch
            import torch_npu
            # Device will be handled by the workload if this adapter is chosen
        except ImportError:
            print("Warning: torch_npu not found.")

    def teardown(self):
        pass
