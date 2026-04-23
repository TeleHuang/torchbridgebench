from .base import BaseAdapter

class TorchNPUAdapter(BaseAdapter):
    @property
    def name(self):
        return "torch-npu"

    @property
    def device(self):
        return "npu:0"

    def setup(self):
        import torch
        import torch_npu
        torch.npu.set_device(0)

    def teardown(self):
        pass
