from .base import BaseAdapter
import torch

class TorchAdapter(BaseAdapter):
    @property
    def name(self):
        return "torch"

    def setup(self):
        pass

    def teardown(self):
        pass
