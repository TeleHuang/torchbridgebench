from .base import BaseAdapter

class MindTorchAdapter(BaseAdapter):
    @property
    def name(self):
        return "mindtorch"

    def setup(self):
        import mindtorch
        import sys
        sys.modules['torch'] = mindtorch

    def teardown(self):
        pass
