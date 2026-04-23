from .base import BaseAdapter

class MindTorchAdapter(BaseAdapter):
    @property
    def name(self):
        return "mindtorch"

    def setup(self):
        try:
            import mindtorch
        except ImportError:
            print("Warning: mindtorch not found.")

    def teardown(self):
        pass
