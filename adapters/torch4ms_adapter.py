from .base import BaseAdapter

class Torch4MSAdapter(BaseAdapter):
    @property
    def name(self):
        return "torch4ms"

    def setup(self):
        import torch4ms
        self.env = torch4ms.default_env()
        self.env.__enter__()

    def teardown(self):
        if hasattr(self, 'env') and self.env:
            self.env.__exit__(None, None, None)
