from .base import BaseAdapter

class Torch4MSAdapter(BaseAdapter):
    @property
    def name(self):
        return "torch4ms"

    def setup(self):
        try:
            import torch4ms
            # Add specific torch4ms initialization if necessary
        except ImportError:
            print("Warning: torch4ms not found.")

    def teardown(self):
        pass
