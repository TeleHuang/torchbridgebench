from .base import BaseAdapter

class MindTorchAdapter(BaseAdapter):
    @property
    def name(self):
        return "mindtorch"

    def setup(self):
        import sys
        import mindtorch_v2
        import mindtorch_v2.nn
        import mindtorch_v2.nn.functional
        import mindtorch_v2.optim
        import mindtorch_v2.utils
        import mindtorch_v2.utils.data
        
        for k, v in list(sys.modules.items()):
            if k.startswith('mindtorch_v2'):
                sys.modules[k.replace('mindtorch_v2', 'torch')] = v

    def teardown(self):
        pass
