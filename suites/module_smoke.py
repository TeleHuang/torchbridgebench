import torch
import torch.nn as nn

LAYER = "module"

class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.conv(x)

def test_linear_module(adapter):
    module = SimpleLinear().to(adapter.device)
    x = torch.randn(2, 10, device=adapter.device)
    out = adapter.run_module(module, x)
    return out.shape == (2, 5)

def test_conv_module(adapter):
    module = SimpleConv().to(adapter.device)
    x = torch.randn(1, 3, 32, 32, device=adapter.device)
    out = adapter.run_module(module, x)
    return out.shape == (1, 16, 30, 30)
