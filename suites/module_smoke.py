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

class SimpleMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)

class SimpleAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)

class SimpleBatchNorm2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        return self.bn(x)

class SimpleLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(10)

    def forward(self, x):
        return self.ln(x)

class SimpleDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(x)

class SimpleEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 5)

    def forward(self, x):
        return self.emb(x)

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

def test_maxpool2d_module(adapter):
    module = SimpleMaxPool2d().to(adapter.device)
    x = torch.randn(1, 16, 30, 30, device=adapter.device)
    out = adapter.run_module(module, x)
    return out.shape == (1, 16, 15, 15)

def test_avgpool2d_module(adapter):
    module = SimpleAvgPool2d().to(adapter.device)
    x = torch.randn(1, 16, 30, 30, device=adapter.device)
    out = adapter.run_module(module, x)
    return out.shape == (1, 16, 15, 15)

def test_batchnorm2d_module(adapter):
    module = SimpleBatchNorm2d().to(adapter.device)
    x = torch.randn(1, 16, 30, 30, device=adapter.device)
    out = adapter.run_module(module, x)
    return out.shape == (1, 16, 30, 30)

def test_layernorm_module(adapter):
    module = SimpleLayerNorm().to(adapter.device)
    x = torch.randn(2, 10, device=adapter.device)
    out = adapter.run_module(module, x)
    return out.shape == (2, 10)

def test_dropout_module(adapter):
    module = SimpleDropout().to(adapter.device)
    x = torch.randn(2, 10, device=adapter.device)
    out = adapter.run_module(module, x)
    return out.shape == (2, 10)

def test_embedding_module(adapter):
    module = SimpleEmbedding().to(adapter.device)
    x = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=torch.long, device=adapter.device)
    out = adapter.run_module(module, x)
    return out.shape == (2, 4, 5)
