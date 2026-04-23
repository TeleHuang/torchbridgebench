import torch
import torch.nn.functional as F

LAYER = "operator"

def test_add(adapter):
    a = torch.tensor([1.0, 2.0, 3.0], device=adapter.device)
    b = torch.tensor([4.0, 5.0, 6.0], device=adapter.device)
    # The adapter can intercept or execute
    c = adapter.run_operator(torch.add, a, b)
    expected = torch.tensor([5.0, 7.0, 9.0], device=adapter.device)
    return torch.allclose(c, expected)

def test_matmul(adapter):
    a = torch.randn(2, 3, device=adapter.device)
    b = torch.randn(3, 4, device=adapter.device)
    c = adapter.run_operator(torch.matmul, a, b)
    return c.shape == (2, 4)

def test_relu(adapter):
    a = torch.tensor([-1.0, 0.0, 1.0], device=adapter.device)
    c = adapter.run_operator(F.relu, a)
    expected = torch.tensor([0.0, 0.0, 1.0], device=adapter.device)
    return torch.allclose(c, expected)
