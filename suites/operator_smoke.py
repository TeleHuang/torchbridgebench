import torch
import torch.nn.functional as F

LAYER = "operator"

def test_add(adapter):
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    # The adapter can intercept or execute
    c = adapter.run_operator(torch.add, a, b)
    expected = torch.tensor([5.0, 7.0, 9.0])
    return torch.allclose(c, expected)

def test_matmul(adapter):
    a = torch.randn(2, 3)
    b = torch.randn(3, 4)
    c = adapter.run_operator(torch.matmul, a, b)
    return c.shape == (2, 4)

def test_relu(adapter):
    a = torch.tensor([-1.0, 0.0, 1.0])
    c = adapter.run_operator(F.relu, a)
    expected = torch.tensor([0.0, 0.0, 1.0])
    return torch.allclose(c, expected)
