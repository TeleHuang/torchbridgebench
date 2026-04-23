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

def test_sub(adapter):
    a = torch.tensor([5.0, 7.0, 9.0], device=adapter.device)
    b = torch.tensor([4.0, 5.0, 6.0], device=adapter.device)
    c = adapter.run_operator(torch.sub, a, b)
    expected = torch.tensor([1.0, 2.0, 3.0], device=adapter.device)
    return torch.allclose(c, expected)

def test_mul(adapter):
    a = torch.tensor([1.0, 2.0, 3.0], device=adapter.device)
    b = torch.tensor([2.0, 3.0, 4.0], device=adapter.device)
    c = adapter.run_operator(torch.mul, a, b)
    expected = torch.tensor([2.0, 6.0, 12.0], device=adapter.device)
    return torch.allclose(c, expected)

def test_div(adapter):
    a = torch.tensor([4.0, 6.0, 8.0], device=adapter.device)
    b = torch.tensor([2.0, 2.0, 4.0], device=adapter.device)
    c = adapter.run_operator(torch.div, a, b)
    expected = torch.tensor([2.0, 3.0, 2.0], device=adapter.device)
    return torch.allclose(c, expected)

def test_sum(adapter):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=adapter.device)
    c = adapter.run_operator(torch.sum, a, dim=0)
    expected = torch.tensor([4.0, 6.0], device=adapter.device)
    return torch.allclose(c, expected)

def test_mean(adapter):
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=adapter.device)
    c = adapter.run_operator(torch.mean, a, dim=1)
    expected = torch.tensor([1.5, 3.5], device=adapter.device)
    return torch.allclose(c, expected)

def test_max(adapter):
    a = torch.tensor([1.0, 5.0, 3.0], device=adapter.device)
    c = adapter.run_operator(torch.max, a)
    expected = torch.tensor(5.0, device=adapter.device)
    return torch.allclose(c, expected)

def test_cat(adapter):
    a = torch.tensor([[1.0, 2.0]], device=adapter.device)
    b = torch.tensor([[3.0, 4.0]], device=adapter.device)
    c = adapter.run_operator(torch.cat, (a, b), dim=0)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=adapter.device)
    return torch.allclose(c, expected)

def test_stack(adapter):
    a = torch.tensor([1.0, 2.0], device=adapter.device)
    b = torch.tensor([3.0, 4.0], device=adapter.device)
    c = adapter.run_operator(torch.stack, (a, b), dim=1)
    expected = torch.tensor([[1.0, 3.0], [2.0, 4.0]], device=adapter.device)
    return torch.allclose(c, expected)

def test_reshape(adapter):
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=adapter.device)
    c = adapter.run_operator(torch.reshape, a, (3, 2))
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=adapter.device)
    return torch.allclose(c, expected)

def test_transpose(adapter):
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=adapter.device)
    c = adapter.run_operator(torch.transpose, a, 0, 1)
    expected = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], device=adapter.device)
    return torch.allclose(c, expected)

def test_squeeze(adapter):
    a = torch.tensor([[[1.0, 2.0]]], device=adapter.device)
    c = adapter.run_operator(torch.squeeze, a, dim=0)
    expected = torch.tensor([[1.0, 2.0]], device=adapter.device)
    return torch.allclose(c, expected)

def test_unsqueeze(adapter):
    a = torch.tensor([1.0, 2.0], device=adapter.device)
    c = adapter.run_operator(torch.unsqueeze, a, dim=0)
    expected = torch.tensor([[1.0, 2.0]], device=adapter.device)
    return torch.allclose(c, expected)

def test_softmax(adapter):
    a = torch.tensor([[0.0, 1.0]], device=adapter.device)
    c = adapter.run_operator(F.softmax, a, dim=-1)
    # math.exp(0)/(math.exp(0)+math.exp(1)) = 1 / (1 + 2.71828) ≈ 0.2689
    # math.exp(1)/(math.exp(0)+math.exp(1)) ≈ 0.7311
    expected = torch.tensor([[0.26894142, 0.73105858]], device=adapter.device)
    return torch.allclose(c, expected, atol=1e-4)

def test_sigmoid(adapter):
    a = torch.tensor([0.0], device=adapter.device)
    c = adapter.run_operator(torch.sigmoid, a)
    expected = torch.tensor([0.5], device=adapter.device)
    return torch.allclose(c, expected)

def test_tanh(adapter):
    a = torch.tensor([0.0], device=adapter.device)
    c = adapter.run_operator(torch.tanh, a)
    expected = torch.tensor([0.0], device=adapter.device)
    return torch.allclose(c, expected)
