import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER = "autograd"

def test_linear_backward(adapter):
    model = nn.Linear(10, 5).to(adapter.device)
    x = torch.randn(2, 10, device=adapter.device, requires_grad=True)
    
    # Forward
    out = adapter.run_module(model, x)
    
    # Compute pseudo-loss
    loss = out.sum()
    
    # Backward
    loss.backward()
    
    # Check if gradients are populated
    return x.grad is not None and model.weight.grad is not None and model.bias.grad is not None

def test_conv2d_backward(adapter):
    model = nn.Conv2d(3, 16, 3).to(adapter.device)
    x = torch.randn(1, 3, 32, 32, device=adapter.device, requires_grad=True)
    
    out = adapter.run_module(model, x)
    loss = out.mean()
    loss.backward()
    
    return x.grad is not None and model.weight.grad is not None

def test_operator_backward(adapter):
    a = torch.tensor([1.0, 2.0, 3.0], device=adapter.device, requires_grad=True)
    b = torch.tensor([4.0, 5.0, 6.0], device=adapter.device, requires_grad=True)
    
    # (a * b) + a
    c = adapter.run_operator(torch.mul, a, b)
    d = adapter.run_operator(torch.add, c, a)
    
    loss = d.sum()
    loss.backward()
    
    # dc/da = b + 1
    # dc/db = a
    expected_grad_a = torch.tensor([5.0, 6.0, 7.0], device=adapter.device)
    expected_grad_b = torch.tensor([1.0, 2.0, 3.0], device=adapter.device)
    
    return torch.allclose(a.grad, expected_grad_a) and torch.allclose(b.grad, expected_grad_b)
