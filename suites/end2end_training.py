import torch
import torch.nn as nn
import torch.optim as optim
import time

LAYER = "end2end"

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Fallback to AvgPool2d if MaxPool2d is problematic on some backends
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_synthetic_mnist_data(num_samples=128, batch_size=32, device='cpu'):
    """Generate synthetic MNIST-like data to avoid download issues in restricted environments."""
    torch.manual_seed(42)
    inputs = torch.randn(num_samples, 1, 28, 28, device=device)
    targets = torch.randint(0, 10, (num_samples,), device=device)
    
    batches = []
    for i in range(0, num_samples, batch_size):
        batches.append((inputs[i:i+batch_size], targets[i:i+batch_size]))
    return batches

def test_mnist_training_smoke(adapter):
    """
    A minimal end-to-end training loop for an MNIST-like CNN.
    Measures compatibility, basic correctness (loss goes down), and records time.
    """
    device = adapter.device
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    batches = get_synthetic_mnist_data(num_samples=64, batch_size=16, device=device)

    model.train()
    start_time = time.time()
    
    initial_loss = None
    final_loss = None

    # Run for 3 epochs
    for epoch in range(3):
        epoch_loss = 0.0
        for inputs, targets in batches:
            optimizer.zero_grad()
            outputs = adapter.run_module(model, inputs)
            
            # Allow adapter to optionally handle loss computation/backward if needed (like torch4ms requires)
            if hasattr(adapter, "run_backward"):
                loss = adapter.run_backward(model, criterion, outputs, targets, optimizer)
            else:
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()
            
        if initial_loss is None:
            initial_loss = epoch_loss
        final_loss = epoch_loss

    end_time = time.time()
    
    # Store execution time for the report
    adapter.last_performance_ms = (end_time - start_time) * 1000.0

    # Basic correctness check: Loss should generally decrease or at least be computable
    return initial_loss is not None and final_loss is not None and final_loss <= initial_loss
