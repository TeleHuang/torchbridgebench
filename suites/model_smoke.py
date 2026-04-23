import torch
import torch.nn as nn

LAYER = "model"

class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 16 * 16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def test_tiny_resnet(adapter):
    model = TinyResNet()
    x = torch.randn(2, 3, 32, 32)
    out = adapter.run_model(model, x)
    return out.shape == (2, 10)
