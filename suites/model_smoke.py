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

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TinyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.emb(x)
        x, (h, c) = self.lstm(x)
        # Take the output from the last sequence step
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def test_tiny_resnet(adapter):
    model = TinyResNet().to(adapter.device)
    x = torch.randn(2, 3, 32, 32, device=adapter.device)
    out = adapter.run_model(model, x)
    return out.shape == (2, 10)

def test_tiny_mlp(adapter):
    model = TinyMLP().to(adapter.device)
    x = torch.randn(4, 10, device=adapter.device)
    out = adapter.run_model(model, x)
    return out.shape == (4, 2)

def test_tiny_lstm(adapter):
    model = TinyLSTM().to(adapter.device)
    x = torch.randint(0, 100, (4, 10), device=adapter.device)
    out = adapter.run_model(model, x)
    return out.shape == (4, 2)
