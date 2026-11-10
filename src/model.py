import torch
import torch.nn as nn


def conv_block(in_c, out_c):                      # lighter: single conv
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
        nn.MaxPool2d(2))


def make_features(in_c):
    return nn.Sequential(
        conv_block(in_c, 32), conv_block(32, 64), conv_block(64, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten())


class SimpleCNN(nn.Module):
    """Baseline: spatial (RGB) branch only."""
    def __init__(self):
        super().__init__()
        self.features = make_features(3)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1))

    def forward(self, x):
        return self.classifier(self.features(x))


def fft_magnitude(x):
    gray = x.mean(dim=1, keepdim=True)
    f = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
    mag = torch.log1p(torch.abs(f))
    mean = mag.mean(dim=(2, 3), keepdim=True)
    std = mag.std(dim=(2, 3), keepdim=True) + 1e-6
    return (mag - mean) / std


class DualBranchCNN(nn.Module):
    """RGB branch + frequency-spectrum branch."""
    def __init__(self):
        super().__init__()
        self.spatial = make_features(3)
        self.freq = make_features(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 1))

    def forward(self, x):
        s = self.spatial(x)
        f = self.freq(fft_magnitude(x))
        return self.classifier(torch.cat([s, f], dim=1))


def build_model(name):
    return DualBranchCNN() if name == "dual" else SimpleCNN()