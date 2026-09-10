"""Model architectures for journal-edition voter retraining.

LeNet-5 (modern): ReLU + cross-entropy (Han 2018 style).
ResNet-20: He et al. CIFAR variant, 20 layers.
WRN-28-10: Zagoruyko & Komodakis 2016.
MLP: Srivastava 2014 dropout MLP, 784-1024-1024-2048-10.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- LeNet-5 (modern ReLU variant) --------------------

class LeNet5(nn.Module):
    """Modern LeNet-5: 2 conv + 3 FC, ReLU, no dropout, no batchnorm.
    Input: 28x28x1 MNIST."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -------------------- ResNet-20 (He 2016 CIFAR variant) --------------------

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """He 2016 CIFAR ResNet-20: 3 stages × 3 basic blocks = 20 weight layers."""
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.linear = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# -------------------- WRN-28-10 (Zagoruyko 2016) --------------------

class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """WRN-28-10 (depth=28, widen_factor=10). 36.5M params on CIFAR-10."""
    def __init__(self, depth: int = 28, widen_factor: int = 10,
                 num_classes: int = 10, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        n_stages = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = n_stages[0]
        self.conv1 = nn.Conv2d(in_channels, n_stages[0], 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(n_stages[1], n, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(n_stages[2], n, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(n_stages[3], n, stride=2, dropout=dropout)
        self.bn = nn.BatchNorm2d(n_stages[3])
        self.linear = nn.Linear(n_stages[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride, dropout):
        layers = [WideBasicBlock(self.in_planes, planes, stride, dropout)]
        self.in_planes = planes
        for _ in range(num_blocks - 1):
            layers.append(WideBasicBlock(planes, planes, 1, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


# -------------------- MLP (Srivastava 2014: 784-1024-1024-2048-10) --------------------

class DropoutMLP(nn.Module):
    """Srivastava et al. 2014 Table 2 best non-conv MLP: 784-1024-1024-2048-10.
    Input dropout p=0.2, hidden dropout p=0.5, ReLU, cross-entropy.
    Max-norm constraint (c=2) is applied externally in common.apply_max_norm.
    """
    def __init__(self, num_classes: int = 10, input_dim: int = 784):
        super().__init__()
        self.drop_in = nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_dim, 1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 2048)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        x = self.drop_in(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.relu(self.fc3(x))
        x = self.drop3(x)
        return self.fc4(x)
