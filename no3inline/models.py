import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out += x
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.hidden_layers = self._make_hidden_layers(64)
        
    def _make_hidden_layers(self):
        layers = []
        for _ in range(3):
            layers.append(ResnetBlock())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.hidden_layers(x)
        return x

    
class Generator(nn.Module):
    def __init__(self, N):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1 , 64, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=0)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(N * N, N * N)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
	x = F.pad(x, (-1, -1, -1, -1), value=-1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return x
