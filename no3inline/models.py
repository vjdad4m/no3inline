import torch
import torch.nn as nn
import torch.nn.functional as F

class PaddedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pad_value=-1):
        super(PaddedConv2d, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.pad_value = pad_value
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,
        )
    
    def forward(self, x):
        x = F.pad(x, [self.padding] * 4, value=self.pad_value)
        return self.conv(x)

class MaskedLogitNetwork(nn.Module):
    def __init__(self, logit_model):
        super(MaskedLogitNetwork, self).__init__()

        # Initialize mask and apply it to the weights
        self.logit_model = logit_model

    def forward(self, x):
        unmasked_logits = self.logit_model(x)
        return (unmasked_logits - torch.exp(1000*x.flatten(1)))

class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv1 = PaddedConv2d(64, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = PaddedConv2d(64, 64, kernel_size=3)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out += x
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = PaddedConv2d(1, 64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.hidden_layers = self._make_hidden_layers()
        self.final = PaddedConv2d(64, 1, kernel_size=3)
        self.flatten = nn.Flatten()

    def _make_hidden_layers(self):
        layers = []
        for _ in range(3):
            layers.append(ResnetBlock())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.hidden_layers(x)
        x = self.final(x)
        x = self.flatten(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, N: int):
        super(Generator, self).__init__()
        self.conv1 = PaddedConv2d(1 , 64, kernel_size=3)
        self.conv2 = PaddedConv2d(64, 64, kernel_size=3)
        self.conv3 = PaddedConv2d(64, 1, kernel_size=3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(N * N, N * N)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear1(x)
        return x