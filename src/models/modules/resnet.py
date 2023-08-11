import torch.nn as nn

class BasicResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, zero_init=False):
        super(BasicResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(32, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.GroupNorm(32, out_channels)
            )
        if zero_init:
            self.conv2.weight.data.zero_()
            if self.conv2.bias is not None:
                self.conv2.bias.data.zero_()

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        identity = self.shortcut(identity)

        out += identity

        return out
