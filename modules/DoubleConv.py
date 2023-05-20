from torch.nn import Module, Sequential, Conv2d, GroupNorm, GELU
from torch.nn.functional import gelu


class DoubleConv(Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = Sequential(
            Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            GroupNorm(1, mid_channels),
            GELU(),
            Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
