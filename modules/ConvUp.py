from torch import cat
from torch.nn import Module, Sequential, Upsample, SiLU, Linear
from modules import DoubleConv


class ConvUp(Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = Sequential(
            SiLU(),
            Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
