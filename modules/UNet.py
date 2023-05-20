from torch import arange, cat, sin, cos, float
from torch.nn import Module, Conv2d, Embedding
from modules import DoubleConv, ConvUp, ConvDown, SelfAttention


class UNet(Module):
    def __init__(self, image_size, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = ConvDown(64, 128)
        self.sa1 = SelfAttention(128, int(image_size/2))
        self.down2 = ConvDown(128, 256)
        self.sa2 = SelfAttention(256, int(image_size/4))
        self.down3 = ConvDown(256, 256)
        self.sa3 = SelfAttention(256, int(image_size/8))

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = ConvUp(512, 128)
        self.sa4 = SelfAttention(128, int(image_size/4))
        self.up2 = ConvUp(256, 64)
        self.sa5 = SelfAttention(64, int(image_size/2))
        self.up3 = ConvUp(128, 64)
        self.sa6 = SelfAttention(64, int(image_size))
        self.outc = Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
