from torch.nn import Module, Sequential, MultiheadAttention, LayerNorm, Linear, GELU


class SelfAttention(Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = MultiheadAttention(channels, 4, batch_first=True)
        self.ln = LayerNorm([channels])
        self.ff_self = Sequential(
            LayerNorm([channels]),
            Linear(channels, channels),
            GELU(),
            Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
