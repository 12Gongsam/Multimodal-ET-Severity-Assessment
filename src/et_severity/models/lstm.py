"""Auto-extracted definitions from notebooks for modular use."""

from ..common_imports import *

class LSTM(nn.Module):
    def __init__(self, in_c, hidden_size, num_layers, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_c,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        h_n, c_n = self.lstm(x)
        return h_n

    def encode(self, x):
        return self.forward(x)
