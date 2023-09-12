import torch
import torch.nn as nn
import torch.nn.functional as F
from conv import Conv
from residual_block import ResidualBlock
import config as cfg


# Conversion from observation to inner abstract state
class Representation(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.layer0 = Conv(self.input_shape[0], cfg.num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(cfg.num_filters) for _ in range(cfg.num_blocks)])

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        return h

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            rp = self(torch.from_numpy(x).unsqueeze(0))
        return rp.cpu().numpy()[0]
