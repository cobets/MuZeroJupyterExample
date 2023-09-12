import torch
import torch.nn as nn
from conv import Conv
from residual_block import ResidualBlock
import config as cfg


# Abstract state transition
class Dynamics(nn.Module):
    def __init__(self, rp_shape, act_shape):
        super().__init__()
        self.rp_shape = rp_shape
        self.layer0 = Conv(rp_shape[0] + act_shape[0], cfg.num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(cfg.num_filters) for _ in range(cfg.num_blocks)])

    def forward(self, rp, a):
        h = torch.cat([rp, a], dim=1)
        h = self.layer0(h)
        for block in self.blocks:
            h = block(h)
        return h

    def inference(self, rp, a):
        self.eval()
        with torch.no_grad():
            rp = self(torch.from_numpy(rp).unsqueeze(0), torch.from_numpy(a).unsqueeze(0))
        return rp.cpu().numpy()[0]
    