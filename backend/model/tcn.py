"""Multi-branch causal TCN with mid-level fusion.

  per modality (emg / imu / hand): [features + mask] -> 2 causal dilated conv blocks (d=1,2) -> x mask
  fusion: concat(branches, masks) -> 1x1 -> residual causal TCN (k=3, d=1,2,4,8,16) -> 1x1 head -> logits
Receptive field = 1 + 2*(1+2) + 2*(1+2+4+8+16) = 69 frames = 1.38 s at 50 Hz.
Input: Z (B, D, L) standardised features, M (B, 3, L) presence masks. Output: logits (B, n_cls, L).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.data.features import LAYOUT


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class Block(nn.Module):
    """causal conv -> GELU -> dropout -> 1x1, with residual."""

    def __init__(self, cin, cout, k, d, drop):
        super().__init__()
        self.c1 = CausalConv(cin, cout, k, d)
        self.c2 = nn.Conv1d(cout, cout, 1)
        self.drop = nn.Dropout(drop)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.drop(F.gelu(self.c1(x)))
        return F.gelu(self.res(x) + self.c2(h))


class TCN(nn.Module):
    def __init__(self, n_cls: int, branch_ch: int = 32, fuse_ch: int = 64, kernel: int = 3,
                 dilations=(1, 2, 4, 8, 16), branch_dilations=(1, 2), dropout: float = 0.1):
        super().__init__()
        self.slices = list(LAYOUT.values())
        self.branches = nn.ModuleList()
        for sl in self.slices:
            cin = sl.stop - sl.start + 1
            layers, c = [], cin
            for d in branch_dilations:
                layers.append(Block(c, branch_ch, kernel, d, dropout))
                c = branch_ch
            self.branches.append(nn.Sequential(*layers))
        self.fuse_in = nn.Conv1d(branch_ch * len(self.slices) + len(self.slices), fuse_ch, 1)
        self.tcn = nn.Sequential(*[Block(fuse_ch, fuse_ch, kernel, d, dropout) for d in dilations])
        self.head = nn.Conv1d(fuse_ch, n_cls, 1)
        self.receptive_field = 1 + (kernel - 1) * (sum(branch_dilations) + sum(dilations))

    def forward(self, Z, M):
        outs = []
        for k, (sl, br) in enumerate(zip(self.slices, self.branches)):
            m = M[:, k:k + 1]
            outs.append(br(torch.cat([Z[:, sl], m], 1)) * m)
        h = self.fuse_in(torch.cat(outs + [M], 1))
        return self.head(self.tcn(h))


def build_model(n_cls: int, mcfg: dict) -> TCN:
    return TCN(n_cls, mcfg["branch_ch"], mcfg["fuse_ch"], mcfg["kernel"], tuple(mcfg["dilations"]),
               tuple(mcfg["branch_dilations"]), mcfg["dropout"])


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
