import torch
import torch.nn as nn
import numpy as np
import random

class GaussianNoiseEditGuard(nn.Module):
    def __init__(self, opt, device):
        super(GaussianNoiseEditGuard, self).__init__()
        self.opt = opt
        # noisesigma is assumed to be in [0, 255] units
        # p is probability of applying noise
        self.p = opt['noise']['GaussianNoise'].get('p', 1.0)
        self.device = device

    def forward(self, y_forw, cover_img=None):
        # With probability p, add noise; otherwise return unchanged
        # if random.random() < self.p:
        # exactly your snippet:
        NL = self.opt['noisesigma'] / 255.0
        noise = np.random.normal(0, NL, size=y_forw.shape).astype(np.float32)
        torchnoise = torch.from_numpy(noise).to(self.device)
        y_forw = y_forw + torchnoise
        return y_forw