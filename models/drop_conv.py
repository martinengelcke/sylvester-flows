import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class DropConv(nn.Module):

    def __init__(self, nin, nout, alpha_init=1.0, alpha_thresh=20):
        """
        Converting between alpha, Bernoulli, and KL:
        alpha = p/(1-p)
        p = alpha/(1+alpha)
        alpha = 0.01 --> KL = 2.94 --> p = 0.01
        alpha = 0.10 --> KL = 1.72 --> p = 0.1
        alpha = 1.00 --> KL = 0.43 --> p = 0.5
        alpha = 5.00 --> KL = 0.10 --> p = 0.83
        alpha = 10.0 --> KL = 0.05 --> p = 0.9
        alpha = 20.0 --> KL = 0.03 --> p = 0.95
        alpha = 50.0 --> KL = 0.010 -> p = 0.98
        alpha = 100. --> KL = 0.005 -> p = 0.99
        """
        super(DropConv, self).__init__()
        # --- Parameters ---
        # Note: not using biases
        # self.repr = f"{nin}, {nout}, init={alpha_init}, thresh={alpha_thresh}"
        # self.repr += f", params={nin*nout}"
        self.theta = nn.Parameter(torch.Tensor(nout, nin, 1, 1))
        self.log_alpha = nn.Parameter(torch.Tensor(nout, nin, 1, 1))
        # --- Parameters init ---
        # TODO(martin) not sure kaiming init appropriate
        nn.init.kaiming_uniform(self.theta)
        self.log_alpha.data.fill_(np.log(alpha_init))
        # --- Thresholding ---
        self.register_buffer('active', torch.ones_like(self.theta))
        self.log_thresh = np.log(alpha_thresh)

    def extra_repr(self):
        return self.repr

    def forward(self, x):
        # see: https://arxiv.org/pdf/1506.02557.pdf
        if self.log_thresh is not None: self.limit_alpha()
        # Set non-active weights to zero
        weights = self.active.detach() * self.theta
        gamma = F.conv2d(x, weights)
        delta = F.conv2d(x**2, torch.exp(self.log_alpha)*weights**2)
        zeta = Variable(torch.randn(gamma.size()))
        return gamma + torch.sqrt(delta+1e-10)*zeta

    def kld(self):
        # see: https://arxiv.org/pdf/1701.05369.pdf
        # TODO(martin) sample KL?
        if self.log_thresh is not None: self.limit_alpha()
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        C = -k1
        neg_kl_w = (k1*torch.sigmoid(k2+k3*self.log_alpha)
                    - 0.5*torch.log(1+torch.exp(-self.log_alpha)) + C)
        # Set KL of non-active weights to zero
        neg_kl_w = self.active.detach() * neg_kl_w
        return -neg_kl_w.sum()

    def limit_alpha(self):
        # Set weights to zero where threshold is exceeded
        self.active.data[(self.log_alpha.data > self.log_thresh)] = 0
        self.log_alpha.data = torch.clamp(
            self.log_alpha.data, -1e10, self.log_thresh+1)
