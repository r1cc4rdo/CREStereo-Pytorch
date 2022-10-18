import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import DisparityHead, ConvGRU
from .extractor import BasicEncoder


class BilinearForm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.a = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        self.b = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=False)
        self.cd = nn.Conv2d(dim, dim, kernel_size=1, padding=0, bias=True)

    def forward(self, x, y):
        return self.a(x) + self.b(y) + self.cd(x * y)  # ax + by + cxy + d


class SweepStereo(nn.Module):
    def __init__(self, max_disp=192, mixed_precision=False, test_mode=False):
        super().__init__()

        self.max_disp = max_disp
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode

        self.feature_dim = 32
        self.hidden_dim = 32
        self.dropout = 0

        self.fnet = BasicEncoder(output_dim=self.feature_dim, norm_fn='instance', dropout=self.dropout)
        self.cnet = BasicEncoder(output_dim=self.hidden_dim, norm_fn='batch', dropout=self.dropout)
        self.bform = BilinearForm(dim=self.feature_dim)
        self.gru = ConvGRU(self.hidden_dim, self.feature_dim + self.hidden_dim)
        self.disp = DisparityHead(self.hidden_dim, hidden_dim=64, output_dim=1)
        # self.mask = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, (4 ** 2) * 9, 1, padding=0))

    # def convex_upsample(self, flow, mask, rate=4):
    #     N, _, H, W = flow.shape
    #
    #     mask = mask.view(N, 1, 9, rate, rate, H, W)
    #     mask = torch.softmax(mask, dim=2)
    #
    #     up_flow = F.unfold(rate * flow, (3, 3), padding=1)
    #     up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
    #
    #     up_flow = torch.sum(mask * up_flow, dim=2)
    #     up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    #     return up_flow.reshape(N, 2, rate * H, rate * W)

    def forward(self, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        fmap1, fmap2 = self.fnet([image1, image2])
        hidden = torch.tanh(self.cnet(image1))

        for disp in range(self.max_disp // 4):

            fmap1_overlap = fmap1[:, disp:]
            fmap2_overlap = fmap2[:, :-disp or None]
            features = self.bform(fmap1_overlap, fmap2_overlap)

            hidden_overlap = hidden[:, disp:]  # [TODO] or no overlap?
            gru_input = torch.cat((hidden_overlap, features), 1)
            hidden_overlap = self.gru(hidden_overlap, gru_input)  # [TODO] or just convolution?
            hidden[:, disp:] = hidden_overlap

        disparity = -self.disp(hidden)
        # up_mask = self.mask(hidden)
        # disparity = -self.convex_upsample(disp, up_mask, rate=4)  # [TODO] or no upsampling?
        return disparity
