import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_layer(norm_fn, out_channels, group_size=8):
    norm_fns = {
        'group': lambda: nn.GroupNorm(num_groups=out_channels // group_size, num_channels=out_channels),
        'instance': lambda: nn.InstanceNorm2d(out_channels, affine=False),
        'batch': lambda: nn.BatchNorm2d(out_channels),
        'none': lambda: nn.Sequential()}
    return norm_fns[norm_fn]()


class ResidualBlock(nn.Module):
    relu = nn.ReLU(inplace=True)

    def __init__(self, in_channels, out_channels, norm_fn='batch', stride=1):
        super().__init__()

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, stride=stride),
            norm_layer(norm_fn, out_channels))

        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1, stride=stride),
            norm_layer(norm_fn, out_channels),
            self.relu,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False, padding=1),
            norm_layer(norm_fn, out_channels),
            self.relu)

    def forward(self, x):
        return self.relu(self.skip_connection(x) + self.residual_branch(x))


class FPNBlock(nn.Module):
    relu = nn.ReLU(inplace=True)

    def __init__(self, dim, norm_fn='batch'):
        super().__init__()

        self.extract = nn.Sequential(
            ResidualBlock(dim, dim, norm_fn),
            ResidualBlock(dim, dim, norm_fn))
        self.downscale = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, bias=False, padding=1, stride=2),
            norm_layer(norm_fn, dim),
            self.relu)
        self.upscale = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, bias=False, padding=1, stride=1),
            norm_layer(norm_fn, dim),
            self.relu)

    def we_need_to_go_deeper(self, x):
        if x.shape[-1] < 64:
            return self.extract(x)

        y = x
        y = self.downscale(y)
        y = self.we_need_to_go_deeper(y)
        y = F.interpolate(y, scale_factor=2., mode='bilinear', align_corners=True)
        y = self.upscale(y)

        return self.relu(self.extract(x) + y)

    def forward(self, x):
        return self.we_need_to_go_deeper(x)


class LookMaNoFeatures(nn.Module):
    relu = nn.ReLU(inplace=True)

    def __init__(self, channels=64, blocks=7, norm_fn='batch', mixed_precision=False):
        super().__init__()

        fpn_blocks = [FPNBlock(channels, norm_fn) for _ in range(blocks)]
        self.layers = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, padding=1, bias=False),
            norm_layer(norm_fn, 16),
            self.relu,
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(norm_fn, 32),
            self.relu,
            nn.Conv2d(32, channels, kernel_size=3, padding=1, bias=False),
            norm_layer(norm_fn, channels),
            self.relu,
            ResidualBlock(channels, channels, norm_fn),
            FPNBlock(channels, norm_fn),
            ResidualBlock(channels, channels, norm_fn),
            ResidualBlock(channels, channels, norm_fn),
            FPNBlock(channels, norm_fn),
            ResidualBlock(channels, channels, norm_fn),
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(8, 1, kernel_size=3, padding=1))

    def forward(self, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        b, c, h, w = image1.shape

        alpha = torch.ones((b, 1, h, w), device=image1.device)
        y_pos = alpha.cumsum(2) - 1
        x_pos = alpha.cumsum(3) - 1

        volume = torch.cat((image1, image2, alpha, x_pos, y_pos), 1)  # 1 x 9 x H x W
        return self.layers(volume)
