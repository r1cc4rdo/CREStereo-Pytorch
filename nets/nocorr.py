import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import AGCL

from .attention import PositionEncodingSine, LocalFeatureTransformer

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class NoCorr(nn.Module):
    def __init__(self, mixed_precision=False):
        super(NoCorr, self).__init__()
        self.mixed_precision = mixed_precision

        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)  # dummy parameters to appease optimizer

    def forward(self, image1, image2):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        b, c, h, w = image1.shape
        alpha = torch.ones((b, 1, h, w), device=image1.device)  # useful only if setting oob value (e.g. to 0)

        pos_encoding = []
        y_pos = alpha.cumsum(2) - 1
        x_pos = alpha.cumsum(3) - 1
        for div in torch.pow(2, torch.arange(0, 9)):  # 2**9 == 512
            pos_encoding.append((torch.sin(x_pos * torch.pi / div) + torch.sin(y_pos * torch.pi / div)) / 2)
            pos_encoding.append((torch.cos(x_pos * torch.pi / div) + torch.cos(y_pos * torch.pi / div)) / 2)
        pos_encoding = pos_encoding[1:]  # drop first positional encoding channel, which is 0

        # from matplotlib import pyplot as plot
        # for index, pe in enumerate(pos_encoding):
        #     plot.imshow(pe[0][0][:50, :50].cpu().numpy(), vmin=-1, vmax=1)
        #     plot.title(f'index {index} -- div4 {index // 4}')
        #     plot.show()

        u, n, d = 5, 3, 3  # number of layers for random Uniform, Normal, Disparity
        randu = torch.rand((b, 5, u, w), device=image1.device)  # no straightforward way to convert uniform to normal
        randn = torch.randn((b, 3, n, w), device=image1.device)  # therefore we add both
        randd = torch.clamp(torch.randint(w // 3, (b, d, h, w), device=image1.device) + x_pos, 0, w-1)

        volume = torch.cat((image1, image2, alpha, x_pos, randu, randn, randd, *pos_encoding), 1)  # 1 x 32 x H x W

        # x = volume
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))
        # x[:, xxxxx, ...] = self.glob_avgpool(x[:, xxxxx, ...])

        # x, y, z = torch.ones((2, 3)), torch.ones((2, 3), requires_grad=True), torch.ones((2, 3), requires_grad=True)
        # val, idx = torch.max(x * (2 * y) * (3 * z), 1)
        # val.backward(torch.tensor([1, 1]))

        torch.nn.functional.softmax(a, dim=-1)
        a / a.max(keepdim=True, dim=-1)[0]

        # 33 == 3 + 3 + 1 + 1 + 5 + 3 + (9 * 2 - 1)
        #      RGB RGB  A   C   R   Rn   pos_enc



        return volume

        # [TODO] check size/nature of volume gradients

        # conv 32 > 64
        # bnorm
        # relu
        #
        # loop 64 > 32 > 64 --- math.floor(math.log2(min(w, h))) - 3/4/5
        #
        # conv 64 > 16
        # bnorm
        # relu
        #
        # conv 16 > 1

    def forward_at_some_point(self, image1, image2):

        fmap1 = fmap2 = hdim = flow_init = iters =  None
        with autocast(enabled=self.mixed_precision):

            # 1/4 -> 1/8
            # feature
            fmap1_dw8 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2_dw8 = F.avg_pool2d(fmap2, 2, stride=2)

            # offset
            offset_dw8 = self.conv_offset_8(fmap1_dw8)
            offset_dw8 = self.range_8 * (torch.sigmoid(offset_dw8) - 0.5) * 2.0

            # context
            net, inp = torch.split(fmap1, [hdim, hdim], dim=1)
            net = torch.tanh(net)
            inp = F.relu(inp)
            net_dw8 = F.avg_pool2d(net, 2, stride=2)
            inp_dw8 = F.avg_pool2d(inp, 2, stride=2)

            # 1/4 -> 1/16
            # feature
            fmap1_dw16 = F.avg_pool2d(fmap1, 4, stride=4)
            fmap2_dw16 = F.avg_pool2d(fmap2, 4, stride=4)
            offset_dw16 = self.conv_offset_16(fmap1_dw16)
            offset_dw16 = self.range_16 * (torch.sigmoid(offset_dw16) - 0.5) * 2.0

            # context
            net_dw16 = F.avg_pool2d(net, 4, stride=4)
            inp_dw16 = F.avg_pool2d(inp, 4, stride=4)

            # positional encoding and self-attention
            pos_encoding_fn_small = PositionEncodingSine(
                d_model=256, max_shape=(image1.shape[2] // 16, image1.shape[3] // 16)
            )
            # 'n c h w -> n (h w) c'
            x_tmp = pos_encoding_fn_small(fmap1_dw16)
            fmap1_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3],
                                                           x_tmp.shape[1])
            # 'n c h w -> n (h w) c'
            x_tmp = pos_encoding_fn_small(fmap2_dw16)
            fmap2_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3],
                                                           x_tmp.shape[1])

            fmap1_dw16, fmap2_dw16 = self.self_att_fn(fmap1_dw16, fmap2_dw16)
            fmap1_dw16, fmap2_dw16 = [
                x.reshape(x.shape[0], image1.shape[2] // 16, -1, x.shape[2]).permute(0, 3, 1, 2)
                for x in [fmap1_dw16, fmap2_dw16]
            ]

        corr_fn = AGCL(fmap1, fmap2)
        corr_fn_dw8 = AGCL(fmap1_dw8, fmap2_dw8)
        corr_fn_att_dw16 = AGCL(fmap1_dw16, fmap2_dw16, att=self.cross_att_fn)

        # Cascaded refinement (1/16 + 1/8 + 1/4)
        predictions = []
        flow = None
        flow_up = None
        if flow_init is not None:
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = -scale * F.interpolate(
                flow_init,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        else:
            # zero initialization
            flow_dw16 = self.zero_init(fmap1_dw16)

            # Recurrent Update Module
            # RUM: 1/16
            for itr in range(iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw16 = flow_dw16.detach()
                out_corrs = corr_fn_att_dw16(
                    flow_dw16, offset_dw16, small_patch=small_patch
                )

                with autocast(enabled=self.mixed_precision):
                    net_dw16, up_mask, delta_flow = self.update_block(
                        net_dw16, inp_dw16, out_corrs, flow_dw16
                    )

                flow_dw16 = flow_dw16 + delta_flow
                flow = self.convex_upsample(flow_dw16, up_mask, rate=4)
                flow_up = -4 * F.interpolate(
                    flow,
                    size=(4 * flow.shape[2], 4 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(flow_up)

            scale = fmap1_dw8.shape[2] / flow.shape[2]
            flow_dw8 = -scale * F.interpolate(
                flow,
                size=(fmap1_dw8.shape[2], fmap1_dw8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

            # RUM: 1/8
            for itr in range(iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw8 = flow_dw8.detach()
                out_corrs = corr_fn_dw8(flow_dw8, offset_dw8, small_patch=small_patch)

                with autocast(enabled=self.mixed_precision):
                    net_dw8, up_mask, delta_flow = self.update_block(
                        net_dw8, inp_dw8, out_corrs, flow_dw8
                    )

                flow_dw8 = flow_dw8 + delta_flow
                flow = self.convex_upsample(flow_dw8, up_mask, rate=4)
                flow_up = -2 * F.interpolate(
                    flow,
                    size=(2 * flow.shape[2], 2 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(flow_up)

            scale = fmap1.shape[2] / flow.shape[2]
            flow = -scale * F.interpolate(
                flow,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

        # RUM: 1/4
        for itr in range(iters):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            flow = flow.detach()
            out_corrs = corr_fn(flow, None, small_patch=small_patch, iter_mode=True)

            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = -self.convex_upsample(flow, up_mask, rate=4)
            predictions.append(flow_up)

        if self.test_mode:
            return flow_up

        return predictions
