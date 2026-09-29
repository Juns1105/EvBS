# E-RAFT: Dense Optical Flow from Event Cameras (Gehrig et al., 3DV 2021).
# Adapted from https://github.com/uzh-rpg/E-RAFT (MIT License, see LICENSE).
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .corr import CorrBlock
from .extractor import BasicEncoder
from .update import BasicUpdateBlock
from .utils import coords_grid, upflow8


class ERAFT(nn.Module):
    def __init__(self, n_first_channels=15, subtype='standard', mixed_precision=False):
        super().__init__()
        assert subtype in ('standard', 'warm_start')
        self.subtype = subtype
        self.args = Namespace(mixed_precision=mixed_precision, corr_levels=4, corr_radius=4)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0,
                                 n_first_channels=n_first_channels)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=0,
                                 n_first_channels=n_first_channels)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0."""
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination."""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None):
        """Estimate optical flow between two event voxel grids of shape (N, bins, H, W).

        H and W must be multiples of 8. Returns (low-res flow, list of full-res predictions).
        """
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        hdim, cdim = self.hidden_dim, self.context_dim
        amp = self.args.mixed_precision

        with torch.autocast(device_type=image1.device.type, enabled=amp):
            fmap1, fmap2 = self.fnet([image1, image2])
        corr_fn = CorrBlock(fmap1.float(), fmap2.float(), radius=self.args.corr_radius)

        with torch.autocast(device_type=image1.device.type, enabled=amp):
            cnet = self.cnet(image2)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with torch.autocast(device_type=image1.device.type, enabled=amp):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        return coords1 - coords0, flow_predictions
