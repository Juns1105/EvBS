"""Blur synthesis B = D_psi(F, E, C) with the retrained ID-Blau (Eq. 3 / Eq. 8)."""
import zlib

import numpy as np
import torch

from ..models.idblau import load_idblau


def normalize_condition(sharp_rgb, flow_um, mask, feda_map, flow_norm=143.0, event_min=-119.0, event_max=63.0):
    """Build the 7-channel condition [C (3), F (3), E (1)] exactly as during ID-Blau training.

    sharp_rgb: uint8 (h, w, 3) RGB; flow_um: (3, h, w) [u/|F|, v/|F|, |F|]; mask: (h, w) in {0, 1};
    feda_map: (h, w). Returns a float tensor (7, h, w) on the flow's device.
    """
    device = flow_um.device
    sharp = torch.from_numpy(np.ascontiguousarray(sharp_rgb)).to(device).permute(2, 0, 1).float() / 255
    sharp = (sharp - 0.5) / 0.5
    flow = torch.cat([flow_um[:2], torch.clamp(flow_um[2:] / flow_norm, max=1.0)]) * mask
    event = 2 * (feda_map - event_min) / (event_max - event_min) - 1
    return torch.cat([sharp, flow, event[None]])


def sample_seed(base_seed, *key):
    """Deterministic per-sample seed so that results do not depend on processing order."""
    return (base_seed * 1000003 + zlib.crc32('/'.join(map(str, key)).encode())) % (2 ** 31)


class Blurrer:
    def __init__(self, checkpoint, sample_timesteps=20, batch_size=8, device='cuda'):
        self.model = load_idblau(checkpoint, device)
        self.sample_timesteps, self.batch_size, self.device = sample_timesteps, batch_size, device

    @torch.no_grad()
    def __call__(self, conditions, seeds):
        """conditions: list of (7, h, w) tensors of equal size; returns a list of BGR uint8 images."""
        out = []
        for i in range(0, len(conditions), self.batch_size):
            cond = torch.stack(conditions[i:i + self.batch_size]).to(self.device)
            noise = torch.stack([
                torch.randn(3, *cond.shape[-2:], generator=torch.Generator().manual_seed(s))
                for s in seeds[i:i + self.batch_size]]).to(self.device)
            blur = self.model.sample(cond, self.sample_timesteps, noise=noise)
            blur = ((blur + 1) / 2).mul(255).add(0.5).clamp(0, 255).byte()
            out += [b.permute(1, 2, 0).cpu().numpy()[:, :, ::-1].copy() for b in blur]
        return out
