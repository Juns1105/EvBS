"""Event-based optical flow (E-RAFT), flow caching, and flow-map utilities."""
import os
from collections import OrderedDict

import numpy as np
import torch

from .events import voxel_grid
from .models.eraft.eraft import ERAFT


class EventFlowEstimator:
    """E-RAFT on event voxel grids. H and W must be multiples of 8."""

    def __init__(self, checkpoint, num_bins=15, iters=12, max_batch=2, device='cuda'):
        self.num_bins, self.iters, self.max_batch, self.device = num_bins, iters, max_batch, device
        self.model = ERAFT(n_first_channels=num_bins).to(device).eval()
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model.load_state_dict(state.get('model', state))

    def voxel(self, ev, height, width):
        return voxel_grid(ev.to(self.device), self.num_bins, height, width)

    @torch.no_grad()
    def from_voxels(self, voxels1, voxels2, max_batch=None):
        """voxels1, voxels2: (N, bins, H, W). Returns flows (N, 2, H, W) from 1 to 2."""
        max_batch = max_batch or self.max_batch
        out = []
        for i in range(0, voxels1.shape[0], max_batch):
            _, preds = self.model(voxels1[i:i + max_batch], voxels2[i:i + max_batch], iters=self.iters)
            out.append(preds[-1])
        return torch.cat(out)

    def __call__(self, pairs, height, width):
        """pairs: list of (Events, Events). Returns (N, 2, H, W) flows from the first to the second stream."""
        v1 = torch.stack([self.voxel(a, height, width) for a, _ in pairs])
        v2 = torch.stack([self.voxel(b, height, width) for _, b in pairs])
        return self.from_voxels(v1, v2)


class FlowStore:
    """Full-frame E-RAFT flows of a target-domain video, computed lazily and cached (memory + disk).

    interframe(video, a, b): F_{a->b} between the event streams of frames a and b.
    segment(video, f, n): flows between n equal-duration segments of frame f's own event stream,
        returned as (forward [F_{s_i -> s_i+1}], backward [F_{s_i+1 -> s_i}]).
    """

    def __init__(self, dataset, estimator, cache_dir=None, memory_size=48):
        self.dataset, self.estimator, self.cache_dir = dataset, estimator, cache_dir
        self._memory = OrderedDict()
        self._memory_size = memory_size

    def _path(self, video, key):
        return None if self.cache_dir is None else os.path.join(self.cache_dir, video, key + '.npy')

    def _load(self, video, key):
        mk = (video, key)
        if mk in self._memory:
            self._memory.move_to_end(mk)
            return self._memory[mk]
        path = self._path(video, key)
        if path is not None and os.path.exists(path):
            flow = torch.from_numpy(np.load(path).astype(np.float32)).to(self.estimator.device)
            self._remember(mk, flow)
            return flow
        return None

    def _save(self, video, key, flow):
        self._remember((video, key), flow)
        path = self._path(video, key)
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + '.tmp.npy'
            np.save(tmp, flow.cpu().numpy().astype(np.float16))
            os.replace(tmp, path)

    def _remember(self, mk, flow):
        self._memory[mk] = flow
        self._memory.move_to_end(mk)
        while len(self._memory) > self._memory_size:
            self._memory.popitem(last=False)

    def interframe(self, video, a, b):
        return self.interframe_many(video, [(a, b)])[0]

    def interframe_many(self, video, pairs):
        H, W = self.dataset.height, self.dataset.width
        keys = [f'{self.dataset.name(a)}_{self.dataset.name(b)}' for a, b in pairs]
        flows = [self._load(video, k) for k in keys]
        missing = [i for i, f in enumerate(flows) if f is None]
        if missing:
            ev_pairs = [(self.dataset.events(video, pairs[i][0]), self.dataset.events(video, pairs[i][1]))
                        for i in missing]
            computed = self.estimator(ev_pairs, H, W)
            for i, flow in zip(missing, computed):
                self._save(video, keys[i], flow)
                flows[i] = flow
        return flows

    def segment(self, video, frame, num_splits=3):
        H, W = self.dataset.height, self.dataset.width
        n = num_splits - 1
        keys = ([f'seg{num_splits}_{self.dataset.name(frame)}_f{i}' for i in range(n)]
                + [f'seg{num_splits}_{self.dataset.name(frame)}_b{i}' for i in range(n)])
        flows = [self._load(video, k) for k in keys]
        if any(f is None for f in flows):
            segs = self.dataset.events(video, frame).split_time(num_splits)
            vox = torch.stack([self.estimator.voxel(s, H, W) for s in segs])
            flows = list(self.estimator.from_voxels(torch.cat([vox[:-1], vox[1:]]),
                                                    torch.cat([vox[1:], vox[:-1]])))
            for k, f in zip(keys, flows):
                self._save(video, k, f)
        return flows[:n], flows[n:]


def flow_map(forward, backward):
    """Bidirectional aggregation F = sum_n (F_{n->n+1} - F_{n+1->n}) / 2."""
    return (sum(forward) - sum(backward)) / 2


def unit_magnitude(flow, eps=1e-12):
    """(2, H, W) flow -> (3, H, W) [u / |F|, v / |F|, |F|] as used by the blurring model."""
    mag = torch.sqrt(flow[0] ** 2 + flow[1] ** 2 + eps)
    return torch.stack([flow[0] / mag, flow[1] / mag, mag])


def crop(tensor, bbox):
    y1, x1, y2, x2 = bbox
    return tensor[..., y1:y2, x1:x2]


def mean_motion(flow, bbox=None, min_mag=0.01):
    """Mean flow vector over pixels with |F| > min_mag, as (unit direction, mean magnitude).

    Returns (None, 0.0) if no pixel is valid.
    """
    if bbox is not None:
        flow = crop(flow, bbox)
    mag = torch.linalg.vector_norm(flow, dim=0)
    valid = mag > min_mag
    if not valid.any():
        return None, 0.0
    mean_vec = flow[:, valid].mean(dim=1)
    norm = torch.linalg.vector_norm(mean_vec)
    direction = mean_vec / norm if norm >= 1e-6 else torch.zeros_like(mean_vec)
    return direction, float(mag[valid].mean())


def mean_orientation(flow, bbox=None, min_mag=0.1):
    """Average of per-pixel unit directions over pixels with |F| > min_mag, renormalized (Eq. 5).

    Returns None if undefined.
    """
    if bbox is not None:
        flow = crop(flow, bbox)
    mag = torch.linalg.vector_norm(flow, dim=0)
    valid = mag > min_mag
    if not valid.any():
        return None
    mean_unit = (flow[:, valid] / mag[valid]).mean(dim=1)
    norm = torch.linalg.vector_norm(mean_unit)
    return mean_unit / norm if norm >= 1e-6 else None


def angle_deg(u, v):
    cos = float(torch.clamp(torch.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))
