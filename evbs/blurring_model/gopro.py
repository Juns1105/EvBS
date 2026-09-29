"""GoPro training data for retraining the blurring model D_psi (ID-Blau) with event conditions.

Sources:
    origin_root/<split>/<video>/<frame>.png              240 fps sharp frames (GOPRO_Large_all)
    composite_root/<split>/<video>/blur/<name>.png       GoPro blurry frames (average of N sharp frames)
    event_root/<split>/<video>/events/<frame>.npz        synthetic events of each sharp frame interval
    meta_dir/<split>_composite_img_frames.json           name -> {origin_img_name (center), frames_num (N)}

For a blurry frame averaging sharp frames f_0..f_{N-1}, the conditions are
    flow map  F = sum_j (F_{f_j -> f_j+1} - F_{f_j+1 -> f_j}) / 2  over the N-1 frame pairs (RAFT on sharp frames)
    mask      M = pixels with at least one event in the groups of f_0..f_{N-2}
    FEDA      E over the same N-1 event groups (see evbs/feda.py)
"""
import json
import os
import random
from argparse import Namespace

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..events import Events, event_mask
from ..feda import feda
from ..flow import flow_map, unit_magnitude
from ..models.raft.raft import RAFT
from ..models.raft.utils import InputPadder


def load_meta(meta_dir, split):
    with open(os.path.join(meta_dir, f'{split}_composite_img_frames.json')) as f:
        return json.load(f)


def composite_frames(origin_names, center_name, num_frames):
    """Names of the N sharp frames averaged into one GoPro blurry frame."""
    center = origin_names.index(center_name)
    first = center - num_frames // 2
    frames = origin_names[first:first + num_frames]
    assert len(frames) == num_frames
    return frames


class RaftFlow:
    """Image-based RAFT (raft-things) between consecutive sharp frames, cached on disk."""

    def __init__(self, checkpoint, iters=20, device='cuda', cache_dir=None):
        args = Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
        model = RAFT(args)
        state = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict({k.replace('module.', '', 1): v for k, v in state.items()})
        self.model = model.to(device).eval()
        self.iters, self.device, self.cache_dir = iters, device, cache_dir

    def _image(self, path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(self.device)

    @torch.no_grad()
    def __call__(self, video, path_a, path_b):
        key = os.path.basename(path_a)[:-4] + '_' + os.path.basename(path_b)[:-4]
        cache = None if self.cache_dir is None else os.path.join(self.cache_dir, video, key + '.npy')
        if cache is not None and os.path.exists(cache):
            return torch.from_numpy(np.load(cache).astype(np.float32)).to(self.device)
        a, b = self._image(path_a), self._image(path_b)
        padder = InputPadder(a.shape)
        a, b = padder.pad(a, b)
        _, flow = self.model(a, b, iters=self.iters, test_mode=True)
        flow = padder.unpad(flow)[0]
        if cache is not None:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, flow.cpu().numpy().astype(np.float16))
        return flow


def composite_conditions(frames, flow_fn, events_fn, height, width, alpha=0.1, prev_frame=None):
    """Conditions of one blurry frame.

    frames: the N sharp frame names; flow_fn(a, b) -> F_{a->b} (2, H, W); events_fn(name) -> Events;
    prev_frame: the sharp frame before frames[0] (None at the start of a video).
    Returns flow (3, H, W) [unit, magnitude], mask (H, W) and FEDA (H, W).
    """
    pairs = list(zip(frames[:-1], frames[1:]))
    fwd = [flow_fn(a, b) for a, b in pairs]
    bwd = [flow_fn(b, a) for a, b in pairs]
    F = flow_map(fwd, bwd)

    back = [flow_fn(frames[0], prev_frame) if prev_frame is not None else None] + bwd[:-1]
    groups = [events_fn(n) for n in frames[:-1]]
    E = feda([g.to(F.device) for g in groups], fwd, back, height, width, alpha)
    M = event_mask(Events.cat(groups).to(F.device), height, width)
    return unit_magnitude(F), M, E


def rotate_flow90(flow_uv, k):
    """Rotate flow vectors consistently with np.rot90(image, k) (counter-clockwise, y pointing down)."""
    u, v = flow_uv[..., 0].copy(), flow_uv[..., 1].copy()
    for _ in range(k % 4):
        u, v = v, -u
    return np.stack([u, v], axis=-1)


class GoProBlurDataset(Dataset):
    """(sharp, blur, flow, event) samples for training D_psi, normalized as in ID-Blau."""

    def __init__(self, cfg, split, augment):
        d = cfg.data
        self.crop = cfg.train.crop_size if augment else None
        self.augment = augment
        n = cfg.normalization
        self.flow_norm, self.event_min, self.event_max = n.flow_norm, n.event_min, n.event_max
        meta = load_meta(d.meta_dir, split)
        self.items = []
        for video in sorted(meta):
            cond_dir = os.path.join(d.condition_root, split, video)
            for name, info in sorted(meta[video].items()):
                if info['frames_num'] == 0:
                    continue
                stem = name[:-4]
                self.items.append({
                    'blur': os.path.join(d.composite_root, split, video, 'blur', name),
                    'sharp': os.path.join(d.origin_root, split, video, info['origin_img_name']),
                    'flow': os.path.join(cond_dir, 'flowmap', stem + '.npy'),
                    'mask': os.path.join(cond_dir, 'mask', stem + '.png'),
                    'feda': os.path.join(cond_dir, 'feda', stem + '.npy'),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        blur = cv2.cvtColor(cv2.imread(it['blur']), cv2.COLOR_BGR2RGB).astype(np.float32)
        sharp = cv2.cvtColor(cv2.imread(it['sharp']), cv2.COLOR_BGR2RGB).astype(np.float32)
        mask = cv2.imread(it['mask'], cv2.IMREAD_GRAYSCALE).astype(np.float32)[..., None] / 255
        flow = np.load(it['flow']).astype(np.float32).transpose(1, 2, 0)   # (H, W, 3) [u, v, |F|]
        flow[..., 2] = np.minimum(flow[..., 2] / self.flow_norm, 1.0)
        flow = flow * mask
        event = np.load(it['feda']).astype(np.float32)[..., None]

        if self.augment:
            H, W = blur.shape[:2]
            top, left = random.randint(0, H - self.crop), random.randint(0, W - self.crop)
            sl = (slice(top, top + self.crop), slice(left, left + self.crop))
            blur, sharp, flow, event = blur[sl], sharp[sl], flow[sl], event[sl]
            if random.random() < 0.5:
                blur, sharp, flow, event = (np.fliplr(a) for a in (blur, sharp, flow, event))
                flow = flow * np.array([-1, 1, 1], np.float32)
            if random.random() < 0.5:
                blur, sharp, flow, event = (np.flipud(a) for a in (blur, sharp, flow, event))
                flow = flow * np.array([1, -1, 1], np.float32)
            k = random.randint(0, 3)
            blur, sharp, flow, event = (np.rot90(a, k) for a in (blur, sharp, flow, event))
            flow = np.concatenate([rotate_flow90(flow[..., :2], k), flow[..., 2:]], axis=-1)

        to_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        return {
            'blur': to_tensor(blur / 127.5 - 1),
            'sharp': to_tensor(sharp / 127.5 - 1),
            'flow': to_tensor(flow),
            'event': to_tensor(2 * (event - self.event_min) / (self.event_max - self.event_min) - 1),
        }
