"""Data for fine-tuning event-based deblurring models on EvBS-synthesized pairs."""
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..events import Events, load_events
from ..da import da, da_asynav


def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class EventRepresentation:
    """Event tensor fed to the deblurring model.

    name: 'da' (MAENet Algorithm 1, default) or 'da_asynav' (MAENet code variant, normalized by the
    absolute maximum of the full-frame voxel). Crops of a frame are normalized with the full-frame
    scale, so training patches match full-frame test inputs.
    """

    def __init__(self, name, height, width):
        if name not in ('da', 'da_asynav'):
            raise ValueError(f'unknown event representation {name!r}')
        self.name, self.height, self.width = name, height, width

    def _raw(self, ev, h, w, t_range):
        if self.name == 'da_asynav':
            return da_asynav(ev, h, w, normalize=False, t_range=t_range)
        return da(ev, h, w, t_range=t_range)

    def _normalize(self, rep, reference):
        if self.name != 'da_asynav':
            return rep
        peak = reference.abs().max()
        return rep / peak if peak > 0 else rep

    def full_frame(self, ev):
        rep = self._raw(ev, self.height, self.width, None)
        return self._normalize(rep, rep)

    def patch(self, ev, bbox, reference_events=None):
        """Representation of the events inside bbox; the time windows and the normalization come from
        reference_events (the full-frame stream the patch belongs to), defaulting to ev itself."""
        ref = ev if reference_events is None else reference_events
        t_range = (int(ref.t.min()), int(ref.t.max()))
        h, w = bbox[2] - bbox[0], bbox[3] - bbox[1]
        rep = self._raw(ev.crop(bbox), h, w, t_range)
        if self.name == 'da_asynav':
            rep = self._normalize(rep, self._raw(ref, self.height, self.width, t_range))
        return rep


class SynthesizedPairs(Dataset):
    """Intrinsic and extrinsic pairs from scripts/synthesis/synthesize_*.py, preloaded in memory.

    Intrinsic pair: input events are the original events of frames t-2..t+1 (the window used to
    synthesize the blur). Extrinsic pair: the warped events of frames t-1 and t.
    """

    def __init__(self, synthesis_dir, dataset, representation, split='test', kinds=('intrinsic', 'extrinsic'),
                 radius=2, crop_size=None, augment=True, device='cuda'):
        self.crop_size, self.augment = crop_size, augment
        self.samples = []
        H, W = dataset.height, dataset.width
        for kind in kinds:
            manifest = json.load(open(os.path.join(synthesis_dir, kind, 'manifest.json')))
            for video in sorted(manifest):
                vdir = os.path.join(synthesis_dir, kind, split, video)
                for rec in tqdm(manifest[video], desc=f'{kind} {video}', leave=False):
                    if kind == 'intrinsic':
                        frame, bbox = rec['frame'], tuple(rec['bbox'])
                        ref = Events.cat([dataset.events(video, n) for n in range(frame - radius, frame + radius)]).to(device)
                        rep = representation.patch(ref, bbox)
                        self._add(kind, video, os.path.join(vdir, 'sharp', rec['name'] + '.png'),
                                  os.path.join(vdir, 'reblur', rec['name'] + '.png'), rep)
                    else:
                        content = rec['content']
                        frame = int(content.split('_')[0])
                        bbox = tuple(int(v) for v in content.split('_')[1:5])
                        ref = Events.cat([dataset.events(video, n) for n in (frame - 1, frame)]).to(device)
                        for pair in rec['pairs']:
                            npz = os.path.join(vdir, 'translated_event_combined', pair['name'] + '.npz')
                            warped = load_events(npz, H, W, device)
                            rep = representation.patch(warped, bbox, reference_events=ref)
                            self._add(kind, video, os.path.join(vdir, 'sharp', content + '.png'),
                                      os.path.join(vdir, 'transblur', pair['name'] + '.png'), rep)

    def _add(self, kind, video, sharp_path, blur_path, rep):
        self.samples.append({'kind': kind, 'video': video, 'sharp': _read_rgb(sharp_path),
                             'blur': _read_rgb(blur_path), 'event': rep.half().cpu().numpy()})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        sharp, blur = s['sharp'], s['blur']
        event = s['event'].astype(np.float32).transpose(1, 2, 0)
        if self.crop_size:
            H, W = sharp.shape[:2]
            top, left = random.randint(0, H - self.crop_size), random.randint(0, W - self.crop_size)
            sl = (slice(top, top + self.crop_size), slice(left, left + self.crop_size))
            sharp, blur, event = sharp[sl], blur[sl], event[sl]
        if self.augment:
            if random.random() < 0.5:
                sharp, blur, event = (np.fliplr(a) for a in (sharp, blur, event))
            if random.random() < 0.5:
                sharp, blur, event = (np.flipud(a) for a in (sharp, blur, event))
            k = random.randint(0, 3)
            sharp, blur, event = (np.rot90(a, k) for a in (sharp, blur, event))
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        return {'sharp': to_t(sharp).float() / 255, 'blur': to_t(blur).float() / 255, 'event': to_t(event)}


class RealBlurFrames:
    """Real blurry frames of the target test set with ground truth, for evaluation."""

    def __init__(self, dataset, gt_dir, representation, step=1):
        self.dataset, self.gt_dir, self.representation = dataset, gt_dir, representation
        self.items = [(v, f) for v in dataset.videos for f in dataset.frames(v)[::step]
                      if os.path.exists(self._gt(v, f))]

    def _gt(self, video, frame):
        return os.path.join(self.dataset.root, video, self.gt_dir, self.dataset.name(frame) + '.png')

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for video, frame in self.items:
            to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float() / 255
            ev = self.dataset.events(video, frame)
            yield {'video': video, 'frame': frame, 'blur': to_t(self.dataset.image(video, frame)),
                   'sharp': to_t(_read_rgb(self._gt(video, frame))),
                   'event': self.representation.full_frame(ev)[None]}
