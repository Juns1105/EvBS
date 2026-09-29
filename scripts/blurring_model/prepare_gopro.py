"""Generate D_psi training conditions (flow map, event mask, FEDA) for GoPro.

Writes <condition_root>/<split>/<video>/{flowmap/*.npy, mask/*.png, feda/*.npy} and
<condition_root>/<split>/stats.json with the global flow-magnitude max and FEDA min/max
(the normalization constants of the training config).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evbs.config import load_config  # noqa: E402
from evbs.events import load_events  # noqa: E402
from evbs.blurring_model.gopro import RaftFlow, composite_conditions, composite_frames, load_meta  # noqa: E402
from evbs.utils import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', nargs='*', default=[])
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    parser.add_argument('--videos', nargs='*', default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg.data
    raft = RaftFlow(cfg.raft.checkpoint, cfg.raft.iters, args.device,
                    cache_dir=os.path.join(d.condition_root, 'cache', 'raft'))

    for split in args.splits:
        meta = load_meta(d.meta_dir, split)
        stats = {'flow_mag_max': 0.0, 'feda_min': 0.0, 'feda_max': 0.0, 'count': 0}
        for video in sorted(meta):
            if args.videos and video not in args.videos:
                continue
            start = time.time()
            origin_dir = os.path.join(d.origin_root, split, video)
            origin = sorted(n for n in os.listdir(origin_dir) if n.endswith('.png'))
            out = os.path.join(d.condition_root, split, video)
            for sub in ('flowmap', 'mask', 'feda'):
                os.makedirs(os.path.join(out, sub), exist_ok=True)

            flow_fn = lambda a, b: raft(video, os.path.join(origin_dir, a), os.path.join(origin_dir, b))  # noqa: E731
            events_fn = lambda n: load_events(  # noqa: E731
                os.path.join(d.event_root, split, video, 'events', n[:-4] + '.npz'), d.height, d.width)

            for name, info in sorted(meta[video].items()):
                if info['frames_num'] == 0:
                    continue
                stem = name[:-4]
                paths = [os.path.join(out, 'flowmap', stem + '.npy'), os.path.join(out, 'mask', stem + '.png'),
                         os.path.join(out, 'feda', stem + '.npy')]
                if not args.overwrite and all(os.path.exists(p) for p in paths):
                    continue
                frames = composite_frames(origin, info['origin_img_name'], info['frames_num'])
                first = origin.index(frames[0])
                with torch.no_grad():
                    F, M, E = composite_conditions(frames, flow_fn, events_fn, d.height, d.width, cfg.feda_alpha,
                                                   prev_frame=origin[first - 1] if first > 0 else None)
                np.save(paths[0], F.cpu().numpy().astype(np.float32))
                cv2.imwrite(paths[1], (M.cpu().numpy() * 255).astype(np.uint8))
                np.save(paths[2], E.cpu().numpy().astype(np.float32))
                stats['flow_mag_max'] = max(stats['flow_mag_max'], float(F[2].max()))
                stats['feda_min'] = min(stats['feda_min'], float(E.min()))
                stats['feda_max'] = max(stats['feda_max'], float(E.max()))
                stats['count'] += 1
            print(f'[{split}/{video}] done ({time.time() - start:.1f}s)')
        if stats['count']:
            write_json(os.path.join(d.condition_root, split, 'stats.json'), stats)
            print(f'[{split}] stats (new samples only): {stats}')


if __name__ == '__main__':
    main()
