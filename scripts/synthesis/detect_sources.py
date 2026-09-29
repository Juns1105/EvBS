"""Step 1 - Dual Source Extractor: find content sources and motion sources in the target domain.

Writes <output_dir>/sources.json.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evbs.config import base_parser, load_config, to_dict  # noqa: E402
from evbs.synthesis.dse import DualSourceExtractor  # noqa: E402
from evbs.synthesis.incg import intrinsic_frames  # noqa: E402
from evbs.runtime import make_dataset, make_flow, select_videos, sources_path  # noqa: E402
from evbs.utils import read_json, write_json  # noqa: E402


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    dataset = make_dataset(cfg, args.device)
    estimator, _ = make_flow(cfg, dataset, args.device)
    dse = DualSourceExtractor(cfg.dse, dataset, estimator, cfg.checkpoints.bme, args.device,
                              cfg.eraft.patch_batch)

    path = sources_path(cfg)
    result = read_json(path) if (args.videos and os.path.exists(path)) else {'videos': {}}
    result['config'] = {'dataset': to_dict(cfg.dataset), 'dse': to_dict(cfg.dse),
                        'intrinsic': to_dict(cfg.intrinsic)}

    for video in select_videos(dataset, args.videos):
        start = time.time()
        frames = dataset.frames(video)
        content_frames = [f for f in frames
                          if dataset.has_frames(video, intrinsic_frames(f, cfg.intrinsic.radius))]
        with torch.no_grad():
            tau, content, motion = dse.run_video(video, content_frames)
        result['videos'][video] = {
            'tau': tau,
            'num_frames': len(frames),
            'content_sources': [{'frame': p['frame'], 'bbox': list(p['bbox']), 'score': p['score']}
                                for p in content],
            'motion_sources': [{'frame': p['frame'], 'bbox': list(p['bbox']), 'score': p['score'],
                                'coherence': p['coherence']} for p in motion],
        }
        write_json(path, result)
        print(f'[{video}] tau={tau:.4f} content={len(content)} motion={len(motion)} '
              f'({time.time() - start:.1f}s)')
    print(f'saved {path}')


if __name__ == '__main__':
    main()
