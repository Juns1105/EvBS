"""Step 2 - Intrinsic-blur synthesis: blur each content source with its own motion (InCG + D_psi).

Reads <output_dir>/sources.json and writes <output_dir>/intrinsic/<split>/<video>/{sharp,reblur}/.
"""
import os
import sys
import time

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evbs.synthesis.blurring import Blurrer, normalize_condition, sample_seed  # noqa: E402
from evbs.config import base_parser, load_config  # noqa: E402
from evbs.synthesis.incg import IntrinsicConditionGenerator  # noqa: E402
from evbs.runtime import make_dataset, make_flow, select_videos, sources_path  # noqa: E402
from evbs.utils import patch_name, read_json, write_json  # noqa: E402
from evbs.vis import event_image, flow_image  # noqa: E402


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    dataset = make_dataset(cfg, args.device)
    _, flows = make_flow(cfg, dataset, args.device)
    incg = IntrinsicConditionGenerator(dataset, flows, cfg.intrinsic.radius, cfg.blurring.feda_alpha)
    b = cfg.blurring
    blurrer = Blurrer(cfg.checkpoints.idblau, b.sample_timesteps, b.batch_size, args.device)
    sources = read_json(sources_path(cfg))['videos']
    out_root = os.path.join(cfg.output_dir, 'intrinsic')
    manifest_path = os.path.join(out_root, 'manifest.json')
    manifest = read_json(manifest_path) if (args.videos and os.path.exists(manifest_path)) else {}

    for video in select_videos(dataset, args.videos):
        if video not in sources:
            print(f'[{video}] no sources, run detect_sources.py first')
            continue
        start = time.time()
        vdir = os.path.join(out_root, cfg.dataset.split, video)
        for sub in ('sharp', 'reblur') + (('vis',) if cfg.save_visualizations else ()):
            os.makedirs(os.path.join(vdir, sub), exist_ok=True)

        items = []
        for src in sorted(sources[video]['content_sources'], key=lambda s: s['frame']):
            frame, bbox = src['frame'], tuple(src['bbox'])
            name = patch_name(frame, bbox)
            with torch.no_grad():
                cond = incg(video, frame, bbox)
            sharp = dataset.image(video, frame)[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            cv2.imwrite(os.path.join(vdir, 'sharp', name + '.png'), sharp[:, :, ::-1])
            if cfg.save_visualizations:
                cv2.imwrite(os.path.join(vdir, 'vis', name + '_flow.png'), flow_image(cond['flow'], cond['mask']))
                cv2.imwrite(os.path.join(vdir, 'vis', name + '_feda.png'), event_image(cond['feda']))
            items.append((name, frame, bbox, normalize_condition(
                sharp, cond['flow'], cond['mask'], cond['feda'], b.flow_norm, b.event_min, b.event_max)))

        blurred = blurrer([c for *_, c in items],
                          [sample_seed(b.seed, 'intrinsic', video, n) for n, *_ in items])
        for (name, frame, bbox, _), img in zip(items, blurred):
            cv2.imwrite(os.path.join(vdir, 'reblur', name + '.png'), img)
        manifest[video] = [{'name': n, 'frame': f, 'bbox': list(bb)} for n, f, bb, _ in items]
        write_json(manifest_path, manifest)
        print(f'[{video}] {len(items)} intrinsic pairs ({time.time() - start:.1f}s)')


if __name__ == '__main__':
    main()
