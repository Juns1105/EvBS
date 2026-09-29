"""Step 3 - Extrinsic-blur synthesis: transfer motion from motion sources to content sources
(ExCG, Algorithm 1) and blur with D_psi.

Reads <output_dir>/sources.json and writes <output_dir>/extrinsic/<split>/<video>/
    sharp/<content>.png
    transblur/<content>_<motion>_mcs<MCS>_deg<angle>.png
    translated_event_combined/<same name>.npz   warped events of frames t-1 and t (sensor coordinates)
"""
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evbs.synthesis.blurring import Blurrer, normalize_condition, sample_seed  # noqa: E402
from evbs.config import base_parser, load_config  # noqa: E402
from evbs.events import Events  # noqa: E402
from evbs.synthesis.excg import ExtrinsicConditionGenerator  # noqa: E402
from evbs.runtime import make_dataset, make_flow, select_videos, sources_path  # noqa: E402
from evbs.utils import patch_name, read_json, write_json  # noqa: E402
from evbs.vis import event_image, flow_image  # noqa: E402


def select_pairs(excg, content, motions, cfg):
    """Angle gate (tau_low < phi < tau_high), motion transfer, then top-n by MCS."""
    lo, hi = cfg.angle_range
    attempts = [(lo, hi)]
    if cfg.relax_range is not None:
        attempts.append(((cfg.relax_range[0] + lo) / 2, (cfg.relax_range[1] + hi) / 2))
    for lo, hi in attempts:
        gated = [(m, a) for m in motions if (a := excg.angle(content, m)) is not None and lo < a < hi]
        results = excg.transfer(content, [m for m, _ in gated])
        if results:
            angles = {id(m): a for m, a in gated}
            for r in results:
                r['angle'] = angles[id(r['motion'])]
            return sorted(results, key=lambda r: r['mcs'], reverse=True)[:cfg.top_n], (lo, hi)
    return [], attempts[-1]


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    dataset = make_dataset(cfg, args.device)
    estimator, flows = make_flow(cfg, dataset, args.device)
    b, x = cfg.blurring, cfg.extrinsic
    alpha = b.feda_alpha if getattr(x, 'feda_alpha', None) is None else x.feda_alpha
    excg = ExtrinsicConditionGenerator(dataset, flows, estimator, x, alpha, cfg.eraft.patch_batch)
    blurrer = Blurrer(cfg.checkpoints.idblau, b.sample_timesteps, b.batch_size, args.device)
    sources = read_json(sources_path(cfg))['videos']
    out_root = os.path.join(cfg.output_dir, 'extrinsic')
    manifest_path = os.path.join(out_root, 'manifest.json')
    manifest = read_json(manifest_path) if (args.videos and os.path.exists(manifest_path)) else {}

    motion_cache = {}

    def motions_of(video):
        if video not in motion_cache:
            with torch.no_grad():
                motion_cache[video] = [excg.motion(video, m['frame'], tuple(m['bbox']))
                                       for m in sources[video]['motion_sources']]
        return motion_cache[video]

    for video in select_videos(dataset, args.videos):
        if video not in sources:
            print(f'[{video}] no sources, run detect_sources.py first')
            continue
        start = time.time()
        vdir = os.path.join(out_root, cfg.dataset.split, video)
        subdirs = ('sharp', 'transblur', 'translated_event_combined', 'motion')
        for sub in subdirs + (('vis',) if cfg.save_visualizations else ()):
            os.makedirs(os.path.join(vdir, sub), exist_ok=True)
        motions = (motions_of(video) if x.same_video_only
                   else [m for v in sources for m in motions_of(v)])

        items, records = [], []
        for src in sorted(sources[video]['content_sources'], key=lambda s: s['frame']):
            frame, bbox = src['frame'], tuple(src['bbox'])
            cname = patch_name(frame, bbox)
            with torch.no_grad():
                content = excg.content(video, frame, bbox)
                pairs, used_range = select_pairs(excg, content, motions, x)
            if not pairs:
                records.append({'content': cname, 'pairs': [], 'angle_range': list(used_range)})
                continue

            sharp = dataset.image(video, frame)[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            cv2.imwrite(os.path.join(vdir, 'sharp', cname + '.png'), sharp[:, :, ::-1])
            rec = {'content': cname, 'angle_range': list(used_range), 'pairs': []}
            for r in pairs:
                m = r['motion']
                mname = patch_name(m['frame'], m['bbox'])
                name = f"{cname}_{mname}_mcs{r['mcs']:.4f}_deg{r['angle']:.2f}"
                with torch.no_grad():
                    cond = excg.condition(r, bbox)
                g_prev, g_cur, _ = r['groups']
                warped = Events.cat([g_prev, g_cur]).uncrop(bbox).numpy()
                np.savez_compressed(os.path.join(vdir, 'translated_event_combined', name + '.npz'), **warped)
                mimg = dataset.image(m['video'], m['frame'])
                my1, mx1, my2, mx2 = m['bbox']
                cv2.imwrite(os.path.join(vdir, 'motion', f"{m['video']}_{mname}.png"),
                            mimg[my1:my2, mx1:mx2, ::-1])
                if cfg.save_visualizations:
                    cv2.imwrite(os.path.join(vdir, 'vis', name + '_flow.png'), flow_image(cond['flow'], cond['mask']))
                    cv2.imwrite(os.path.join(vdir, 'vis', name + '_feda.png'), event_image(cond['feda']))
                items.append((name, normalize_condition(sharp, cond['flow'], cond['mask'], cond['feda'],
                                                        b.flow_norm, b.event_min, b.event_max)))
                rec['pairs'].append({'name': name, 'motion_video': m['video'], 'motion': mname,
                                     'mcs': r['mcs'], 'angle': r['angle']})
            records.append(rec)

        blurred = blurrer([c for _, c in items], [sample_seed(b.seed, 'extrinsic', video, n) for n, _ in items])
        for (name, _), img in zip(items, blurred):
            cv2.imwrite(os.path.join(vdir, 'transblur', name + '.png'), img)
        manifest[video] = records
        write_json(manifest_path, manifest)
        n_content = sum(1 for r in records if r['pairs'])
        print(f'[{video}] {len(items)} extrinsic pairs for {n_content}/{len(records)} content sources '
              f'({time.time() - start:.1f}s)')


if __name__ == '__main__':
    main()
