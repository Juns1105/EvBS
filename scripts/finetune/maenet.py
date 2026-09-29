"""Fine-tune a pre-trained MAENet on EvBS-synthesized pairs and evaluate on the real target frames.

Writes <output_dir>/{final.pth, metrics.json, train.log, eval_images/}.
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evbs.config import base_parser, load_config, to_dict  # noqa: E402
from evbs.data import EventVideoDataset  # noqa: E402
from evbs.finetune.data import EventRepresentation, RealBlurFrames, SynthesizedPairs  # noqa: E402
from evbs.models.maenet import Baseline_Biattention_CrossRecurrent_FC  # noqa: E402
from evbs.utils import seed_everything, write_json  # noqa: E402


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.sqrt((x - y) ** 2 + self.eps ** 2).mean()


def build_model(cfg, device):
    m = cfg.model
    net = Baseline_Biattention_CrossRecurrent_FC(in_chn=3, ev_chn=m.ev_chn, width=m.width,
                                                 middle_blk_num=m.middle_blk_num, enc_blk_nums=m.enc_blk_nums,
                                                 dec_blk_nums=m.dec_blk_nums, num_heads=m.num_heads)
    state = torch.load(m.checkpoint, map_location='cpu', weights_only=False)
    state = state.get('params', state.get('model_state', state))
    net.load_state_dict({k.replace('module.', '', 1): v for k, v in state.items()})
    return net.to(device)


@torch.no_grad()
def evaluate(net, frames, device, save_dir=None, save_images=0):
    net.eval()
    per_frame = []
    for i, s in enumerate(tqdm(frames, total=len(frames), desc='eval', leave=False)):
        blur, sharp, event = s['blur'].to(device), s['sharp'].to(device), s['event'].to(device)
        out = net(blur, event).clamp(0, 1)
        per_frame.append({
            'video': s['video'], 'frame': s['frame'],
            'psnr': peak_signal_noise_ratio(out, sharp, data_range=1.0).item(),
            'ssim': structural_similarity_index_measure(out, sharp, data_range=1.0).item(),
            'psnr_blur': peak_signal_noise_ratio(blur, sharp, data_range=1.0).item(),
            'ssim_blur': structural_similarity_index_measure(blur, sharp, data_range=1.0).item(),
        })
        if save_dir and i < save_images:
            os.makedirs(save_dir, exist_ok=True)
            row = [(t[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[:, :, ::-1]
                   for t in (blur, out, sharp)]
            cv2.imwrite(os.path.join(save_dir, f"{s['video']}_{s['frame']:05d}.jpg"), cv2.hconcat(row))
    summary = {k: float(np.mean([f[k] for f in per_frame])) for k in ('psnr', 'ssim', 'psnr_blur', 'ssim_blur')}
    videos = sorted({f['video'] for f in per_frame})
    summary['per_video_psnr'] = {v: float(np.mean([f['psnr'] for f in per_frame if f['video'] == v])) for v in videos}
    return summary


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    t = cfg.train
    seed_everything(t.seed)
    device = torch.device(args.device)
    os.makedirs(cfg.output_dir, exist_ok=True)
    log = open(os.path.join(cfg.output_dir, 'train.log'), 'a')

    def say(msg):
        print(msg)
        log.write(msg + '\n')
        log.flush()

    d = cfg.dataset
    dataset = EventVideoDataset(d.root, d.split, d.height, d.width, d.blur_dir, d.event_dir, device=args.device)
    rep = EventRepresentation(cfg.representation, d.height, d.width)
    start = time.time()
    pairs = SynthesizedPairs(cfg.synthesis_dir, dataset, rep, d.split, tuple(cfg.kinds), cfg.intrinsic_radius,
                             crop_size=t.crop_size, augment=True, device=args.device)
    counts = {k: sum(s['kind'] == k for s in pairs.samples) for k in cfg.kinds}
    say(f'config: {json.dumps(to_dict(cfg))}')
    say(f'training pairs: {counts} (loaded in {time.time() - start:.0f}s)')
    loader = DataLoader(pairs, batch_size=t.batch_size, shuffle=True, num_workers=t.num_workers,
                        drop_last=True, pin_memory=True)
    frames = RealBlurFrames(dataset, d.gt_dir, rep, cfg.eval.frame_step)

    net = build_model(cfg, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=t.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t.epochs, eta_min=t.min_lr)
    criterion = CharbonnierLoss()
    metrics = {}

    def run_eval(epoch):
        res = evaluate(net, frames, device, os.path.join(cfg.output_dir, 'eval_images', f'epoch{epoch:03d}'),
                       cfg.eval.save_images)
        metrics[epoch] = res
        write_json(os.path.join(cfg.output_dir, 'metrics.json'), metrics)
        say(f"epoch {epoch}: PSNR {res['psnr']:.3f} SSIM {res['ssim']:.4f} ({len(frames)} frames)")

    if 0 in cfg.eval.epochs:
        run_eval(0)
    for epoch in range(1, t.epochs + 1):
        net.train()
        total, n = 0.0, 0
        pbar = tqdm(loader, desc=f'epoch {epoch}/{t.epochs}', leave=False)
        for s in pbar:
            blur, sharp, event = (s[k].to(device, non_blocking=True) for k in ('blur', 'sharp', 'event'))
            loss = criterion(net(blur, event), sharp)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), t.grad_clip)
            optimizer.step()
            total, n = total + loss.item(), n + 1
            pbar.set_postfix(loss=total / n)
        scheduler.step()
        say(f'epoch {epoch}: train loss {total / n:.5f} lr {optimizer.param_groups[0]["lr"]:.2e}')
        if epoch in cfg.eval.epochs or epoch == t.epochs:
            run_eval(epoch)
    if t.epochs > 0:
        torch.save({'params': net.state_dict(), 'config': to_dict(cfg)}, os.path.join(cfg.output_dir, 'final.pth'))
    say(f'done in {(time.time() - start) / 60:.1f} min')


if __name__ == '__main__':
    main()
