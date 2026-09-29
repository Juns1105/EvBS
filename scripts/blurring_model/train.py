"""Train the blurring model D_psi (ID-Blau with flow-map + FEDA conditions) on GoPro.

Run scripts/blurring_model/prepare_gopro.py first. Checkpoints: <output_dir>/{last,best,final}.pth
(plus epoch_<n>.pth every ckpt_every epochs); they load with evbs.models.idblau.load_idblau.
"""
import argparse
import math
import os
import sys
from itertools import islice

import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evbs.config import load_config, to_dict  # noqa: E402
from evbs.blurring_model.gopro import GoProBlurDataset  # noqa: E402
from evbs.models.idblau import build_idblau  # noqa: E402
from evbs.utils import seed_everything  # noqa: E402


def to_image(x):
    """[-1, 1] (3, H, W) tensor -> BGR uint8."""
    x = ((x.clamp(-1, 1) + 1) / 2).mul(255).add(0.5).clamp(0, 255).byte()
    return x.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]


@torch.no_grad()
def validate(model, loader, cfg, epoch, save_dir, device):
    model.eval()
    psnrs = []
    for i, s in enumerate(islice(loader, cfg.train.val_samples)):
        cond = torch.cat([s['sharp'], s['flow'], s['event']], 1).to(device)
        out = model.sample(cond, cfg.train.sample_timesteps).clamp(-1, 1)
        blur = s['blur'].to(device)
        mse = torch.mean(((out + 1) / 2 - (blur + 1) / 2) ** 2).item()
        psnrs.append(10 * math.log10(1 / max(mse, 1e-10)))
        if i < 3:
            os.makedirs(save_dir, exist_ok=True)
            row = [to_image(s['sharp'][0]), to_image(out[0]), to_image(blur[0])]
            cv2.imwrite(os.path.join(save_dir, f'{epoch:05d}_{i}.png'), cv2.hconcat(row))
    return sum(psnrs) / len(psnrs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', nargs='*', default=[])
    parser.add_argument('--init', default=None, help='initialize weights from a checkpoint (e.g. to fine-tune)')
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    t = cfg.train
    seed_everything(t.seed)
    device = torch.device('cuda')
    os.makedirs(t.output_dir, exist_ok=True)

    train_loader = DataLoader(GoProBlurDataset(cfg, 'train', augment=True), batch_size=t.batch_size,
                              shuffle=True, num_workers=t.num_workers, drop_last=False)
    val_loader = DataLoader(GoProBlurDataset(cfg, 'test', augment=False), batch_size=1,
                            shuffle=False, num_workers=t.num_workers)

    hparams = to_dict(cfg.model)
    hparams['channel_mults'] = tuple(hparams['channel_mults'])
    model = build_idblau(**hparams).to(device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=t.lr, betas=(0.9, 0.999))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t.epochs, eta_min=t.min_lr)
                 if t.scheduler == 'cosine' else None)

    start_epoch, best_psnr = 1, 0.0
    last_path = os.path.join(t.output_dir, 'last.pth')
    if os.path.exists(last_path):
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        if scheduler and state.get('scheduler_state'):
            scheduler.load_state_dict(state['scheduler_state'])
        start_epoch, best_psnr = state['epoch'] + 1, state.get('best_psnr', 0.0)
        print(f'resumed from epoch {state["epoch"]}')
    elif args.init:
        state = torch.load(args.init, map_location='cpu', weights_only=False)
        model.load_state_dict({k.replace('model.module.', 'model.', 1): v for k, v in state['model_state'].items()})
        print(f'initialized from {args.init}')

    if torch.cuda.device_count() > 1:
        model.model = torch.nn.DataParallel(model.model)
    writer = SummaryWriter(os.path.join(t.output_dir, 'log'))

    def save(path, epoch, full=True):
        net_state = {k.replace('model.module.', 'model.', 1): v for k, v in model.state_dict().items()}
        state = {'model_state': net_state, 'hparams': hparams, 'config': to_dict(cfg), 'epoch': epoch,
                 'best_psnr': best_psnr}
        if full:
            state['optimizer_state'] = optimizer.state_dict()
            state['scheduler_state'] = scheduler.state_dict() if scheduler else None
        torch.save(state, path)

    for epoch in range(start_epoch, t.epochs + 1):
        model.train()
        total, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f'epoch {epoch}/{t.epochs}')
        for s in pbar:
            blur = s['blur'].to(device)
            cond = torch.cat([s['sharp'], s['flow'], s['event']], 1).to(device)
            loss = model(blur, cond)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
            optimizer.step()
            total, n = total + loss.item(), n + 1
            pbar.set_postfix(loss=total / n, lr=optimizer.param_groups[0]['lr'])
        if scheduler:
            scheduler.step()
        writer.add_scalar('train/loss', total / n, epoch)

        if epoch % t.val_every == 0 or epoch == t.epochs:
            psnr = validate(model, val_loader, cfg, epoch, os.path.join(t.output_dir, 'val'), device)
            writer.add_scalar('val/psnr', psnr, epoch)
            print(f'epoch {epoch}: val PSNR {psnr:.3f}')
            if psnr > best_psnr:
                best_psnr = psnr
                save(os.path.join(t.output_dir, 'best.pth'), epoch, full=False)
        save(last_path, epoch)
        if epoch % t.ckpt_every == 0:
            save(os.path.join(t.output_dir, f'epoch_{epoch}.pth'), epoch)
    save(os.path.join(t.output_dir, 'final.pth'), t.epochs, full=False)


if __name__ == '__main__':
    main()
