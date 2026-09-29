"""Conditional diffusion blurring model D_psi (ID-Blau, Wu et al., CVPR 2024).

EvBS retrains ID-Blau with a 7-channel condition: sharp image C (3), flow map F (3) and FEDA
representation E (1). The UNet therefore receives 3 (noisy blur) + 7 (condition) = 10 channels.
"""
import math

import numpy as np
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -(math.log(10000) / half_dim))
        emb = torch.outer(x, emb)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

    def forward(self, x, time_emb):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x, time_emb):
        return self.upsample(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, time_dim=None, activatedfun=nn.SiLU):
        super().__init__()
        self.time_dim = time_dim
        self.layer1 = nn.Sequential(activatedfun(), nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.layer2 = nn.Sequential(
            activatedfun(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        if time_dim:
            self.time_layer = nn.Sequential(activatedfun(), nn.Linear(time_dim, out_channels))
        self.shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x, time_emb=None):
        output = self.layer1(x)
        if self.time_dim:
            output += self.time_layer(time_emb)[:, :, None, None]
        return self.layer2(output) + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, img_channels=10, base_channels=64, channel_mults=(1, 2, 3), num_res_blocks=2,
                 time_dim=256, activatedfun=nn.SiLU, dropout=0.0):
        super().__init__()
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downblocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResidualBlock(now_channels, out_channels, dropout,
                                                     time_dim=time_dim, activatedfun=activatedfun))
                now_channels = out_channels
                channels.append(now_channels)
            if i != len(channel_mults) - 1:
                self.downblocks.append(Downsample(now_channels))
                channels.append(now_channels)

        self.mid = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, dropout, time_dim=time_dim, activatedfun=activatedfun),
            ResidualBlock(now_channels, now_channels, dropout, time_dim=time_dim, activatedfun=activatedfun),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResidualBlock(channels.pop() + now_channels, out_channels, dropout,
                                                   time_dim=time_dim, activatedfun=activatedfun))
                now_channels = out_channels
            if i != 0:
                self.upblocks.append(Upsample(now_channels))
        assert len(channels) == 0

        self.last_layer = nn.Sequential(activatedfun(), nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, x, time):
        time_emb = self.time_embedding(time)
        x = self.init_conv(x)
        skips = [x]
        for layer in self.downblocks:
            x = layer(x, time_emb)
            skips.append(x)
        for layer in self.mid:
            x = layer(x, time_emb)
        for layer in self.upblocks:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb)
        return self.last_layer(x)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def _extract(a, t, x_shape):
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def linear_betas(num_timesteps, beta_1, beta_T):
    return torch.linspace(beta_1, beta_T, num_timesteps).double()


class ConditionalDDIM(nn.Module):
    """Epsilon-prediction diffusion trained with DDPM noise and sampled with DDIM."""

    def __init__(self, model, betas, criterion='l1'):
        super().__init__()
        self.model = model
        self.num_timesteps = len(betas)
        if criterion == 'l1':
            self.criterion = CharbonnierLoss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("criterion must be 'l1' or 'l2'")

        betas = np.asarray(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        def buf(name, value):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

        # Buffer names are kept identical to the original ID-Blau checkpoints.
        buf('betas', betas)
        buf('alphas', alphas)
        buf('alphas_cumprod', alphas_cumprod)
        buf('sqrt_alphas_cumprod', np.sqrt(alphas_cumprod))
        buf('sqrt_one_minus_alphas_cumprod', np.sqrt(1 - alphas_cumprod))
        buf('reciprocal_sqrt_alphas', np.sqrt(1 / alphas))
        buf('remove_noise_coeff', betas / np.sqrt(1 - alphas_cumprod))
        buf('sigma', np.sqrt(betas))

    def forward(self, x, condition):
        """Training loss for target blur x (in [-1, 1]) given the condition."""
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        x_t = (_extract(self.sqrt_alphas_cumprod, t, x.shape) * x
               + _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise)
        pred_noise = self.model(torch.cat([x_t, condition], dim=1), t)
        return self.criterion(pred_noise, noise)

    @torch.no_grad()
    def sample(self, condition, sample_timesteps=20, eta=0.0, noise=None):
        """DDIM sampling. Returns the predicted blur in [-1, 1] (unclamped)."""
        b, _, h, w = condition.shape
        device = condition.device
        seq = np.arange(0, self.num_timesteps, self.num_timesteps // sample_timesteps) + 1
        prev_seq = np.append(np.array([0]), seq[:-1])
        x = noise if noise is not None else torch.randn((b, 3, h, w), device=device)

        for i in reversed(range(sample_timesteps)):
            t = torch.full((b,), int(seq[i]), device=device, dtype=torch.long)
            t_prev = torch.full((b,), int(prev_seq[i]), device=device, dtype=torch.long)
            a_t = _extract(self.alphas_cumprod, t, x.shape)
            a_prev = _extract(self.alphas_cumprod, t_prev, x.shape)

            pred_noise = self.model(torch.cat([x, condition], dim=1), t)
            pred_x0 = (x - torch.sqrt(1. - a_t) * pred_noise) / torch.sqrt(a_t)
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
            x = (torch.sqrt(a_prev) * pred_x0 + torch.sqrt(1 - a_prev - sigma ** 2) * pred_noise
                 + sigma * torch.randn_like(x))
        return x


def _strip_module_prefix(state_dict):
    return {k.replace('model.module.', 'model.', 1): v for k, v in state_dict.items()}


def build_idblau(num_timesteps=2000, beta_1=1e-6, beta_T=1e-2, base_channels=64,
                 channel_mults=(1, 2, 3), num_res_blocks=2, time_dim=256, dropout=0.0, criterion='l1'):
    net = UNet(img_channels=10, base_channels=base_channels, channel_mults=tuple(channel_mults),
               num_res_blocks=num_res_blocks, time_dim=time_dim, dropout=dropout)
    return ConditionalDDIM(net, linear_betas(num_timesteps, beta_1, beta_T), criterion=criterion)


def load_idblau(path, device='cuda'):
    """Load a checkpoint with {'model_state', 'hparams'} (released / scripts/blurring_model/train.py)
    or one written by the original ID-Blau trainer (pickled argparse namespace)."""
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    hp = ckpt.get('hparams')
    if hp is None:  # original trainer pickles its argparse namespace
        a = ckpt['args']
        hp = dict(num_timesteps=a.num_timesteps, beta_1=a.beta_1, beta_T=a.beta_T,
                  base_channels=a.base_channels, channel_mults=tuple(a.channel_mults),
                  num_res_blocks=getattr(a, 'num_res_blocks', 2), time_dim=a.time_dim, dropout=a.dropout)
    model = build_idblau(**hp)
    model.load_state_dict(_strip_module_prefix(ckpt['model_state']))
    return model.to(device).eval()
