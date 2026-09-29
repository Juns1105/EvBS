"""Deviation Accumulation (DA) event representations: the event input of the deblurring model.

`da` follows Algorithm 1 of MAENet (Sun et al., ECCV 2024): for each pixel, D accumulates the
polarities in temporal order, R accumulates D, and V = R / N. It is computed in 6 temporal windows
centered on the exposure (3 growing to the left, 3 growing to the right).

`da_asynav` reproduces the variant in the MAENet data-preparation code (voxel_method ASYNAV):
mean of exp(0.2 * D) per pixel, left windows negated, and the voxel divided by its absolute maximum.
"""
import torch


def running_polarity(ev, width):
    """Per-event running polarity sum D at its pixel (events are time-sorted).

    Returns (pixel index, D) for the events sorted by pixel with a stable sort, so the temporal
    order is kept within each pixel.
    """
    pix_sorted, order = torch.sort(ev.y * width + ev.x, stable=True)
    _, counts = torch.unique_consecutive(pix_sorted, return_counts=True)
    run = torch.repeat_interleave(torch.arange(counts.numel(), device=pix_sorted.device), counts)
    start = torch.cumsum(counts, 0) - counts
    p = ev.p[order]
    cs = torch.cumsum(p, 0)
    return pix_sorted, cs - (cs[start] - p[start])[run]


def _pixel_mean(ev, height, width, transform):
    out = torch.zeros(height * width, device=ev.device)
    if len(ev) == 0:
        return out.view(height, width)
    pix, D = running_polarity(ev, width)
    out.scatter_add_(0, pix, transform(D))
    n = torch.bincount(pix, minlength=height * width).clamp(min=1)
    return (out / n).view(height, width)


def deviation_accumulation(ev, height, width):
    """Algorithm 1 of MAENet: V = (sum_i D_i) / N per pixel."""
    return _pixel_mean(ev, height, width, lambda D: D)


def asyn_average(ev, height, width, gamma=0.2):
    """MAENet code variant: mean over the events of a pixel of exp(gamma * D_i)."""
    return _pixel_mean(ev, height, width, lambda D: torch.exp(gamma * D))


def _time_range(ev, t_range):
    t = ev.t.double()
    if t_range is None:
        return t, float(t.min()), float(t.max())
    return t, float(t_range[0]), float(t_range[1])


def da(ev, height, width, scales=(1.0, 0.66, 0.33), t_range=None):
    """DA in 6 windows: [c - s*T/2, c] for s in scales, then [c, c + s*T/2] for s in reversed scales.

    t_range: (t_start, t_end) of the full event stream when `ev` is a spatial crop of it.
    """
    t, t0, t1 = _time_range(ev, t_range)
    c, half = (t0 + t1) / 2, (t1 - t0) / 2
    maps = [deviation_accumulation(ev.select((t >= c - half * s) & (t <= c)), height, width) for s in scales]
    maps += [deviation_accumulation(ev.select((t >= c) & (t <= c + half * s)), height, width) for s in scales[::-1]]
    return torch.stack(maps)


def da_asynav(ev, height, width, num_bins=6, normalize=True, t_range=None):
    """MAENet data-preparation variant (ASYNAV + norm_voxel). Windows [t0 + k*dt/B, t_mid) on the left
    (negated) and [t_mid, t0 + (k+1)*dt/B) on the right."""
    t, t0, t1 = _time_range(ev, t_range)
    dt, t_mid = t1 - t0, t0 + (t1 - t0) / 2
    bins = [-asyn_average(ev.select((t >= t0 + dt / num_bins * b) & (t < t_mid)), height, width)
            for b in range(num_bins // 2)]
    bins += [asyn_average(ev.select((t >= t_mid) & (t < t0 + dt / num_bins * (b + 1))), height, width)
             for b in range(num_bins // 2, num_bins)]
    vox = torch.stack(bins)
    if normalize:
        peak = vox.abs().max()
        vox = vox / peak if peak > 0 else vox
    return vox
