"""Flow-Enhanced Deviation Accumulation (FEDA, Algorithm S1).

For each event group i (the events of one frame interval, time-sorted), the quadratic flow at the
event location is

    Q(dt) = v0 * dt + a * dt^2,   v0 = (F_{i->i+1} - F_{i->i-1}) / 2,   a = (F_{i->i+1} + F_{i->i-1}) / 2

with dt in [0, 1] normalized inside the group (Q(dt) = F_{i->i+1} * dt when F_{i->i-1} is missing).
Each event contributes d = p * (1 + alpha * |Q(dt)|). Per pixel and group, the deviation
accumulation sums the running polarity sum, i.e. sum_k d_k * (n - k) for the k-th of n events.
The FEDA value is the sum over groups divided by the total event count of the pixel.

This is the representation the retrained ID-Blau was trained on (alpha = 0.1). The implementation
is fully vectorized: one stable sort over all events replaces the per-pixel Python loops of the
original code.
"""
import torch


def event_flow_magnitude(ev, flow_fwd, flow_bwd=None):
    """|Q(dt)| for every event of one group; flows are (2, H, W) in the events' coordinate frame."""
    t = ev.t.double()
    dt = ((t - t[0]) / (t[-1] - t[0] + 1e-12)).float().unsqueeze(1)
    f = flow_fwd[:, ev.y, ev.x].T
    if flow_bwd is None:
        q = f * dt
    else:
        b = flow_bwd[:, ev.y, ev.x].T
        q = 0.5 * (f - b) * dt + 0.5 * (f + b) * dt ** 2
    return torch.linalg.vector_norm(q, dim=1)


def feda(groups, flows_fwd, flows_bwd, height, width, alpha=0.1):
    """FEDA map of shape (H, W).

    groups: list of time-sorted Events (patch coordinates), one per frame interval.
    flows_fwd[i]: F_{i->i+1} (2, H, W); flows_bwd[i]: F_{i->i-1} (2, H, W) or None.
    """
    device = flows_fwd[0].device
    pix, weight, gid = [], [], []
    for i, ev in enumerate(groups):
        if len(ev) == 0:
            continue
        ev = ev.to(device)
        mag = event_flow_magnitude(ev, flows_fwd[i], flows_bwd[i])
        pix.append(ev.y * width + ev.x)
        weight.append(ev.p * (1.0 + alpha * mag))
        gid.append(torch.full_like(ev.x, i))

    HW = height * width
    if not pix:
        return torch.zeros(height, width, device=device)
    pix, weight, gid = torch.cat(pix), torch.cat(weight), torch.cat(gid)

    # Group events by (group, pixel) while keeping temporal order inside each run.
    key = gid * HW + pix
    key_sorted, order = torch.sort(key, stable=True)
    _, counts = torch.unique_consecutive(key_sorted, return_counts=True)
    run = torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts)
    start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(key.numel(), device=device) - start[run]          # k within its run
    remaining = (counts[run] - rank).to(weight.dtype)                      # n - k

    R = torch.zeros(HW, device=device).scatter_add_(0, pix[order], weight[order] * remaining)
    N = torch.bincount(pix, minlength=HW).to(R.dtype)
    return torch.where(N > 0, R / N.clamp(min=1), torch.zeros_like(R)).view(height, width)
