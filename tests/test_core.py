"""Unit tests that need no dataset or checkpoints: `python -m pytest tests`."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evbs.events import Events, voxel_grid  # noqa: E402
from evbs.synthesis.excg import transfer_motion  # noqa: E402
from evbs.feda import feda  # noqa: E402
from evbs.blurring_model.gopro import rotate_flow90  # noqa: E402
from reference_feda import deviation_accumulation_scatter_add_v3  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def random_events(n, h, w, t0, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, w, (n,), generator=g)
    y = torch.randint(0, h, (n,), generator=g)
    t = torch.sort(torch.randint(t0, t0 + 10000, (n,), generator=g)).values
    p = torch.where(torch.rand(n, generator=g) > 0.5, 1.0, -1.0)
    return Events(x, y, t, p).to(DEVICE)


@pytest.fixture
def stable_reference_sort(monkeypatch):
    # The original code relies on torch.sort keeping the temporal order of events at the same
    # pixel. That holds for the CUDA sort it ran on but not on CPU, so compare with a stable sort.
    import reference_feda
    unstable = torch.sort
    monkeypatch.setattr(reference_feda.torch, 'sort', lambda x, **kw: unstable(x, stable=True))


@pytest.mark.parametrize('with_prev', [True, False])
def test_feda_matches_reference(with_prev, stable_reference_sort):
    h, w = 32, 48
    groups = [random_events(3000, h, w, 10000 * i, i) for i in range(3)]
    g = torch.Generator().manual_seed(7)
    fwd = [(torch.randn(2, h, w, generator=g) * 3).to(DEVICE) for _ in range(3)]
    bwd = [(torch.randn(2, h, w, generator=g) * 3).to(DEVICE) for _ in range(3)]
    back = ([bwd[0]] if with_prev else [None]) + bwd[1:]

    ours = feda(groups, fwd, back, h, w, alpha=0.1)
    ref_bwd = [b[None] for b in back if b is not None]
    ref = deviation_accumulation_scatter_add_v3(
        [[e.x for e in groups], [e.y for e in groups], [e.t.float() for e in groups], [e.p for e in groups]],
        [f[None] for f in fwd], ref_bwd, (h, w), alpha=0.1, device=DEVICE)
    assert torch.allclose(ours, ref, atol=1e-4)


def test_deviation_accumulation_single_pixel():
    # events +1, +1, -1 at one pixel with zero flow: running sums 1, 2, 1 -> R = 4, N = 3
    ev = Events(torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long),
                torch.arange(3), torch.tensor([1.0, 1.0, -1.0])).to(DEVICE)
    zero = torch.zeros(2, 1, 1, device=DEVICE)
    out = feda([ev], [zero], [None], 1, 1, alpha=0.1)
    assert out.item() == pytest.approx(4 / 3)


@pytest.mark.parametrize('k', [0, 1, 2, 3])
def test_flow_rotation_matches_image_rotation(k):
    # a bright dot moving by (u, v) = (3, 1) px; rotating both frames must rotate the displacement
    img0, img1 = np.zeros((16, 16)), np.zeros((16, 16))
    img0[5, 4] = 1
    img1[5 + 1, 4 + 3] = 1
    r0, r1 = np.rot90(img0, k), np.rot90(img1, k)
    (y0, x0), (y1, x1) = np.argwhere(r0)[0], np.argwhere(r1)[0]
    uv = rotate_flow90(np.array([[[3.0, 1.0]]]), k)[0, 0]
    assert tuple(uv) == (x1 - x0, y1 - y0)


def test_transfer_without_motion_difference_is_identity():
    h, w = 20, 20
    prev, cur, nxt = (random_events(500, h, w, 10000 * i, i) for i in range(3))
    local = {k: torch.zeros(2, h, w, device=DEVICE) for k in (-1, 1)}
    glob = {k: torch.zeros(2, device=DEVICE) for k in (-1, 1)}
    out = transfer_motion(prev, cur, nxt, local, glob)
    for a, b in zip((prev, cur, nxt), out):
        assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y) and torch.equal(a.t, b.t)


def test_transfer_moves_far_ends_by_offset():
    h, w = 100, 100
    mk = lambda t: Events(torch.tensor([50]), torch.tensor([50]), torch.tensor([t]), torch.tensor([1.0])).to(DEVICE)
    prev = Events.cat([mk(0), mk(5)])
    cur = Events.cat([mk(10), mk(20)])
    nxt = Events.cat([mk(25), mk(30)])
    local = {k: torch.zeros(2, h, w, device=DEVICE) for k in (-1, 1)}
    glob = {-1: torch.tensor([-8.0, 0.0], device=DEVICE), 1: torch.tensor([0.0, 6.0], device=DEVICE)}
    p, c, n = transfer_motion(prev, cur, nxt, local, glob)
    # median(cur.t) = 10, so k=-1 covers t in [0, 10] and k=+1 covers t in [20, 30]
    assert p.x.tolist() == [42, 46]            # weights 1 and 0.5 -> 50 - 8, 50 - 4
    assert c.x.tolist() == [50, 50] and c.y.tolist() == [50, 50]   # weight 0 at both interval starts
    assert n.y.tolist() == [53, 56]            # weights 0.5 and 1 -> 50 + 3, 50 + 6


def test_voxel_grid_conserves_polarity_sum():
    ev = random_events(1000, 8, 8, 0, 3)
    vox = voxel_grid(ev, 5, 8, 8, normalize=False)
    assert vox.sum().item() == pytest.approx(ev.p.sum().item(), abs=1e-3)


def test_asyn_average_matches_maenet_loop():
    from evbs.da import asyn_average
    h, w = 6, 7
    ev = random_events(400, h, w, 0, 11).to('cpu')
    # per-event loop of asynevents_to_avimage_torch in the MAENet repository
    img, state, times = torch.zeros(h, w), torch.zeros(h, w).long(), torch.zeros(h, w).long()
    times.index_put_((ev.y, ev.x), torch.ones_like(ev.x), accumulate=True)
    times[times == 0] = 1
    for j in range(len(ev)):
        state[ev.y[j], ev.x[j]] += int(ev.p[j])
        img[ev.y[j], ev.x[j]] += torch.exp(state[ev.y[j], ev.x[j]] * 0.2)
    assert torch.allclose(asyn_average(ev, h, w), img / times, atol=1e-5)


def test_deviation_accumulation_algorithm1():
    from evbs.da import deviation_accumulation
    ev = Events(torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long),
                torch.arange(3), torch.tensor([1.0, 1.0, -1.0]))
    assert deviation_accumulation(ev, 1, 1).item() == pytest.approx((1 + 2 + 1) / 3)


def test_spatio_temporal_selection_skips_neighbouring_duplicates():
    from evbs.synthesis.dse import select_sources
    box, other = (0, 0, 256, 256), (0, 512, 256, 768)
    cands = [{'frame': f, 'bbox': box, 'score': 0.1 + 0.01 * f} for f in range(10)]
    cands.append({'frame': 1, 'bbox': other, 'score': 0.5})
    plain = select_sources(cands, 4, lower_better=True)
    assert [p['frame'] for p in plain] == [0, 1, 2, 3]
    st = select_sources(cands, 4, lower_better=True, frame_gap=5, iou_threshold=0.3)
    assert [(p['frame'], p['bbox']) for p in st] == [(0, box), (6, box), (1, other)]
