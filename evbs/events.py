"""Event stream container and dense event representations."""
import numpy as np
import torch


class Events:
    """A time-sorted event stream on one device.

    x, y: int64 pixel coordinates; t: int64 timestamps (us); p: float32 polarity in {-1, +1}.
    """

    __slots__ = ('x', 'y', 't', 'p')

    def __init__(self, x, y, t, p):
        self.x, self.y, self.t, self.p = x, y, t, p

    @classmethod
    def empty(cls, device='cpu'):
        z = torch.zeros(0, dtype=torch.int64, device=device)
        return cls(z, z.clone(), z.clone(), torch.zeros(0, dtype=torch.float32, device=device))

    @classmethod
    def cat(cls, streams):
        streams = list(streams)
        return cls(*(torch.cat([getattr(s, k) for s in streams]) for k in cls.__slots__))

    def __len__(self):
        return self.t.numel()

    @property
    def device(self):
        return self.t.device

    def to(self, device):
        return Events(*(getattr(self, k).to(device) for k in self.__slots__))

    def select(self, index):
        return Events(*(getattr(self, k)[index] for k in self.__slots__))

    def in_box(self, bbox):
        y1, x1, y2, x2 = bbox
        return (self.x >= x1) & (self.x < x2) & (self.y >= y1) & (self.y < y2)

    def crop(self, bbox):
        """Keep events inside bbox = (y1, x1, y2, x2) and shift them to patch coordinates."""
        ev = self.select(self.in_box(bbox))
        ev.x = ev.x - bbox[1]
        ev.y = ev.y - bbox[0]
        return ev

    def uncrop(self, bbox):
        return Events(self.x + bbox[1], self.y + bbox[0], self.t, self.p)

    def split_median(self):
        """Split at the median timestamp: (t <= median, t > median)."""
        if len(self) == 0:
            return self, self
        median = torch.median(self.t)
        return self.select(self.t <= median), self.select(self.t > median)

    def split_time(self, num_splits):
        """Split into equal-duration segments (the last segment includes its right edge)."""
        t = self.t.double()
        edges = torch.linspace(float(t.min()), float(t.max()), num_splits + 1, dtype=torch.float64)
        segments = []
        for i in range(num_splits):
            upper = t <= edges[i + 1] if i == num_splits - 1 else t < edges[i + 1]
            segments.append(self.select((t >= edges[i]) & upper))
        return segments

    def numpy(self, polarity01=False):
        p = self.p.cpu().numpy()
        if polarity01:
            p = (p > 0).astype(np.int8)
        return {'x': self.x.cpu().numpy(), 'y': self.y.cpu().numpy(), 't': self.t.cpu().numpy(),
                'p': p.astype(np.int8)}


def load_events(path, height, width, device='cpu'):
    """Load an .npz event file with keys x, y, t, p.

    Sub-pixel coordinates are rounded to the nearest pixel and events outside the sensor are dropped.
    Polarity {0, 1} or {-1, 1} is mapped to {-1, +1}.
    """
    with np.load(path) as data:
        x = np.round(data['x']).astype(np.int64)
        y = np.round(data['y']).astype(np.int64)
        t = data['t'].astype(np.int64)
        p = np.where(data['p'] > 0, 1.0, -1.0).astype(np.float32)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]
    if t.size > 1 and np.any(np.diff(t) < 0):
        order = np.argsort(t, kind='stable')
        x, y, t, p = x[order], y[order], t[order], p[order]
    return Events(*(torch.from_numpy(a) for a in (x, y, t, p))).to(device)


def voxel_grid(ev, num_bins, height, width, normalize=True):
    """Voxel grid with bilinear interpolation in time (E-RAFT / E2VID convention)."""
    voxel = torch.zeros(num_bins * height * width, dtype=torch.float32, device=ev.device)
    if len(ev) > 0:
        t = ev.t.double()
        dt = t[-1] - t[0]
        ts = (num_bins - 1) * (t - t[0]) / (dt if dt > 0 else 1.0)
        ti = torch.floor(ts)
        frac = (ts - ti).float()
        ti = ti.long()
        base = ev.x + ev.y * width
        for offset, weight in ((0, 1.0 - frac), (1, frac)):
            b = ti + offset
            valid = (b >= 0) & (b < num_bins)
            voxel.index_add_(0, base[valid] + b[valid] * height * width, ev.p[valid] * weight[valid])
    voxel = voxel.view(num_bins, height, width)

    if normalize:
        nz = voxel != 0
        if nz.any():
            values = voxel[nz]
            std = values.std() if values.numel() > 1 else values.new_tensor(0.)
            voxel[nz] = (values - values.mean()) / std if std > 0 else values - values.mean()
    return voxel


def count_map(ev, height, width):
    counts = torch.bincount(ev.y * width + ev.x, minlength=height * width)
    return counts.view(height, width).float()


def event_mask(ev, height, width):
    """Binary mask M: 1 where at least one event occurred."""
    return (count_map(ev, height, width) > 0).float()
