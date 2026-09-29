"""Intrinsic-blur Condition Generator (InCG, Sec. 3.2 / Fig. 4)."""
from ..events import Events, event_mask
from ..feda import feda
from ..flow import crop, flow_map, unit_magnitude


def intrinsic_frames(frame, radius=2):
    """Frames needed for a content source at `frame`: the window [t - r, t + r] plus t - r - 1,
    whose events give the backward flow of the first event group."""
    return list(range(frame - radius - 1, frame + radius + 1))


class IntrinsicConditionGenerator:
    """Builds (F_I, E_I, M) for a content source from its own events in [t - r, t + r].

    Following the ID-Blau training convention, a window of frames n_0..n_K uses the event groups
    of n_0..n_{K-1} (one per frame interval) and the K bidirectional flows between them.
    """

    def __init__(self, dataset, flows, radius=2, alpha=0.1):
        self.dataset, self.flows, self.radius, self.alpha = dataset, flows, radius, alpha

    def __call__(self, video, frame, bbox):
        r = self.radius
        window = list(range(frame - r, frame + r + 1))
        groups = window[:-1]
        fwd_pairs = [(n, n + 1) for n in groups]
        bwd_pairs = [(n + 1, n) for n in groups]
        prev = [(groups[0], groups[0] - 1)] if frame - r - 1 in self.dataset.frames(video) else []
        flows = self.flows.interframe_many(video, fwd_pairs + bwd_pairs + prev)
        fwd, bwd = flows[:len(groups)], flows[len(groups):2 * len(groups)]

        F_full = flow_map(fwd, bwd)
        # F_{n -> n-1} for each group: the previous group's backward flow (or the extra one for n_0).
        back_of_group = ([flows[-1]] if prev else [None]) + bwd[:-1]

        events = [self.dataset.events(video, n).crop(bbox) for n in groups]
        h, w = bbox[2] - bbox[0], bbox[3] - bbox[1]
        E = feda(events, [crop(f, bbox) for f in fwd],
                 [None if b is None else crop(b, bbox) for b in back_of_group], h, w, self.alpha)
        M = event_mask(Events.cat(events).to(E.device), h, w)
        return {'flow': unit_magnitude(crop(F_full, bbox)), 'mask': M, 'feda': E}
