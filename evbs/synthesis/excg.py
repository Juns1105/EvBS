"""Extrinsic-blur Condition Generator (ExCG, Sec. 3.2 / Fig. 5 / Algorithm 1)."""
import torch

from ..events import Events, event_mask
from ..feda import feda
from ..flow import angle_deg, flow_map, mean_motion, mean_orientation, unit_magnitude


def transfer_motion(prev, cur, nxt, local_motion, global_motion):
    """Algorithm 1: warp the content events so that they follow the motion source's global motion.

    prev, cur, nxt: time-sorted Events of frames t-1, t, t+1 (sensor coordinates).
    local_motion[k]: V_C^k (2, H, W) for k in {-1, +1}; global_motion[k]: v_M^k (2,).
    Events in [t-1, t_med] are warped with direction k=-1 and those in (t_med, t+1] with k=+1,
    scaled by a temporal weight that is 0 at t_med and 1 at the far end of the interval.
    Returns the warped (prev, cur, nxt) with rounded coordinates.
    """
    first, second = cur.split_median()
    warped = {}
    for k, parts in ((-1, (prev, first)), (1, (second, nxt))):
        seq = Events.cat(parts)
        if len(seq) == 0:
            warped[k] = [p for p in parts]
            continue
        t = seq.t.double()
        s = ((t - t.min()) / max(float(t.max() - t.min()), 1e-12)).float()
        w = (1 - s if k == -1 else s).unsqueeze(1)
        T = (global_motion[k][None] - local_motion[k][:, seq.y, seq.x].T) * w
        moved = Events(torch.round(seq.x + T[:, 0]).long(), torch.round(seq.y + T[:, 1]).long(), seq.t, seq.p)
        n0 = len(parts[0])
        warped[k] = [moved.select(slice(0, n0)), moved.select(slice(n0, None))]
    return warped[-1][0], Events.cat([warped[-1][1], warped[1][0]]), warped[1][1]


class ExtrinsicConditionGenerator:
    def __init__(self, dataset, flows, estimator, cfg, alpha=0.1, patch_batch=16):
        self.dataset, self.flows, self.estimator, self.cfg = dataset, flows, estimator, cfg
        self.alpha, self.patch_batch = alpha, patch_batch

    def content(self, video, frame, bbox):
        """Content-source context: raw events of t-1..t+1, local motion V_C^k and orientation theta_C."""
        f_prev, f_next, b_cur, b_next = self.flows.interframe_many(
            video, [(frame - 1, frame), (frame, frame + 1), (frame, frame - 1), (frame + 1, frame)])
        theta, _ = mean_motion(flow_map([f_prev, f_next], [b_cur, b_next]), bbox, self.cfg.gate_min_mag)
        return {
            'video': video, 'frame': frame, 'bbox': tuple(bbox), 'theta': theta,
            'events': [self.dataset.events(video, f) for f in (frame - 1, frame, frame + 1)],
            'local': {1: (f_next - b_next) / 2, -1: (b_cur - f_prev) / 2},   # (F_{t->t+k} - F_{t+k->t}) / 2
        }

    def motion(self, video, frame, bbox):
        """Motion-source context from flows between three segments s_{-1}, s_0, s_1 of the frame's events."""
        (f0, f1), (b0, b1) = self.flows.segment(video, frame, 3)   # f0: s-1->s0, f1: s0->s1, b0: s0->s-1, b1: s1->s0
        F_M = flow_map([f0, f1], [b0, b1])
        theta, _ = mean_motion(F_M, bbox, self.cfg.gate_min_mag)
        global_motion = {}
        for k, V in ((1, (f1 - b1) / 2), (-1, (b0 - f0) / 2)):
            direction, magnitude = mean_motion(V, bbox, self.cfg.gate_min_mag)
            global_motion[k] = (torch.zeros(2, device=V.device) if direction is None
                                else direction * magnitude)                          # v_M^k (Eq. 6)
        return {'video': video, 'frame': frame, 'bbox': tuple(bbox), 'theta': theta,
                'theta_mcs': mean_orientation(F_M, bbox, self.cfg.mcs_min_mag), 'global': global_motion}

    @staticmethod
    def angle(content, motion):
        if content['theta'] is None or motion['theta'] is None:
            return None
        return angle_deg(content['theta'], motion['theta'])

    @torch.no_grad()
    def transfer(self, content, motions):
        """Warp the content events with each motion source and re-estimate flows on the patch.

        Returns one dict per motion source with warped patch events, flows and the motion
        consistency score MCS = (theta~_C . theta_M + 1) / 2 (None when undefined).
        """
        bbox = content['bbox']
        h, w = bbox[2] - bbox[0], bbox[3] - bbox[1]
        results = []
        for m in motions:
            warped = transfer_motion(*content['events'], content['local'], m['global'])
            groups = [ev.crop(bbox) for ev in warped]
            results.append({'motion': m, 'groups': groups, 'valid': all(len(g) > 0 for g in groups)})

        valid = [r for r in results if r['valid']]
        if valid:
            vox = [torch.stack([self.estimator.voxel(g, h, w) for g in r['groups']]) for r in valid]
            v1 = torch.cat([torch.stack([v[0], v[1], v[2], v[1]]) for v in vox])
            v2 = torch.cat([torch.stack([v[1], v[2], v[1], v[0]]) for v in vox])
            flows = self.estimator.from_voxels(v1, v2, self.patch_batch).view(len(valid), 4, 2, h, w)
            for r, (f_m1, f_0, b_1, b_0) in zip(valid, flows):
                r['fwd'], r['bwd'] = [f_m1, f_0], [b_1, b_0]      # F(-1->0), F(0->1) | F(1->0), F(0->-1)
                r['flow'] = flow_map(r['fwd'], r['bwd'])
                theta_c = mean_orientation(r['flow'], None, self.cfg.mcs_min_mag)
                theta_m = r['motion']['theta_mcs']
                r['mcs'] = (None if theta_c is None or theta_m is None
                            else (float(torch.dot(theta_c, theta_m)) + 1) / 2)
        return [r for r in results if r['valid'] and r['mcs'] is not None]

    def condition(self, result, bbox):
        """(F_E, E_E, M~) for a transferred candidate; FEDA/mask use the groups of t-1 and t."""
        h, w = bbox[2] - bbox[0], bbox[3] - bbox[1]
        g_prev, g_cur, _ = result['groups']
        f_m1, f_0 = result['fwd']
        _, b_0 = result['bwd']
        E = feda([g_prev, g_cur], [f_m1, f_0], [None, b_0], h, w, self.alpha)
        M = event_mask(Events.cat([g_prev, g_cur]).to(E.device), h, w)
        return {'flow': unit_magnitude(result['flow']), 'mask': M, 'feda': E}
