"""Dual Source Extractor (Sec. 3.1): content sources (pseudo-sharp patches) and motion sources."""
import numpy as np
import torch
import torch.nn.functional as F

from ..events import count_map
from ..models.bme import BlurMagnitudeEstimator


def iou(a, b):
    y1, x1 = max(a[0], b[0]), max(a[1], b[1])
    y2, x2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y2 - y1) * max(0, x2 - x1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(patches, iou_threshold, lower_better):
    patches = sorted(patches, key=lambda p: p['score'], reverse=not lower_better)
    kept = []
    while patches:
        cur = patches.pop(0)
        kept.append(cur)
        patches = [p for p in patches if iou(cur['bbox'], p['bbox']) < iou_threshold]
    return kept


def select_sources(candidates, target, lower_better, frame_gap=0, iou_threshold=0.3):
    """Take up to `target` candidates in score order.

    With frame_gap > 0, a candidate is skipped when an already selected source lies within
    frame_gap frames and overlaps it with IoU >= iou_threshold (spatio-temporal NMS), so that the
    same region is not picked again from neighboring frames.
    """
    kept = []
    for p in sorted(candidates, key=lambda p: p['score'], reverse=not lower_better):
        if len(kept) >= target:
            break
        if frame_gap > 0 and any(abs(p['frame'] - q['frame']) <= frame_gap
                                 and iou(p['bbox'], q['bbox']) >= iou_threshold for q in kept):
            continue
        kept.append(p)
    return kept


class DualSourceExtractor:
    def __init__(self, cfg, dataset, flow_estimator, bme_checkpoint, device='cuda', patch_batch=16):
        self.cfg, self.dataset, self.flow, self.device = cfg, dataset, flow_estimator, device
        self.patch_batch = patch_batch
        self.bme = BlurMagnitudeEstimator().to(device).eval()
        self.bme.load_state_dict(torch.load(bme_checkpoint, map_location='cpu', weights_only=True))
        self._coherence = {}

    @torch.no_grad()
    def score_frame(self, video, frame):
        """Blur score B = w_count * C_E / count_norm + w_mag * M * mag_scale (Eq. 1), pooled per window."""
        c, H, W = self.cfg, self.dataset.height, self.dataset.width
        counts = count_map(self.dataset.events(video, frame).to(self.device), H, W)
        rgb = torch.from_numpy(self.dataset.image(video, frame)).to(self.device).permute(2, 0, 1).float() / 255
        magnitude = self.bme(torch.cat([rgb, counts[None] / c.count_norm])[None])[0, 0]

        score = c.weight_count * counts / c.count_norm + c.weight_magnitude * magnitude * c.magnitude_scale
        k, s = c.patch_size, c.stride
        means = F.avg_pool2d(score[None, None], k, s)[0, 0].cpu().numpy()
        event_sums = (F.avg_pool2d(counts[None, None], k, s)[0, 0] * k * k).cpu().numpy()
        bboxes = [(i * s, j * s, i * s + k, j * s + k)
                  for i in range(means.shape[0]) for j in range(means.shape[1])]
        return {'means': means.ravel(), 'event_sums': event_sums.ravel(), 'bboxes': bboxes,
                'total_events': float(counts.sum())}

    def candidates(self, frame_scores, tau):
        """Per-frame sharp (B <= P_tau) and blurry (B >= P_{1-tau}) windows after NMS."""
        c = self.cfg
        means = frame_scores['means']
        sharp_thr = np.percentile(means, tau * 100)
        blur_thr = np.percentile(means, (1 - tau) * 100)
        min_events = frame_scores['total_events'] * c.min_event_ratio
        sharp, blurry = [], []
        for bbox, m, e in zip(frame_scores['bboxes'], means, frame_scores['event_sums']):
            if m <= sharp_thr and e >= min_events:
                sharp.append({'bbox': bbox, 'score': float(m)})
            if m >= blur_thr:
                blurry.append({'bbox': bbox, 'score': float(m)})
        return nms(sharp, c.nms_iou, True), nms(blurry, c.nms_iou, False)

    @torch.no_grad()
    def motion_coherence(self, video, frame, bboxes):
        """MS Selector (Supp. B): cosine between the average flow orientation and the orientation
        at the maximum-magnitude pixel, with flow estimated between two halves of the frame's events."""
        todo = [b for b in bboxes if (video, frame, b) not in self._coherence]
        if todo:
            H, W = self.dataset.height, self.dataset.width
            halves = self.dataset.events(video, frame).split_time(2)
            v_old, v_new = (self.flow.voxel(h, H, W) for h in halves)
            crops_old = torch.stack([v_old[:, y1:y2, x1:x2] for y1, x1, y2, x2 in todo])
            crops_new = torch.stack([v_new[:, y1:y2, x1:x2] for y1, x1, y2, x2 in todo])
            flows = self.flow.from_voxels(crops_old, crops_new, self.patch_batch)
            mag = torch.linalg.vector_norm(flows, dim=1, keepdim=True)
            unit = (flows / (mag + 1e-6)).flatten(2)                      # (N, 2, HW)
            avg = F.normalize(unit.mean(dim=2), dim=1)                   # (N, 2)
            peak = unit[torch.arange(len(todo)), :, mag.flatten(1).argmax(dim=1)]
            cos = F.cosine_similarity(avg, peak, dim=1)
            for b, v in zip(todo, cos.tolist()):
                self._coherence[(video, frame, b)] = v
        return [self._coherence[(video, frame, b)] for b in bboxes]

    def run_video(self, video, content_frames, log=print):
        """Returns (tau, content_sources, motion_sources) for one video.

        content_frames: frames whose temporal window allows condition generation.
        tau starts at tau_init and is increased (bisection towards tau_max) until about
        ratio * num_frames content sources are found.
        """
        c = self.cfg
        frames = self.dataset.frames(video)
        scores = {f: self.score_frame(video, f) for f in frames}
        content_frames = set(content_frames)
        target = int(len(frames) * c.ratio)

        gap = getattr(c, 'temporal_nms_frames', 0) or 0
        st_iou = getattr(c, 'temporal_nms_iou', c.nms_iou)
        tau = c.tau_init
        for it in range(c.max_iters):
            content = []
            for f in frames:
                if f in content_frames:
                    content += [dict(p, frame=f) for p in self.candidates(scores[f], tau)[0]]
            content = select_sources(content, target, True, gap, st_iou)
            log(f'  [{video}] iter {it}: tau={tau:.5f} content={len(content)}/{target}')
            if len(content) >= target - c.count_tolerance:
                break
            tau = (tau + c.tau_max) / 2

        motion = []
        for f in frames:
            blurry = self.candidates(scores[f], tau)[1]
            coherence = self.motion_coherence(video, f, [p['bbox'] for p in blurry])
            motion += [dict(p, frame=f, coherence=v) for p, v in zip(blurry, coherence)
                       if v > c.ms_cos_threshold]
        motion = select_sources(motion, target, False, gap, st_iou)
        log(f'  [{video}] motion sources: {len(motion)}/{target}')
        return tau, content, motion
