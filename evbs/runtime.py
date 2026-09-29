"""Shared setup for the pipeline scripts."""
import os

from .data import EventVideoDataset
from .flow import EventFlowEstimator, FlowStore


def make_dataset(cfg, device):
    d = cfg.dataset
    return EventVideoDataset(d.root, d.split, d.height, d.width, d.blur_dir, d.event_dir, device=device)


def make_flow(cfg, dataset, device):
    estimator = EventFlowEstimator(cfg.checkpoints.eraft, cfg.eraft.num_bins, cfg.eraft.iters,
                                   cfg.eraft.full_frame_batch, device)
    store = FlowStore(dataset, estimator, os.path.join(cfg.output_dir, 'cache', 'flows'))
    return estimator, store


def select_videos(dataset, requested):
    videos = dataset.videos
    if requested:
        unknown = set(requested) - set(videos)
        if unknown:
            raise ValueError(f'unknown videos: {sorted(unknown)}')
        videos = [v for v in videos if v in requested]
    return videos


def sources_path(cfg):
    return os.path.join(cfg.output_dir, 'sources.json')
