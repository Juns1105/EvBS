"""Group the simulated GoPro events by the 240 fps frame interval they belong to.

Input:
  <frames_dir>/<split>/<video>/<frame>.png          GOPRO_Large_all (names of the 240 fps frames)
  <upsampled_dir>/<split>/<video>/timestamps.txt    timestamps of the upsampled frames (rpg_vid2e)
  <esim_dir>/<split>/<video>/%010d.npz              ESIM events between upsampled frames j and j + 1
Output:
  <output_dir>/<split>/<video>/events/<frame>.npz   events between <frame> and the next 240 fps frame

    python scripts/blurring_model/integrate_gopro_events.py --frames_dir data/GOPRO_Large_all \\
        --upsampled_dir GOPRO_upsampled --esim_dir GOPRO_esim --output_dir data/GOPRO_events
"""
import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

FPS = 240


def frame_index(timestamps, fps=FPS):
    """240 fps interval in which each upsampled interval starts. The tolerance absorbs the float
    error of the original frames' timestamps in timestamps.txt."""
    return np.floor(np.asarray(timestamps, dtype=np.float64) * fps + 1e-3).astype(np.int64)


def integrate_video(frames, timestamps, esim_files, out_dir):
    if len(esim_files) != len(timestamps) - 1:
        raise ValueError(f'{out_dir}: {len(timestamps)} timestamps but {len(esim_files)} event files')
    k = frame_index(timestamps[:-1])
    bounds = np.searchsorted(k, np.arange(len(frames)))   # k is non-decreasing
    os.makedirs(out_dir, exist_ok=True)
    for i, frame in enumerate(frames[:-1]):
        parts = [np.load(esim_files[j]) for j in range(bounds[i], bounds[i + 1])]
        if not parts:
            raise ValueError(f'{out_dir}: no upsampled interval between frames {i} and {i + 1}')
        np.savez(os.path.join(out_dir, os.path.splitext(frame)[0] + '.npz'),
                 **{key: np.concatenate([p[key] for p in parts]) for key in ('x', 'y', 't', 'p')})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames_dir', required=True)
    parser.add_argument('--upsampled_dir', required=True)
    parser.add_argument('--esim_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'test'])
    args = parser.parse_args()

    for split in args.splits:
        for video in tqdm(sorted(os.listdir(os.path.join(args.esim_dir, split))), desc=split):
            frames = sorted(f for f in os.listdir(os.path.join(args.frames_dir, split, video)) if f.endswith('.png'))
            timestamps = np.genfromtxt(os.path.join(args.upsampled_dir, split, video, 'timestamps.txt'),
                                       dtype=np.float64)
            esim_files = sorted(glob.glob(os.path.join(args.esim_dir, split, video, '*.npz')))
            integrate_video(frames, timestamps, esim_files, os.path.join(args.output_dir, split, video, 'events'))


if __name__ == '__main__':
    main()
