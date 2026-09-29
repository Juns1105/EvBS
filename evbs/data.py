"""Target-domain dataset access.

Expected layout (FEVD / REVD, as distributed):
    <root>/<split>/<video>/blur_down/<frame:05d>.png
    <root>/<split>/<video>/warped_events/<frame:05d>.npz   (x, y, t, p)
"""
import os
from collections import OrderedDict

import cv2

from .events import load_events


class EventVideoDataset:
    def __init__(self, root, split, height, width, blur_dir='blur_down', event_dir='warped_events',
                 name_digits=5, event_cache_size=16, device='cpu'):
        self.root = os.path.join(root, split)
        self.height, self.width = height, width
        self.blur_dir, self.event_dir = blur_dir, event_dir
        self.name_digits = name_digits
        self.device = device
        self._cache = OrderedDict()
        self._cache_size = event_cache_size
        self._frames = {}

    @property
    def videos(self):
        return sorted(v for v in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, v)))

    def name(self, frame):
        return str(frame).zfill(self.name_digits)

    def image_path(self, video, frame):
        return os.path.join(self.root, video, self.blur_dir, self.name(frame) + '.png')

    def event_path(self, video, frame):
        return os.path.join(self.root, video, self.event_dir, self.name(frame) + '.npz')

    def frames(self, video):
        """Sorted frame indices that have both an image and an event file."""
        if video not in self._frames:
            images = {int(f[:-4]) for f in os.listdir(os.path.join(self.root, video, self.blur_dir))
                      if f.endswith('.png')}
            events = {int(f[:-4]) for f in os.listdir(os.path.join(self.root, video, self.event_dir))
                      if f.endswith('.npz')}
            self._frames[video] = sorted(images & events)
        return self._frames[video]

    def has_frames(self, video, frames):
        available = set(self.frames(video))
        return all(f in available for f in frames)

    def image(self, video, frame):
        """RGB uint8 image of shape (H, W, 3)."""
        img = cv2.imread(self.image_path(video, frame), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.image_path(video, frame))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def events(self, video, frame):
        key = (video, frame)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        ev = load_events(self.event_path(video, frame), self.height, self.width, self.device)
        self._cache[key] = ev
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return ev
