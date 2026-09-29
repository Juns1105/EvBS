import json
import os
import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patch_name(frame, bbox, digits=5):
    """'<frame>_<y1>_<x1>_<y2>_<x2>', the naming used by the fine-tuning data loaders."""
    return '_'.join([str(frame).zfill(digits)] + [str(int(v)) for v in bbox])


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)
