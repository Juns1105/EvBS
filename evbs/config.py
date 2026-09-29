"""YAML configuration with dotted command-line overrides (e.g. `--set dse.ratio=0.2`)."""
import argparse
from types import SimpleNamespace

import yaml


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    return ns


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split('=', 1)
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return _to_namespace(cfg)


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', required=True, help='YAML config file')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='override config entries')
    parser.add_argument('--videos', nargs='*', default=None, help='only process these videos')
    parser.add_argument('--device', default='cuda')
    return parser
