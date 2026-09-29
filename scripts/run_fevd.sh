#!/usr/bin/env bash
# End-to-end EvBS on FEVD: source detection, intrinsic/extrinsic synthesis, MAENet fine-tuning.
# Usage: bash scripts/run_fevd.sh [extra --set overrides for the synthesis config]
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/synthesis/detect_sources.py       --config configs/synthesis/fevd.yaml "$@"
python scripts/synthesis/synthesize_intrinsic.py --config configs/synthesis/fevd.yaml "$@"
python scripts/synthesis/synthesize_extrinsic.py --config configs/synthesis/fevd.yaml "$@"
python scripts/finetune/maenet.py                --config configs/finetune/maenet_fevd.yaml
