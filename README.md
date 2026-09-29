# EvBS: Event-guided Blur Synthesis for Domain-adaptive Motion Deblurring

Official code for **EvBS** (ACM MM 2026) · [arXiv:2608.08066](https://arxiv.org/abs/2608.08066) ·
[DOI 10.1145/3767308.3835301](https://doi.org/10.1145/3767308.3835301)

EvBS adapts a pre-trained deblurring model to a target domain by synthesizing blur–sharp training
pairs from the target data itself. Events decouple motion from content, so a pseudo-sharp patch can
be blurred with its own motion (**intrinsic blur**) or with motion transferred from another patch
(**extrinsic blur**). The synthesized pairs are then used to fine-tune the deblurring model.

| Component | Paper | Code |
|---|---|---|
| Dual Source Extractor (content and motion sources) | Sec. 3.1, Fig. 3, Supp. B | `evbs/synthesis/dse.py` |
| Intrinsic-blur Condition Generator (InCG) | Sec. 3.2, Fig. 4 | `evbs/synthesis/incg.py` |
| Extrinsic-blur Condition Generator (ExCG) | Sec. 3.2, Fig. 5, Alg. 1 | `evbs/synthesis/excg.py` |
| Flow-Enhanced Deviation Accumulation (FEDA) | Supp. A, Alg. S1 | `evbs/feda.py` |
| Blur synthesis with the retrained ID-Blau D_ψ | Sec. 3.3, Eq. 8 | `evbs/synthesis/blurring.py` |
| Retraining D_ψ on GoPro | Sec. 3.2, 4.1 | `scripts/blurring_model/` |
| Fine-tuning a deblurring model (MAENet) | Sec. 4.1 | `scripts/finetune/maenet.py` |

## Repository layout

```
configs/
  synthesis/fevd.yaml          blur synthesis on FEVD (REVD)
  finetune/maenet_fevd.yaml    MAENet fine-tuning and evaluation
  blurring_model/gopro.yaml    retraining ID-Blau on GoPro
evbs/
  events.py data.py flow.py    event streams, datasets, E-RAFT flows (cached)
  feda.py                      FEDA (condition of the blurring model)
  da.py                        deviation accumulation (event input of MAENet)
  synthesis/                   DSE, InCG, ExCG, blurring
  blurring_model/              GoPro training data for ID-Blau
  finetune/                    synthesized-pair and evaluation datasets
  models/                      E-RAFT, RAFT, BME, ID-Blau, MAENet
scripts/
  download_checkpoints.py
  synthesis/  finetune/  run_fevd.sh
  blurring_model/              GoPro event grouping, condition preparation, training
assets/gopro/                  GoPro blurry-frame metadata
tests/
```

Paths in the configs are relative to the repository root: datasets under `data/`, weights under
`checkpoints/`, results under `outputs/` (symbolic links work).

## Installation

Tested with Python 3.9, PyTorch 2.0.1 and CUDA 11.8.

```bash
pip install -r requirements.txt
```

## Pretrained models

| File | Model | Needed for | Download |
|---|---|---|---|
| `bme_events.pth` | Blur Magnitude Estimator (RGB + event count map), retrained for EvBS | source detection | [release v1.0](https://github.com/Juns1105/EvBS/releases/download/v1.0/bme_events.pth) |
| `idblau_evbs.pth` | ID-Blau D_ψ with flow-map + FEDA conditions, retrained on GoPro | blur synthesis | [release v1.0](https://github.com/Juns1105/EvBS/releases/download/v1.0/idblau_evbs.pth) |
| `eraft_dsec.tar` | E-RAFT trained on DSEC | blur synthesis (event flow) | [E-RAFT](https://github.com/uzh-rpg/E-RAFT) |
| `raft-things.pth` | RAFT trained on FlyingThings | retraining ID-Blau only | [RAFT](https://github.com/princeton-vl/RAFT) |
| `maenet_gopro.pth` | MAENet trained on GoPro | fine-tuning only | manual, see below |

```bash
python scripts/download_checkpoints.py      # all except MAENet, into checkpoints/, with SHA-256 check
```

The EvBS weights can also be downloaded directly:

```bash
mkdir -p checkpoints
wget -P checkpoints https://github.com/Juns1105/EvBS/releases/download/v1.0/bme_events.pth
wget -P checkpoints https://github.com/Juns1105/EvBS/releases/download/v1.0/idblau_evbs.pth
```

`idblau_evbs.pth` was trained with `configs/blurring_model/gopro.yaml`
(see [Retraining the blurring model](#retraining-the-blurring-model)).

The GoPro-pretrained MAENet is not redistributed here: download the GoPro model from the
[official MAENet repository](https://github.com/ZhijingS/DA_event_deblur) (Baidu Netdisk, extraction code
`xeuh`), save it as `checkpoints/maenet_gopro.pth` and run `download_checkpoints.py` again to verify it.

## Data

### FEVD (target domain)

The real-world event-based deblurring dataset of [FEVD](https://sites.google.com/view/fevd-cvpr2024)
(Kim et al., CVPR 2024), the REVD benchmark of the paper, 768×1024. Download the
[test split](https://drive.google.com/drive/folders/11a5NG4RwmbVxlSdxiU0MGrF46U6Lj09b) (8 videos, used
for synthesis, fine-tuning and evaluation; the
[train split](https://drive.google.com/drive/folders/1nu84bu_rLnUz-HQZirtlAqd0fMFvcm6K) is not needed)
and keep its folder names:

```
data/FEVD/test/<video>/blur_down/00000.png ...          blurry frames
data/FEVD/test/<video>/warped_events/00000.npz ...      events of each frame (x, y, t, p), aligned to the frames
data/FEVD/test/<video>/gt_down_corrected/00000.png ...  ground truth, used for evaluation only
```

### GoPro (only to retrain the blurring model)

[GOPRO_Large.zip](https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip) (blurry
frames) and [GOPRO_Large_all.zip](https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large_all.zip)
(all 240 fps frames) from the [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro), unpacked to
`data/GOPRO_Large` and `data/GOPRO_Large_all`.

The synthetic events are simulated at the original 1280×720 resolution with the data preparation of
[AFB](https://github.com/Juns1105/AFB) (`tools/data_preparation`), frame interpolation of
[rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e) and ESIM with contrast thresholds drawn per video from
N(0.2, 0.03²), and then grouped by 240 fps frame interval:

```bash
# 1. rpg_vid2e input layout (<video>/imgs, fps.txt), at the original resolution
python AFB/tools/data_preparation/downsample_gopro.py --input_dir data/GOPRO_Large_all \
    --output_dir GOPRO_frames --width 1280 --height 720
# 2. adaptive frame interpolation, inside rpg_vid2e (for split in train test)
python upsampling/upsample.py --input_dir GOPRO_frames/$split --output_dir GOPRO_upsampled/$split
# 3. event simulation
python AFB/tools/data_preparation/generate_events.py --input_dir GOPRO_upsampled --output_dir GOPRO_esim
# 4. events between consecutive 240 fps frames -> data/GOPRO_events/<split>/<video>/events/<frame>.npz
python scripts/blurring_model/integrate_gopro_events.py --frames_dir data/GOPRO_Large_all \
    --upsampled_dir GOPRO_upsampled --esim_dir GOPRO_esim --output_dir data/GOPRO_events
```

## Usage

### Blur synthesis

```bash
python scripts/synthesis/detect_sources.py       --config configs/synthesis/fevd.yaml
python scripts/synthesis/synthesize_intrinsic.py --config configs/synthesis/fevd.yaml
python scripts/synthesis/synthesize_extrinsic.py --config configs/synthesis/fevd.yaml
```

Config entries can be overridden (`--set dse.ratio=0.2 extrinsic.top_n=3`) and `--videos` restricts
a run to some videos. Outputs in `outputs/synthesis/FEVD/`:

```
sources.json                                   content and motion sources per video
intrinsic/test/<video>/sharp/<frame>_<y1>_<x1>_<y2>_<x2>.png     pseudo-sharp content source
intrinsic/test/<video>/reblur/<same name>.png                   intrinsic blur
extrinsic/test/<video>/sharp/<content>.png
extrinsic/test/<video>/transblur/<content>_<motion>_mcs<MCS>_deg<angle>.png
extrinsic/test/<video>/translated_event_combined/<same name>.npz    warped events (sensor coordinates)
intrinsic/manifest.json, extrinsic/manifest.json
cache/flows/                                   E-RAFT flows (float16), shared by all steps
```

### Fine-tuning and evaluation

```bash
python scripts/finetune/maenet.py --config configs/finetune/maenet_fevd.yaml
```

The GoPro-pretrained MAENet is fine-tuned on the synthesized pairs, with the pseudo-sharp content
sources as targets (10 epochs, Charbonnier loss, Adam, cosine learning rate 1e-5 → 1e-7), and
evaluated on all real frames of the target set. Events are converted with deviation accumulation
(Algorithm 1 of MAENet, `evbs/da.py`). `outputs/finetune/maenet_fevd/metrics.json` holds PSNR/SSIM
overall and per video.

`bash scripts/run_fevd.sh` runs synthesis and fine-tuning end to end.

### Retraining the blurring model

```bash
python scripts/blurring_model/prepare_gopro.py --config configs/blurring_model/gopro.yaml
python scripts/blurring_model/train.py         --config configs/blurring_model/gopro.yaml
```

For every GoPro blurry frame averaging sharp frames f_0..f_{N-1}, `prepare_gopro.py` computes the flow
map (RAFT on the sharp frames), the event mask and FEDA over the event groups of f_0..f_{N-2}, and
reports the normalization statistics in `stats.json`.

## Citation

```bibtex
@inproceedings{jung2026evbs,
  title     = {EvBS: Event-guided Blur Synthesis for Domain-adaptive Motion Deblurring},
  author    = {Jung, Junsik and Choi, Seokryun and Cho, Yoonki and Kim, Woo Jae and Jeong, Andrew and Yoon, Sung-Eui},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3835301}
}
```

## Acknowledgements

This code builds on [ID-Blau](https://github.com/plusgood-steven/ID-Blau),
[DADeblur](https://github.com/Jin-Ting-He/DADeblur) and its [BME](https://github.com/Jin-Ting-He/BME),
[E-RAFT](https://github.com/uzh-rpg/E-RAFT) (MIT, `evbs/models/eraft/LICENSE`),
[RAFT](https://github.com/princeton-vl/RAFT) (BSD-3-Clause, `evbs/models/raft/LICENSE`) and
[MAENet](https://github.com/ZhijingS/DA_event_deblur) (MIT, network in `evbs/models/maenet.py`).

## License

This project is released under the [MIT License](LICENSE). Third-party code keeps its original license.
