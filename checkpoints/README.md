# Checkpoints

`python scripts/download_checkpoints.py` fills this directory and checks the SHA-256 of every file
(`scripts/download_checkpoints.py` lists the sources).

| File | Model | Source |
|---|---|---|
| `bme_events.pth` | Blur Magnitude Estimator (RGB + event count map), retrained for EvBS | Hugging Face Hub |
| `idblau_evbs.pth` | ID-Blau blurring model with flow-map + FEDA conditions, retrained for EvBS | Hugging Face Hub |
| `eraft_dsec.tar` | E-RAFT trained on DSEC | [E-RAFT](https://github.com/uzh-rpg/E-RAFT) |
| `raft-things.pth` | RAFT (FlyingThings), for retraining ID-Blau only | [RAFT](https://github.com/princeton-vl/RAFT) |
| `maenet_gopro.pth` | MAENet trained on GoPro, for fine-tuning only | GoPro model of the [official MAENet repository](https://github.com/ZhijingS/DA_event_deblur) (Baidu Netdisk, code `xeuh`); download manually and rename |
