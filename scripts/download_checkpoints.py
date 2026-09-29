"""Download the checkpoints into checkpoints/ and verify their SHA-256.

EvBS checkpoints (retrained BME and ID-Blau) are hosted on the Hugging Face Hub; third-party
checkpoints are fetched from their official locations. MAENet is not redistributed: its official
repository only offers Baidu Netdisk links, so it has to be downloaded manually.

    python scripts/download_checkpoints.py                 # everything available
    python scripts/download_checkpoints.py --only bme_events.pth idblau_evbs.pth
"""
import argparse
import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile

HF_REPO = ''   # TODO: Hugging Face model repository (<user>/<repo>) holding the EvBS weights
HF_URL = 'https://huggingface.co/{repo}/resolve/main/{file}'

CHECKPOINTS = {
    'bme_events.pth': {
        'what': 'Blur Magnitude Estimator retrained with RGB + event count map (EvBS)',
        'hf': True,
        'sha256': '3e075117aa4f111e21d3cabab8ebc37c704f5ab7f4e091a78727a837390e851a',
    },
    'idblau_evbs.pth': {
        'what': 'ID-Blau blurring model retrained with flow-map + FEDA conditions (EvBS)',
        'hf': True,
        'sha256': 'a2ddceed9b6903e3a89c41907c3ce273018e67d4ee29d8f940085e77c4ca9f56',
    },
    'eraft_dsec.tar': {
        'what': 'E-RAFT trained on DSEC (Gehrig et al., 3DV 2021)',
        'url': 'https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/dsec.tar',
        'sha256': '82dc19a46f7a3b21121be7e99079ff6a6d14b1525ca74264be86ca93285bae49',
    },
    'raft-things.pth': {
        'what': 'RAFT trained on FlyingThings (Teed and Deng, ECCV 2020); only needed to retrain ID-Blau',
        'url': 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip',
        'zip_member': 'raft-things.pth',
        'sha256': 'fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1',
    },
    'maenet_gopro.pth': {
        'what': 'MAENet trained on GoPro (Sun et al., ECCV 2024); only needed for fine-tuning',
        'manual': 'the GoPro model of the official repository https://github.com/ZhijingS/DA_event_deblur '
                  '(Baidu Netdisk https://pan.baidu.com/s/1HiS0Bo03gu06FAvnLEgd8w?pwd=xeuh, code: xeuh)',
        'sha256': '662463e6c7599dca864ac25c650d872a78585472a36240401993d68c513ce7a9',
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url):
    print(f'  downloading {url}')
    with urllib.request.urlopen(url) as r:
        return r.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', default='checkpoints')
    parser.add_argument('--only', nargs='*', default=None, choices=sorted(CHECKPOINTS))
    parser.add_argument('--hf-repo', default=HF_REPO)
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    missing = []
    for name, info in CHECKPOINTS.items():
        if args.only and name not in args.only:
            continue
        path = os.path.join(args.dir, name)
        print(f'{name}: {info["what"]}')
        if not os.path.exists(path):
            if 'manual' in info:
                print(f'  download manually {info["manual"]} and save it as {path}')
                missing.append(name)
                continue
            if info.get('hf') and not args.hf_repo:
                print('  no Hugging Face repository configured (HF_REPO / --hf-repo)')
                missing.append(name)
                continue
            data = fetch(HF_URL.format(repo=args.hf_repo, file=name) if info.get('hf') else info['url'])
            if 'zip_member' in info:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    member = next(m for m in z.namelist() if m.endswith(info['zip_member']))
                    data = z.read(member)
            tmp = path + '.part'
            with open(tmp, 'wb') as f:
                f.write(data)
            shutil.move(tmp, path)
        ok = sha256(path) == info['sha256']
        print(f'  {"ok" if ok else "SHA-256 MISMATCH"}: {path}')
        if not ok:
            missing.append(name)
    if missing:
        sys.exit(f'not available: {", ".join(missing)}')


if __name__ == '__main__':
    main()
