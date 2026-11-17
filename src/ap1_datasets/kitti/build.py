"""Build the WEAP26/KITTI_mcap dataset from KITTI raw drives.

    pixi run build-kitti [--drive 0005] [--out out/kitti]

For every drive in drives.yaml (or only --drive):
  1. download the sync zip, tracklets and the date's calibration (skipped when
     a verified copy exists) and check their sha256
  2. unzip into <out>/raw/
  3. kitti2mcap (pinned in drives.yaml, run with uvx in its own environment
     because it pulls pip OpenCV) -> <out>/mcap/<date>_drive_<NNNN>_kitti.mcap
  4. adapter.py -> <out>/hf/<date>_drive_<NNNN>/, the AP1 bag

<out>/hf/ is exactly what gets uploaded: the bags plus dataset_card.md as
README.md. Existing outputs are kept; delete them to redo a step.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import yaml

from ap1_datasets.kitti import adapter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, expected):
    """Download url to dest unless a verified copy exists."""
    if dest.exists() and sha256(dest) == expected:
        return
    print(f'downloading {url}', flush=True)
    part = dest.with_suffix(dest.suffix + '.part')
    with urllib.request.urlopen(url) as r, open(part, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 20)
    actual = sha256(part)
    if actual != expected:
        part.unlink()
        sys.exit(f'{dest.name}: sha256 {actual} does not match drives.yaml ({expected})')
    part.rename(dest)


def unzip(archive, into, marker):
    if not marker.exists():
        print(f'extracting {archive.name}', flush=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)


def kitti2mcap(spec, raw, date, drive, output):
    if output.exists():
        return
    print(f'kitti2mcap {date} drive {drive}', flush=True)
    part = output.with_suffix('.mcap.part')
    subprocess.run(['uvx', spec, '--date', date, '--drive', drive, '--kitti-dir', str(raw),
                    '--output', str(part)], check=True)
    part.rename(output)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--drive', action='append', help='drive number, e.g. 0005 (repeatable)')
    p.add_argument('--out', type=Path, default=REPO / 'out' / 'kitti')
    args = p.parse_args()

    cfg = yaml.safe_load((HERE / 'drives.yaml').read_text())
    drives = [d for d in cfg['drives'] if not args.drive or str(d['drive']) in args.drive]
    if args.drive and len(drives) != len(set(args.drive)):
        sys.exit(f'unknown drive in {args.drive}; drives.yaml has '
                 f'{[d["drive"] for d in cfg["drives"]]}')
    downloads, raw, mcaps, hf = (args.out / d for d in ('downloads', 'raw', 'mcap', 'hf'))
    for d in (downloads, raw, mcaps, hf):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / 'dataset_card.md', hf / 'README.md')
    base = cfg['base_url']

    for d in drives:
        date, drive = str(d['date']), str(d['drive'])
        name = f'{date}_drive_{drive}'
        calib = downloads / f'{date}_calib.zip'
        sync = downloads / f'{name}_sync.zip'
        tracklets = downloads / f'{name}_tracklets.zip'
        fetch(f'{base}/{calib.name}', calib, cfg['calib'][date]['sha256'])
        fetch(f'{base}/{name}/{sync.name}', sync, d['sync_sha256'])
        fetch(f'{base}/{name}/{tracklets.name}', tracklets, d['tracklets_sha256'])

        drive_dir = raw / date / f'{name}_sync'
        unzip(calib, raw, raw / date / 'calib_cam_to_cam.txt')
        unzip(sync, raw, drive_dir / 'velodyne_points')
        unzip(tracklets, raw, drive_dir / 'tracklet_labels.xml')

        mcap = mcaps / f'{name}_kitti.mcap'
        kitti2mcap(cfg['kitti2mcap'], raw, date, drive, mcap)
        bag = hf / name
        if not bag.exists():
            print(f'adapter -> {bag}', flush=True)
            labels = drive_dir / 'tracklet_labels.xml'
            adapter.main([str(mcap), str(bag), '--tracklets', str(labels)])
    print(f'done: upload {hf} with `pixi run upload-kitti`')


if __name__ == '__main__':
    main()
