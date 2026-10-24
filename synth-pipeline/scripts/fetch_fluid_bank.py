"""Fetch the Debian-packaged FluidR3 GM data and preserve its license."""
import argparse
import io
import json
import re
import tarfile
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = 'https://deb.debian.org/debian/pool/main/f/fluid-soundfont/'
    listing = urllib.request.urlopen(base, timeout=30).read().decode()
    names = sorted(set(re.findall(r'href="(fluid-soundfont-gm_[^"]+_all.deb)"', listing)))
    if not names:
        raise RuntimeError('No GM package in Debian index')
    url = base + names[-1]
    package = args.output / names[-1]
    if not package.exists():
        urllib.request.urlretrieve(url, package)
    data = package.read_bytes()
    if data[:8] != b'!<arch>\n':
        raise ValueError('Invalid Debian archive')
    offset = 8
    while offset + 60 <= len(data):
        header = data[offset:offset+60]
        name = header[:16].decode().strip().rstrip('/')
        size = int(header[48:58])
        body = data[offset+60:offset+60+size]
        if name.startswith('data.tar'):
            with tarfile.open(fileobj=io.BytesIO(body), mode='r:*') as archive:
                for member in archive.getmembers():
                    if member.isfile() and (member.name.endswith('.sf2') or member.name.endswith('/copyright')):
                        target = args.output / Path(member.name).name
                        target.write_bytes(archive.extractfile(member).read())
            break
        offset += 60 + size + size % 2
    if not (args.output/'FluidR3_GM.sf2').is_file():
        raise RuntimeError('SoundFont missing from package')
    (args.output/'source.json').write_text(json.dumps({'source':url,'license_file':'copyright'},indent=2))
    print(args.output/'FluidR3_GM.sf2', flush=True)


if __name__ == '__main__':
    main()
