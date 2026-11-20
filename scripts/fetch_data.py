import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def matches(path, entry):
    if path.stat().st_size != entry['bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']


def fetch(name, entry, directory):
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Expected a file name')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / name
    if output.exists():
        if not matches(output, entry):
            raise ValueError(f'{output} differs from the frozen dataset; move it before downloading')
        return output
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix='.' + name, suffix='.part', delete=False) as stream:
            temporary = Path(stream.name)
            remaining = entry['bytes'] + 1
            with urllib.request.urlopen(entry['url'], timeout=60) as response:
                while remaining and (chunk := response.read(min(1 << 20, remaining))):
                    stream.write(chunk)
                    remaining -= len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if not matches(temporary, entry):
            raise ValueError(f'{name}: download size or checksum mismatch')
        # Publish atomically without replacing another downloader's file.
        try:
            os.link(temporary, output)
        except FileExistsError:
            if not matches(output, entry):
                raise ValueError(f'{output} changed during the download')
        return output
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Download pinned chess datasets')
    parser.add_argument('files', nargs='*', help='File names; default: evaluation snapshots')
    parser.add_argument('--directory', default='data')
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'data/downloads.json').read_text())
    if set(args.files) - manifest.keys():
        parser.error('Unknown dataset name')
    for name in args.files or [name for name, entry in manifest.items() if entry.get('default', True)]:
        print(fetch(name, manifest[name], args.directory))


if __name__ == '__main__':
    main()
