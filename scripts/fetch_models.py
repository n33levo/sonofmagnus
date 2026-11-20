"""Download a checksum-pinned release model without overwriting local weights."""
import argparse
import json
from pathlib import Path
import tarfile
import tempfile

from model.artifacts import artifact_info
from scripts.fetch_data import fetch


def fetch_model(entry,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if 'directory' not in entry:return fetch(entry['file'],entry,directory)
    destination=directory/entry['directory']
    if destination.exists():
        if artifact_info(destination)['sha256']!=entry['artifact_sha256']:raise ValueError('Existing model differs from the release')
        return destination
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        archive=fetch(entry['file'],entry,temporary)
        with tarfile.open(archive) as tar:tar.extractall(temporary,filter='data')
        model=Path(temporary)/entry['directory']
        if artifact_info(model)['sha256']!=entry['artifact_sha256']:raise ValueError('Extracted model differs from the release')
        model.rename(destination)
    return destination


if __name__=='__main__':
    manifest=json.loads((Path(__file__).resolve().parents[1]/'results/models.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=manifest);parser.add_argument('--directory',default='ckpts')
    args=parser.parse_args();print(fetch_model(manifest[args.model],args.directory))
