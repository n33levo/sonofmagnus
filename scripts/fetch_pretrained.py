import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

NETWORKS = {
    't74': ('744706_spsa072024.pb.gz', '2fbc783b016005c8bb20178a9d0096c8c0749bce3df1b26e59d0231ba293a7ba'),
    'bt3': ('BT3-768x15x24h-swa-2790000.pb.gz', 'e3067757d1fc2dfc66947b21d15ace0cedf4c54254fc1de83d77c378a3e8b8e1'),
    't1': ('t1-256x10-distilled-swa-2432500.pb.gz', 'bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406'),
    'bt4': ('BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz', 'e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618'),
}


def fetch(name, directory='ckpts'):
    filename, digest = NETWORKS[name]
    url = 'https://storage.lczero.org/files/networks-contrib/' + filename
    output = Path(directory) / (name + '.pb.gz')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.part')
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
            while chunk := response.read(1 << 20):
                stream.write(chunk)
        with temporary.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != digest:
            raise ValueError('Pretrained weights checksum mismatch')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    output.with_suffix('.json').write_text(json.dumps({'source': url, 'sha256': digest,
        'architecture': 'LC0 pretrained transformer; not the compact ResNet',
        'runtime_source': 'https://github.com/LeelaChessZero/lc0', 'runtime_license': 'GPL-3.0'}, indent=2) + '\n')
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', choices=NETWORKS, default='bt3')
    parser.add_argument('--directory', default='ckpts')
    args = parser.parse_args()
    fetch(args.network, args.directory)
