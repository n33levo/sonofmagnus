import hashlib
from pathlib import Path


def artifact_info(path):
    path = Path(path)
    if path.is_file():
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'model_bytes': path.stat().st_size, 'sha256': digest}
    digest, size = hashlib.sha256(), 0
    files = sorted(p for p in path.rglob('*') if p.is_file())
    if not files:
        raise ValueError(f'Empty model artifact: {path}')
    for file in files:
        digest.update(str(file.relative_to(path)).encode() + b'\0')
        with file.open('rb') as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
                size += len(chunk)
    return {'model_bytes': size, 'sha256': digest.hexdigest()}
