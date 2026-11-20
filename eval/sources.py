from contextlib import contextmanager
from pathlib import Path

from model.artifacts import artifact_info


ROOT = Path(__file__).resolve().parents[1]


def runtime_hashes(evaluation):
    files = [ROOT / name for name in ('eval/sources.py', *evaluation)]
    files += [path for package in ('play', 'chessio', 'model') for path in (ROOT / package).glob('*.py')]
    return {str(path.relative_to(ROOT)): artifact_info(path)['sha256'] for path in sorted(set(files))}


@contextmanager
def frozen_runtime(*evaluation):
    hashes = runtime_hashes(evaluation)
    yield hashes
    if runtime_hashes(evaluation) != hashes:
        raise ValueError('Runtime source changed during evaluation')
