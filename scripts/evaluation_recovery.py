"""Repair missing exclusion metadata only for explicit evaluation queries."""
import json
from pathlib import Path
import shutil

from model.artifacts import artifact_info


def copy_evaluation_requests(requests, output):
    requests, output = Path(requests).resolve(), Path(output).resolve()
    original_meta = requests.with_suffix('.json')
    metadata = json.loads(original_meta.read_text())
    if output.exists() or output.with_suffix('.json').exists() or 'exclusions' in metadata:
        raise ValueError('Use a new output; only a missing exclusions field may be repaired')
    if artifact_info(requests) != metadata['artifact']:
        raise ValueError('Original evaluation queries changed')
    rows = list(map(json.loads, requests.open()))
    if not rows or any(row.get('validation_only') is not True or row.get('diagnostic_only') is not True
                       or row.get('policy_supervision') is not False for row in rows):
        raise ValueError('Every query must explicitly prohibit fitting')
    frozen = {**metadata['inputs'], **{str(path): artifact_info(path)
        for path in (requests, original_meta, Path(__file__).resolve())}}
    if any(artifact_info(path) != value for path, value in frozen.items()):
        raise ValueError('Evaluation evidence changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as target, requests.open('rb') as source:
        shutil.copyfileobj(source, target)
    repaired = dict(metadata, artifact=artifact_info(output), inputs=frozen, exclusions=[],
        approved_for_fit=False, metadata_recovery=dict(kind='missing-evaluation-exclusions',
            original_metadata=artifact_info(original_meta), query_bytes_unchanged=True,
            scope='Evaluation-only teacher queries. Existing training exclusions and all original evidence remain unchanged.'))
    if repaired['artifact'] != metadata['artifact'] or any(artifact_info(path) != value for path, value in frozen.items()):
        raise ValueError('Original evidence or copied queries changed')
    output.with_suffix('.json').write_text(json.dumps(repaired, indent=2) + '\n')
    return dict(roots=len(rows), artifact=repaired['artifact'], inputs=frozen,
                exclusions=[], approved_for_fit=False, query_bytes_unchanged=True)
