import copy
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from model.artifacts import artifact_info


def policy_views(graph, parameters):
    mutable, nodes = set(parameters), []
    for node in graph.nodes:
        if mutable.intersection(node[1]):
            mutable.update(node[2])
            nodes.append(node)
    needed, head_nodes = {graph.outputs[0]}, []
    for node in reversed(nodes):
        if needed.intersection(node[2]):
            needed.update(node[1])
            head_nodes.append(node)
    boundary = needed - mutable - graph.weights.keys() - graph.constants.keys()
    if len(boundary) != 1:
        raise ValueError('Policy caching requires a single frozen feature tensor')
    feature = boundary.pop()
    head = copy.copy(graph)
    head.nodes, head.inputs = list(reversed(head_nodes)), [feature]
    head.outputs, head.value_logits = [graph.outputs[0], feature], feature
    frozen = copy.copy(graph)
    frozen.outputs, frozen.value_logits = head.outputs, feature
    needed, frozen.nodes = set(frozen.outputs), []
    for node in reversed(graph.nodes):
        if needed.intersection(node[2]):
            needed.update(node[1])
            frozen.nodes.append(node)
    frozen.nodes.reverse()
    frozen.run = mx.compile(frozen.forward)
    return frozen, head


def cache_features(graph, parameters, dataset, prefix, batch_size, identity, collate):
    frozen, head = policy_views(graph, parameters)
    identity = {**identity, 'trainable_tensors': sorted(parameters), 'feature_input': head.inputs}
    prefix = Path(prefix)
    paths = [prefix.with_suffix('.features.npy'), prefix.with_suffix('.policy.npy')]
    metadata_path = prefix.with_suffix('.json')
    if metadata_path.exists() and all(path.exists() for path in paths):
        metadata = json.loads(metadata_path.read_text())
        if metadata['identity'] == identity and metadata['artifacts'] == [artifact_info(p) for p in paths]:
            return head, [np.load(p, mmap_mode='r') for p in paths]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    temporary = [path.with_suffix('.tmp.npy') for path in paths]
    arrays = None
    for start in range(0, len(dataset), batch_size):
        rows = [dataset.read_row(i) for i in range(start, min(start + batch_size, len(dataset)))]
        inputs, _, _ = collate(rows)
        policy, features = frozen.run(inputs.astype(mx.float16), graph.weights)
        feature_values = np.array(features)
        if feature_values.shape[0] % len(rows):
            raise ValueError('Frozen features do not preserve the board batch')
        values = [feature_values.reshape(len(rows), -1, *feature_values.shape[1:]), np.array(policy)]
        if any(not np.isfinite(value).all() for value in values):
            raise FloatingPointError('Non-finite frozen policy features')
        if arrays is None:
            arrays = [np.lib.format.open_memmap(str(p), mode='w+', dtype=np.float16,
                shape=(len(dataset), *value.shape[1:])) for p, value in zip(temporary, values)]
        for array, value in zip(arrays, values):
            array[start:start + len(rows)] = value
        if start % (batch_size * 50) == 0:
            print(json.dumps({'features_cached': start + len(rows), 'total': len(dataset)}), flush=True)
    for array in arrays:
        array.flush()
    del arrays
    for source, target in zip(temporary, paths):
        source.replace(target)
    metadata = {'identity': identity, 'artifacts': [artifact_info(p) for p in paths]}
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return head, [np.load(p, mmap_mode='r') for p in paths]
