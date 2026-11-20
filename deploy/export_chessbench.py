import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from deploy.export_mlx import quantize_weights
from model.artifacts import artifact_info


CONFIGS = {'9M': (8, 256), '136M': (8, 1024), '270M': (16, 1024)}


def export(parameters, archive, directory, model='270M', bits=8):
    layers, dimension = CONFIGS[model]
    source, archive_identity = artifact_info(parameters), artifact_info(archive)
    raw = np.load(parameters, allow_pickle=False)
    if raw['embed/embeddings'].shape != (1968, dimension) or raw['embed_1/embeddings'].shape != (79, dimension) or raw[f'linear_{3*layers}/w'].shape != (dimension, 128):
        raise ValueError('Checkpoint differs from the official action-value architecture')
    weights, count = {}, 0
    for name in raw.files:
        value = raw[name].astype(np.float16)
        if not np.isfinite(value).all():
            raise ValueError('Non-finite parameter: ' + name)
        count += value.size
        weights[name] = mx.array(value)
    raw.close()
    quantized = quantize_weights(weights, {name for name in weights if name.endswith('/w')}, bits, 64)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    mx.save_safetensors(str(directory / 'weights.safetensors'), weights)
    metadata = {'architecture': 'chessbench-action-value', 'parameters': count, 'dimension': dimension,
        'layers': layers, 'heads': 8, 'bits': bits, 'quantized': quantized, 'epsilon': 1e-5,
        'causal_mask': False, 'checkpoint_step': 6400000, 'ema': False, 'source_parameters': source,
        'source': f'https://storage.googleapis.com/searchless_chess/checkpoints/{model}.zip',
        'source_archive': archive_identity, 'license': 'CC-BY-4.0',
        'source_code': 'https://github.com/google-deepmind/searchless_chess',
        'configuration_source': 'https://github.com/google-deepmind/searchless_chess/blob/main/src/engines/constants.py',
        'inference': 'One batched forward pass of current FEN and legal action tokens; no successor positions or repetition lookahead',
        'strength': 'Not established by downloading or conversion; requires its own recorded tests'}
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if source != artifact_info(parameters) or archive_identity != artifact_info(archive):
        raise ValueError('Source checkpoint changed during conversion')
    return artifact_info(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model', choices=CONFIGS, default='270M')
    parser.add_argument('--bits', type=int, choices=[2, 3, 4, 6, 8, 16], default=8)
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    print(json.dumps(export(args.parameters, args.archive, args.out, args.model, args.bits), indent=2))
