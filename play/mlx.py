import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from play.runner import ChessRunner
from chessio.lc0 import encode_history, policy_index
from chessio.policy_map import move_to_policy_index


class Graph:
    def __init__(self, directory):
        directory = Path(directory)
        specification = json.loads((directory / 'graph.json').read_text())
        loaded = mx.load(str(directory / 'weights.safetensors'))
        self.constants = {name: np.array(loaded[name]) for name in specification['integer_constants']}
        self.weights = {name: value for name, value in loaded.items() if name not in self.constants}
        self.nodes, self.inputs, self.outputs = [specification[key] for key in ('nodes', 'inputs', 'outputs')]
        self.value_logits = next((node[1][0] for node in self.nodes if node[0] == 'Softmax' and self.outputs[1] in node[2]), None)
        if self.value_logits is None:
            raise ValueError('Expected a WDL softmax output')
        self.quantized = specification.get('quantized', {})
        self.nodes = fuse_attention(self.nodes, self.outputs, self.weights)
        self.run = mx.compile(self.forward)

    def forward(self, x, weights, raw_value=False, collect=False):
        data = {**self.constants, **weights, self.inputs[0]: x}
        moments = {}
        for op, names, outputs, attrs in self.nodes:
            args = [data.get(n) for n in names]
            a = args[0]
            if op == 'MatMul':
                name = names[1]
                if collect and name in weights and weights[name].ndim == 2:
                    features = a.reshape(-1, a.shape[-1]).astype(mx.float32)
                    moment = features.T @ features
                    moments[name] = moments.get(name, 0) + moment
                if name in self.quantized:
                    configuration = self.quantized[name]
                    group, bits = configuration[:2]
                    transpose = configuration[2] if len(configuration) > 2 else True
                    y = mx.quantized_matmul(a, weights[name + '.packed'], weights[name + '.scales'],
                                           weights[name + '.biases'], transpose=transpose, group_size=group, bits=bits)
                else:
                    y = mx.matmul(a, args[1])
            elif op == 'Add': y = a + args[1]
            elif op == 'Mul': y = a * args[1]
            elif op == 'Conv':
                pads = attrs.get('pads', [0, 0, 0, 0])
                if pads[:2] != pads[2:]:
                    raise ValueError('MLX convolution requires symmetric padding')
                y = mx.conv2d(a.transpose(0, 2, 3, 1), args[1].transpose(0, 2, 3, 1),
                              stride=tuple(attrs.get('strides', [1, 1])), padding=tuple(pads[:2]),
                              dilation=tuple(attrs.get('dilations', [1, 1])), groups=attrs.get('group', 1))
                if len(args) > 2:
                    y = y + args[2]
                y = y.transpose(0, 3, 1, 2)
            elif op == 'ReduceMean':
                axes = args[1] if len(args) > 1 else attrs.get('axes')
                y = mx.mean(a, axis=tuple(int(axis) for axis in axes) if axes is not None else None,
                            keepdims=bool(attrs.get('keepdims', 1)))
            elif op == 'Reshape': y = a.reshape(tuple(int(n) if n else a.shape[i] for i, n in enumerate(args[1])))
            elif op == 'Transpose': y = a.transpose(attrs.get('perm'))
            elif op == 'Shape': y = np.asarray(a.shape, dtype=np.int32)
            elif op == 'Slice':
                slices = [slice(None)] * a.ndim
                axes = args[3] if len(args) >= 4 else range(len(args[1]))
                steps = args[4] if len(args) >= 5 else [1] * len(args[1])
                for axis, start, end, step in zip(axes, args[1], args[2], steps):
                    slices[int(axis)] = slice(int(start), int(end), int(step))
                y = a[tuple(slices)]
            elif op == 'Concat':
                y = np.concatenate(args, axis=attrs['axis']) if all(isinstance(v, np.ndarray) for v in args) else mx.concatenate(args, axis=attrs['axis'])
            elif op == 'Expand': y = mx.broadcast_to(a, tuple(int(n) for n in args[1]))
            elif op == 'Sigmoid': y = mx.sigmoid(a)
            elif op == 'Softplus': y = mx.logaddexp(a, mx.array(0, dtype=a.dtype))
            elif op == 'Tanh': y = mx.tanh(a)
            elif op == 'Relu': y = mx.maximum(a, 0)
            elif op == 'LayerNormalization':
                if attrs.get('axis', -1) % a.ndim != a.ndim - 1:
                    raise ValueError('MLX layer normalization requires the last axis')
                y = mx.fast.layer_norm(a, args[1], args[2], eps=attrs.get('epsilon', 1e-5))
            elif op == 'Attention':
                y = mx.fast.scaled_dot_product_attention(a, args[1].swapaxes(-1, -2), args[2],
                                                        scale=attrs['scale'], mask=args[3])
            elif op == 'Softmax': y = mx.softmax(a, axis=attrs.get('axis', -1), precise=True)
            elif op == 'Gather': y = mx.take(a, mx.array(args[1], dtype=mx.int32), axis=attrs.get('axis', 0))
            elif op == 'Split':
                sizes = args[1] if len(args) > 1 else attrs.get('split')
                y = mx.split(a, np.cumsum(sizes)[:-1].tolist() if sizes is not None else len(outputs), axis=attrs.get('axis', 0))
            else: raise ValueError(f'Unsupported MLX operation: {op}')
            data.update(zip(outputs, y if op == 'Split' else [y]))
        result = (data[self.outputs[0]], data[self.value_logits if raw_value else self.outputs[1]])
        return (result, moments) if collect else result


def fuse_attention(nodes, outputs, weights):
    producers = {out: node for node in nodes for out in node[2]}
    replaced = []
    for op, inputs, out, attrs in nodes:
        softmax = producers.get(inputs[0])
        if op == 'MatMul' and softmax and softmax[0] == 'Softmax' and softmax[3].get('axis') == 3:
            add = producers.get(softmax[1][0])
            scale = producers.get(add[1][0]) if add and add[0] == 'Add' else None
            product = producers.get(scale[1][0]) if scale and scale[0] == 'Mul' else None
            if product and product[0] == 'MatMul' and '/mha/QK/' in product[2][0] and scale[1][1] in weights:
                factor = float(np.array(weights[scale[1][1]]).item())
                replaced.append(['Attention', [*product[1], inputs[1], add[1][1]], out, {'scale': factor}])
                continue
        replaced.append([op, inputs, out, attrs])
    needed, result = set(outputs), []
    for node in reversed(replaced):
        if needed.intersection(node[2]):
            needed.update(node[1])
            result.append(node)
    return list(reversed(result))


class MlxRunner(ChessRunner):
    def __init__(self, directory):
        self.graph = Graph(directory)
        self.temp_opening, self.temp_midgame, self.temp_endgame = .6, .4, .2
        self.last_latency_ms, self.forward_calls = 0., 0

    def batch_policy(self, boards):
        tensor = mx.array(np.stack([encode_history(b) for b in boards]).astype(np.float16))
        policy, wdl = self.graph.run(tensor, self.graph.weights)
        mx.eval(policy, wdl)
        self.forward_calls += 1
        policy, wdl = np.array(policy), np.array(wdl)
        if policy.shape != (len(boards), 1858) or wdl.shape != (len(boards), 3):
            raise ValueError('Expected 1858 policy logits and three WDL probabilities per board')
        if not np.isfinite(policy).all() or not np.isfinite(wdl).all():
            raise ValueError('Non-finite MLX output')
        if (wdl < 0).any() or (wdl > 1).any() or not np.allclose(wdl.sum(-1), 1, atol=.002):
            raise ValueError('Invalid WDL probabilities')
        result = []
        for board, scores, value in zip(boards, policy, wdl):
            logits = np.full((8, 8, 73), -np.inf, dtype=np.float32)
            for move in board.legal_moves:
                square, plane = move_to_policy_index(move, board)
                logits[square // 8, square % 8, plane] = scores[policy_index(move, board)]
            result.append((logits, float(value[0] - value[2])))
        return result
