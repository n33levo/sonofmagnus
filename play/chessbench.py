import json
import math
import time
from pathlib import Path

import chess
import mlx.core as mx
import numpy as np

from chessio.policy_map import move_to_policy_index
from play.runner import ChessRunner


CHARACTERS = '0123456789abcdefghpnrkqPBNRQKw.'


def tokenize(fen):
    cells, side, castle, ep, clock, full = fen.split()
    cells = ''.join('.' * int(c) if c.isdigit() else c for c in cells if c != '/')
    text = side + cells + ('' if castle == '-' else castle).ljust(4, '.') + ('..' if ep == '-' else ep) + clock.ljust(3, '.') + full.ljust(3, '.')
    if len(text) != 77:
        raise ValueError('ChessBench supports three-digit FEN counters')
    return np.array([CHARACTERS.index(c) for c in text], np.int32)


def action_vocabulary():
    board, moves = chess.BaseBoard.empty(), []
    for square in range(64):
        board.set_piece_at(square, chess.Piece.from_symbol('Q'))
        targets = list(board.attacks(square))
        board.set_piece_at(square, chess.Piece.from_symbol('N'))
        targets += list(board.attacks(square))
        board.remove_piece_at(square)
        moves.extend(chess.square_name(square) + chess.square_name(target) for target in targets)
    for rank, end in [('2', '1'), ('7', '8')]:
        for file in range(8):
            targets = [file] + ([file - 1] if file else []) + ([file + 1] if file < 7 else [])
            for target in targets:
                moves.extend('abcdefgh'[file] + rank + 'abcdefgh'[target] + end + piece for piece in 'qrbn')
    return {move: i for i, move in enumerate(moves)}


class ChessBenchRunner(ChessRunner):
    def __init__(self, directory):
        mx.set_cache_limit(256 << 20)
        directory = Path(directory)
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        if self.metadata.get('architecture') != 'chessbench-action-value':
            raise ValueError('Expected a ChessBench action-value checkpoint')
        self.weights = mx.load(str(directory / 'weights.safetensors'))
        self.quantized = self.metadata['quantized']
        self.dimension, self.layers, self.heads = [self.metadata[key] for key in ('dimension', 'layers', 'heads')]
        if min(self.dimension, self.layers, self.heads) < 1 or self.dimension % self.heads:
            raise ValueError('Invalid transformer dimensions')
        if self.weights['embed/embeddings'].shape != (1968, self.dimension) or self.weights['embed_1/embeddings'].shape != (79, self.dimension):
            raise ValueError('Invalid token or position embeddings')
        self.actions = action_vocabulary()
        self.forward_calls = self.root_positions = self.action_queries = 0
        self.temp_opening = self.temp_midgame = self.temp_endgame = .005
        self.last_latency_ms = 0.
        self.run = mx.compile(self.forward)

    def move_probs(self, board):
        return dict(sorted(super().move_probs(board).items(), key=lambda pair: pair[0].uci()))

    def select_move(self, board, temperature=None, deterministic=False):
        if not deterministic and temperature != 0:
            return super().select_move(board, temperature, deterministic)
        if board.is_game_over():
            return None
        start = time.perf_counter()
        logits, _ = self.policy(board)
        def score(move):
            square, plane = move_to_policy_index(move, board)
            return logits[square // 8, square % 8, plane]
        move = max(sorted(board.legal_moves, key=lambda m: m.uci()), key=score)
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        return move

    def forward(self, tokens, weights):
        def scope(prefix, index):
            return prefix + ('_' + str(index) if index else '')
        def linear(x, name, wide=False):
            x = x.astype(mx.float32 if wide else weights['embed/embeddings'].dtype)
            name += '/w'
            if name not in self.quantized:
                return x @ weights[name].astype(x.dtype)
            group, bits, transpose = self.quantized[name]
            return mx.quantized_matmul(x, weights[name + '.packed'], weights[name + '.scales'].astype(x.dtype),
                weights[name + '.biases'].astype(x.dtype), group_size=group, bits=bits, transpose=transpose)
        def norm(x, index):
            name = scope('layer_norm', index)
            return mx.fast.layer_norm(x, weights[name + '/scale'], weights[name + '/offset'], eps=1e-5)
        # Later residuals exceed FP16's range in the released 270M checkpoint.
        h = weights['embed/embeddings'][tokens].astype(mx.float32) * math.sqrt(self.dimension) + weights['embed_1/embeddings'].astype(mx.float32)
        width = self.dimension // self.heads
        for i in range(self.layers):
            a, name = norm(h, 2*i), scope('multi_head_dot_product_attention', i)
            q, k, v = [linear(a, name + '/' + scope('linear', j)).reshape(-1, 79, self.heads, width).transpose(0, 2, 1, 3) for j in range(3)]
            attention = mx.fast.scaled_dot_product_attention(q, k, v, scale=width ** -.5)
            h = h + linear(attention.transpose(0, 2, 1, 3).reshape(-1, 79, self.dimension), name + '/linear_3', wide=True)
            a, gate_name = norm(h, 2*i + 1), 3*i
            gate = linear(a, scope('linear', gate_name))
            h = h + linear(gate * mx.sigmoid(gate) * linear(a, scope('linear', gate_name + 1)), scope('linear', gate_name + 2), wide=True)
        name = scope('linear', 3*self.layers)
        return (linear(norm(h[:, -1], 2*self.layers), name) + weights[name + '/b']).astype(mx.float32)

    def batch_policy(self, boards):
        if any(board.chess960 for board in boards):
            raise ValueError('ChessBench supports standard chess')
        groups = [sorted(board.legal_moves, key=lambda move: move.uci()) for board in boards]
        if not groups or any(not moves for moves in groups):
            raise ValueError('Expected nonterminal root positions')
        sequences = []
        for board, moves in zip(boards, groups):
            # The upstream decoder shifts [FEN, action, return] right by one token.
            row = np.concatenate([[0], tokenize(board.fen()), [0]]).astype(np.int32)
            queries = np.tile(row, (len(moves), 1))
            queries[:, -1] = [self.actions[move.uci()] for move in moves]
            sequences.append(queries)
        output = self.run(mx.array(np.concatenate(sequences)), self.weights)
        values = mx.softmax(output, axis=-1) @ ((mx.arange(128, dtype=mx.float32) + .5) / 128)
        mx.eval(values)
        values = np.array(values)
        if values.shape != (sum(map(len, groups)),) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError('Invalid action-value predictions')
        self.forward_calls += 1
        self.root_positions += len(boards)
        self.action_queries += len(values)
        result, offset = [], 0
        for board, moves in zip(boards, groups):
            logits = np.full((8, 8, 73), -np.inf, np.float32)
            scores = values[offset:offset + len(moves)]
            offset += len(moves)
            for move, score in zip(moves, scores):
                square, plane = move_to_policy_index(move, board)
                logits[square // 8, square % 8, plane] = score
            result.append((logits, float(2*scores.max() - 1)))
        return result
