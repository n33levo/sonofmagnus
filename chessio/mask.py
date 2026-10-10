import numpy as np
import torch

from chessio.policy_map import move_to_policy_index


def legal_indices(board):
    moves = list(board.legal_moves)
    indices = [square * 73 + plane for square, plane in
               (move_to_policy_index(move, board) for move in moves)]
    return moves, np.asarray(indices, dtype=np.int64)


def build_legal_mask(board):
    _, indices = legal_indices(board)
    mask = np.zeros(64 * 73, dtype=np.float32)
    mask[indices] = 1
    return mask.reshape(8, 8, 73)


def build_legal_mask_torch(board, device="cpu"):
    return torch.from_numpy(build_legal_mask(board)).to(device)


def apply_legal_mask(logits, mask, large_negative=-np.inf):
    return np.where(mask != 0, logits, large_negative)


def apply_legal_mask_torch(logits, mask, large_negative=-torch.inf):
    return logits.masked_fill(mask == 0, large_negative)


def legal_scores(policy_logits, board):
    moves, indices = legal_indices(board)
    if not moves:
        return moves, np.empty(0)
    logits = np.asarray(policy_logits)
    if logits.shape != (8, 8, 73):
        raise ValueError(f"Expected (8, 8, 73) logits, got {logits.shape}")
    scores = logits.reshape(-1)[indices].astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite legal policy logits")
    return moves, scores


def sample_legal_move(policy_logits, board, temperature=1.0):
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative")
    moves, scores = legal_scores(policy_logits, board)
    if not moves:
        return None
    if temperature == 0:
        return moves[int(scores.argmax())]
    probs = np.exp((scores - scores.max()) / temperature)
    probs /= probs.sum()
    return moves[int(np.random.choice(len(moves), p=probs))]


def argmax_legal_move(policy_logits, board):
    return sample_legal_move(policy_logits, board, temperature=0)


def get_legal_move_probs(policy_logits, board):
    moves, scores = legal_scores(policy_logits, board)
    if not moves:
        return {}
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    return dict(zip(moves, probs.tolist()))
