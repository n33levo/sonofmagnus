"""IO layer for chess board encoding and policy mapping."""

from chessio.encode import encode_board, encode_board_batch, swap_colors, flip_board_horizontal, flip_board_vertical
from chessio.policy_map import move_to_policy_index, policy_index_to_move, build_policy_target, build_policy_distribution
from chessio.mask import build_legal_mask, build_legal_mask_torch, apply_legal_mask, apply_legal_mask_torch, sample_legal_move, argmax_legal_move, get_legal_move_probs

__all__ = [
    'encode_board',
    'encode_board_batch',
    'swap_colors',
    'flip_board_horizontal',
    'flip_board_vertical',
    'move_to_policy_index',
    'policy_index_to_move',
    'build_policy_target',
    'build_policy_distribution',
    'build_legal_mask',
    'build_legal_mask_torch',
    'apply_legal_mask',
    'apply_legal_mask_torch',
    'sample_legal_move',
    'argmax_legal_move',
    'get_legal_move_probs',
]
