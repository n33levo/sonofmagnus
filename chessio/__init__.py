"""IO layer for chess board encoding and policy mapping."""

from importlib import import_module

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


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    index = __all__.index(name)
    module = 'encode' if index < 5 else 'policy_map' if index < 9 else 'mask'
    value = getattr(import_module('chessio.' + module), name)
    globals()[name] = value
    return value
