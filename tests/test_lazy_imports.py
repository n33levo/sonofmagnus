import json
import importlib.util
import subprocess
import sys

import chess
import numpy as np
import pytest


@pytest.mark.parametrize('imports', [
    'from model.artifacts import artifact_info; from chessio.records import record_board',
    pytest.param('import play.mlx; import play.chessbench',marks=pytest.mark.skipif(
        importlib.util.find_spec('mlx') is None,reason='MLX is an optional Apple Silicon backend')),
    'from chessio import encode_board, build_legal_mask, apply_legal_mask, move_to_policy_index',
])
def test_non_torch_paths_do_not_import_torch(imports):
    code = imports + '; import sys; assert "torch" not in sys.modules'
    result = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_lazy_model_api_preserves_checkpoint_and_inference(tmp_path):
    import torch
    from model import ChessNet, create_model, load_model
    from play.runner import ChessRunner, load_runner

    torch.manual_seed(37)
    config = dict(hidden_channels=8, num_blocks=1, num_groups=4)
    original = create_model(config).eval()
    assert isinstance(original, ChessNet)
    path = tmp_path / 'model.pt'
    torch.save(dict(config=config, model_state_dict=original.state_dict()), path)
    restored = load_model(path)
    assert original.state_dict().keys() == restored.state_dict().keys()
    for name in original.state_dict():
        assert torch.equal(original.state_dict()[name], restored.state_dict()[name])
    board = chess.Board()
    expected = ChessRunner(original, device='cpu').policy(board)
    loaded = load_runner(path, device='cpu')
    actual = loaded.policy(board)
    np.testing.assert_array_equal(expected[0], actual[0])
    assert expected[1] == actual[1] and loaded.forward_calls == 1
    assert all(parameter.grad is None for parameter in loaded.model.parameters())


def test_package_exports_and_unknown_attributes():
    import chessio
    import model
    for name in chessio.__all__:
        assert callable(getattr(chessio, name))
    for name in (*model.__all__, 'create_model', 'load_model', 'resolve_device'):
        assert callable(getattr(model, name))
    for module in (chessio, model):
        with pytest.raises(AttributeError):
            getattr(module, 'not_a_public_api')


def test_numpy_and_torch_masks_still_agree():
    import torch
    from chessio import build_legal_mask, build_legal_mask_torch, apply_legal_mask, apply_legal_mask_torch
    board = chess.Board()
    mask = build_legal_mask(board)
    tensor = build_legal_mask_torch(board)
    np.testing.assert_array_equal(mask, tensor.numpy())
    logits = np.arange(mask.size, dtype=np.float32).reshape(mask.shape)
    np.testing.assert_array_equal(apply_legal_mask(logits, mask),
                                  apply_legal_mask_torch(torch.from_numpy(logits), tensor).numpy())
