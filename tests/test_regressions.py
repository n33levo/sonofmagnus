import json
import random
import subprocess
import sys
from unittest.mock import patch

import chess
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from chessio.encode import encode_board, swap_colors
from chessio.mask import argmax_legal_move, sample_legal_move, get_legal_move_probs, build_legal_mask
from chessio.policy_map import move_to_policy_index, policy_index_to_move
from eval.arena import play_match, match_summary
from eval.puzzles import solve_puzzle
from model import ChessNet, load_model
from play.runner import ChessRunner
from train.dataset import ChessPositionDataset
from train.distill_labeler import cp_to_q_value, label_position
from train.train import ChessLoss, train_epoch


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return str(path)


def test_mirror_preserves_relative_outcome(tmp_path):
    row = {'fen': chess.STARTING_FEN, 'move': 'e2e4', 'value': .8, 'q_values': {'e2e4': .9}}
    dataset = ChessPositionDataset(write_rows(tmp_path / 'rows.jsonl', [row]))
    with patch('train.dataset.random.random', return_value=0):
        sample = dataset[0]
    mirrored = chess.Board().mirror()
    assert torch.equal(sample['board_tensor'], torch.from_numpy(encode_board(mirrored)))
    assert sample['value_target'].item() == pytest.approx(.8)
    square, plane = move_to_policy_index(chess.Move.from_uci('e7e5'), mirrored)
    assert sample['q_value_target'][square // 8, square % 8, plane].item() == pytest.approx(.9)
    np.testing.assert_array_equal(swap_colors(encode_board(chess.Board())), encode_board(mirrored))


def test_mixed_q_batches_and_missing_value(tmp_path):
    rows = [{'fen': chess.STARTING_FEN, 'move': 'e2e4', 'value': .3, 'q_values': {'e2e4': .9}},
            {'fen': chess.STARTING_FEN, 'move': 'd2d4'}]
    dataset = ChessPositionDataset(write_rows(tmp_path / 'rows.jsonl', rows), augment=False)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch['q_value_target'].shape == (2, 8, 8, 73)
    assert batch['q_value_target'][1].max() == -999
    assert batch['value_mask'].tolist() == [True, False]
    assert batch['legal_mask'].sum(1).sum(1).sum(1).tolist() == [20, 20]


def test_extreme_logits_cannot_select_illegal_moves():
    board = chess.Board()
    logits = np.full((8, 8, 73), -1e30, dtype=np.float32)
    for selector in [argmax_legal_move, sample_legal_move]:
        assert selector(logits, board) in board.legal_moves
    assert sum(get_legal_move_probs(logits, board).values()) == pytest.approx(1)
    terminal = chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')
    assert argmax_legal_move(logits, terminal) is None
    assert get_legal_move_probs(logits, terminal) == {}
    with pytest.raises(ValueError):
        sample_legal_move(logits, board, temperature=-1)


def test_policy_roundtrip_both_colours_and_special_moves():
    fens = ['r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
            '1r5k/P7/8/8/8/8/7p/K5R1 w - - 0 1',
            '1r5k/P7/8/8/8/8/7p/K5R1 b - - 0 1',
            '7k/8/8/3pP3/8/8/8/K7 w - d6 0 1']
    rng = random.Random(5)
    board = chess.Board()
    for _ in range(500):
        if board.is_game_over():
            board.reset()
        fens.append(board.fen())
        board.push(rng.choice(list(board.legal_moves)))
    for fen in fens:
        board = chess.Board(fen)
        keys = set()
        for move in board.legal_moves:
            key = move_to_policy_index(move, board)
            assert key not in keys
            keys.add(key)
            assert policy_index_to_move(*key, board) == move
        assert build_legal_mask(board).sum() == len(keys)
    with pytest.raises(ValueError):
        move_to_policy_index(chess.Move.from_uci('a1b4'), chess.Board())
    with pytest.raises(ValueError):
        policy_index_to_move(chess.A1, 100, chess.Board())


def test_q_loss_fp16_sentinels_and_unlabeled_rows():
    logits = torch.zeros(2, 8, 8, 73, dtype=torch.float16, requires_grad=True)
    policy = torch.zeros_like(logits)
    policy[:, 1, 4, 1] = 1
    q = torch.full_like(logits, -999)
    q[0, 1, 4, 1] = .9
    values = torch.zeros(2, 1, requires_grad=True)
    loss, _, q_loss, _ = ChessLoss(q_value_weight=.4)(logits, values, policy, torch.zeros(2), q)
    assert torch.isfinite(loss)
    assert q_loss.item() == pytest.approx(.16, abs=.001)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


class CountingModel(ChessNet):
    def __init__(self):
        super().__init__(hidden_channels=16, num_blocks=1)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


def test_one_forward_per_move_and_no_board_mutation():
    model = CountingModel()
    runner = ChessRunner(model, device='cpu')
    board = chess.Board()
    before = board.fen()
    assert runner.select_move(board, deterministic=True) in board.legal_moves
    assert model.calls == 1
    assert board.fen() == before
    assert runner.select_move(chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')) is None
    assert model.calls == 1


class ScriptRunner:
    def __init__(self, moves):
        self.moves = iter(moves)
        self.calls = []

    def select_move(self, board, deterministic=True):
        self.calls.append(board.fen())
        return chess.Move.from_uci(next(self.moves))


def test_lichess_setup_and_single_attempt():
    fen = 'q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17'
    solution = 'e8d7 a2e6 d7d8 f7f8'.split()
    runner = ScriptRunner(['a2e6', 'f7f8'])
    assert solve_puzzle(runner, fen, solution)
    board = chess.Board(fen)
    board.push_uci('e8d7')
    assert runner.calls[0] == board.fen()
    assert len(runner.calls) == 2
    assert not solve_puzzle(ScriptRunner(['a2b3']), fen, solution)
    with pytest.raises(ValueError):
        solve_puzzle(runner, fen, solution, max_attempts=3)


def test_training_updates_mps_and_scheduler(tmp_path):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    row = {'fen': chess.STARTING_FEN, 'move': 'e2e4', 'value': .2, 'q_values': {'e2e4': .7}}
    loader = DataLoader(ChessPositionDataset(write_rows(tmp_path / 'rows.jsonl', [row] * 4), augment=False), batch_size=2)
    model = ChessNet(hidden_channels=16, num_blocks=1).to(device)
    before = model.backbone.input_conv.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    metrics = train_epoch(model, loader, optimizer, ChessLoss(q_value_weight=.4), device,
                          use_amp=device == 'mps', amp_dtype='fp16', scheduler=scheduler)
    assert np.isfinite(metrics['loss'])
    assert metrics['samples'] == 4
    assert scheduler.last_epoch == 2
    assert not torch.equal(before, model.backbone.input_conv.weight)


def test_finetune_runs_additional_epochs(tmp_path):
    config = {'model_size': 'mini', 'hidden_channels': 16, 'num_blocks': 1, 'input_channels': 18}
    import yaml
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    row = {'fen': chess.STARTING_FEN, 'move': 'e2e4', 'value': .2}
    data = write_rows(tmp_path / 'rows.jsonl', [row] * 2)
    initial = tmp_path / 'initial.pt'
    torch.save({'config': config, 'epoch': 9, 'model_state_dict': ChessNet(hidden_channels=16, num_blocks=1).state_dict()}, initial)
    subprocess.run([sys.executable, '-m', 'train.train', '--config', str(path), '--data', data,
                    '--epochs', '1', '--batch-size', '2', '--device', 'cpu', '--amp', 'none',
                    '--resume', str(initial), '--ckpt-dir', str(tmp_path / 'ckpts')], check=True, capture_output=True)
    checkpoint = torch.load(tmp_path / 'ckpts/latest.pt', weights_only=True)
    assert checkpoint['epoch'] == 10
    assert checkpoint['metrics']['train']['samples'] == 2
    assert isinstance(load_model(tmp_path / 'ckpts/latest.pt'), ChessNet)


def test_q_conversion_and_complete_stockfish_labels():
    assert cp_to_q_value(100) == pytest.approx(.5621765)
    assert cp_to_q_value(400) == pytest.approx(.7310586)
    assert cp_to_q_value(-1e9) == 0
    assert cp_to_q_value(1e9) == 1
    import shutil
    if not shutil.which('stockfish'):
        pytest.skip('Stockfish not installed')
    import chess.engine
    with chess.engine.SimpleEngine.popen_uci('stockfish') as engine:
        labels = label_position(chess.Board(), engine, depth=2, time_limit=0, include_q_values=True)
    assert set(labels['q_values']) == {m.uci() for m in chess.Board().legal_moves}
    assert sum(labels['move_probs'].values()) == pytest.approx(1)
    assert labels['label_depth_min'] >= 2


def test_arena_pair_results_and_pgn(tmp_path):
    a, b = ScriptRunner(['f2f3', 'g2g4', 'e7e5', 'd8h4']), ScriptRunner(['e7e5', 'd8h4', 'f2f3', 'g2g4'])
    path = tmp_path / 'match.pgn'
    results = play_match(a, b, num_games=2, max_plies=4, pgn_path=path)
    assert results['wins_a'] == results['wins_b'] == 1
    assert results['score_a'] == .5
    assert results['elo_difference'] == 0
    assert path.read_text().count('[Result ') == 2
    assert 'checkmate' in path.read_text()
    with pytest.raises(ValueError):
        play_match(a, b, num_games=1)
    assert match_summary([1., 1.])['elo_difference'] is None


def test_uci_reconstructs_history_and_terminal_position():
    import io
    from play.uci import uci_loop
    runner = ScriptRunner(['g1f3'])
    commands = io.StringIO('uci\nisready\nposition startpos moves e2e4 e7e5\ngo nodes 100\nquit\n')
    output = io.StringIO()
    uci_loop(runner, commands, output)
    assert 'uciok' in output.getvalue() and 'readyok' in output.getvalue()
    assert 'bestmove g1f3' in output.getvalue()
    board = chess.Board()
    board.push_uci('e2e4')
    board.push_uci('e7e5')
    assert runner.calls == [board.fen()]


def test_pretrained_one_forward_and_legal_mapping():
    from pathlib import Path
    weights = Path('ckpts/t1-256x10.pb.gz')
    if not weights.exists():
        pytest.skip('Optional pretrained weights not downloaded')
    pytest.importorskip('lczero.backends')
    from play.pretrained import PretrainedRunner
    runner = PretrainedRunner(weights, 'metal' if torch.backends.mps.is_available() else 'blas')
    fens = [chess.STARTING_FEN, '1r5k/P7/8/8/8/8/7p/K5R1 b - - 0 1',
            '7k/8/8/3pP3/8/8/8/K7 w - d6 0 1']
    for idx, fen in enumerate(fens):
        board = chess.Board(fen)
        before = board.fen()
        assert runner.select_move(board, deterministic=True) in board.legal_moves
        assert runner.forward_calls == idx + 1
        assert board.fen() == before


def test_game_disjoint_pgn_split_and_compression(tmp_path):
    import io
    import zstandard
    from train.dataset import PGNStreamDataset, build_jsonl_from_pgn
    text = ''
    for idx in range(20):
        game = chess.pgn.Game()
        game.headers.update(WhiteElo='2400', BlackElo='2500', Result='1-0')
        board = chess.Board()
        node = game
        rng = random.Random(idx)
        for _ in range(12):
            move = rng.choice(list(board.legal_moves))
            node = node.add_variation(move)
            board.push(move)
        text += str(game) + '\n\n'
    pgn = tmp_path / 'games.pgn.zst'
    pgn.write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
    train, val = tmp_path / 'train.jsonl', tmp_path / 'val.jsonl'
    build_jsonl_from_pgn(str(pgn), str(train), min_elo=2200, sample_rate=1., max_positions=1000,
                         val_path=str(val), val_fraction=.5)
    train_rows, val_rows = [[json.loads(l) for l in p.read_text().splitlines()] for p in (train, val)]
    assert train_rows and val_rows
    assert not {r['game_id'] for r in train_rows} & {r['game_id'] for r in val_rows}
    assert len(train_rows) + len(val_rows) == 240
    assert train_rows[0]['weight'] == 2.


def test_pipeline_a_b_c_smoke(tmp_path):
    import yaml
    config = tmp_path / 'mini.yaml'
    config.write_text(yaml.safe_dump({'model_size': 'mini', 'hidden_channels': 16, 'num_blocks': 1, 'input_channels': 18}))
    data = write_rows(tmp_path / 'data.jsonl', [{'fen': chess.STARTING_FEN, 'move': 'e2e4', 'value': .2}] * 2)
    ckpts = tmp_path / 'ckpts'
    common = ['--config', str(config), '--ckpt-dir', str(ckpts), '--device', 'cpu', '--amp', 'none',
              '--batch-size', '2', '--epochs', '1', '--skip-tests', '--skip-puzzles', '--skip-agreement']
    subprocess.run([sys.executable, '-m', 'pipelines.phase_a_pipeline', '--dataset-path', data,
                    '--skip-download', *common], check=True, capture_output=True)
    latest = ckpts / 'latest.pt'
    assert torch.load(latest, weights_only=True)['epoch'] == 0
    subprocess.run([sys.executable, '-m', 'pipelines.phase_b_pipeline', '--labels-path', data,
                    '--q-values', *common], check=True, capture_output=True)
    assert torch.load(latest, weights_only=True)['epoch'] == 1
    import shutil
    if shutil.which('stockfish'):
        subprocess.run([sys.executable, '-m', 'pipelines.phase_c_pipeline', '--selfplay-path', str(tmp_path / 'selfplay.jsonl'),
                        '--labels-path', str(tmp_path / 'labels.jsonl'), '--games', '1', '--max-moves', '2',
                        '--max-label-positions', '2', '--selfplay-device', 'cpu', *common], check=True, capture_output=True)
        assert torch.load(latest, weights_only=True)['epoch'] == 2


def test_remote_sync_paths_and_quoting(tmp_path):
    from pipelines.utils import PipelineRunner, SSHConfig
    path = tmp_path / 'data with spaces.jsonl'
    path.write_text('test')
    runner = PipelineRunner(python='.venv/bin/python', ssh=SSHConfig('host', 'user', 'key with spaces', '/remote/project'))
    with patch('pipelines.utils.Path.cwd', return_value=tmp_path), patch('pipelines.utils.subprocess.run') as run:
        runner.sync_to_remote(path)
        runner.run(['python', '-c', 'print(1)'], remote=True, env={'NAME': 'hello world'})
    calls = [call.args[0] for call in run.call_args_list]
    assert '/remote/project/' in calls[1][-1]
    assert '-e' in calls[1]
    assert "'key with spaces'" in calls[1][calls[1].index('-e')+1]
    assert "'NAME=hello world'" in calls[2][-1]


def test_export_fp16_and_int8_execute_dynamic_batches(tmp_path):
    import onnxruntime as ort
    from deploy.export_onnx import export_to_onnx
    from deploy.quantize_int8 import quantize_model
    from deploy.size_latency_check import check_model
    config = {'model_size': 'mini', 'hidden_channels': 16, 'num_blocks': 1, 'input_channels': 18}
    checkpoint = tmp_path / 'model.pt'
    torch.save({'config': config, 'model_state_dict': ChessNet(hidden_channels=16, num_blocks=1).state_dict()}, checkpoint)
    fp32, fp16, int8 = [tmp_path / f'model_{name}.onnx' for name in ('fp32', 'fp16', 'int8')]
    export_to_onnx(checkpoint, fp32)
    export_to_onnx(checkpoint, fp16, fp16=True)
    calibration = write_rows(tmp_path / 'calibration.jsonl', [{'fen': chess.STARTING_FEN}])
    metrics = quantize_model(str(fp32), str(int8), calibration_data=calibration)
    assert metrics['size_mb'] < 1
    session = ort.InferenceSession(str(int8), providers=['CPUExecutionProvider'])
    policy, value = session.run(None, {'board': np.stack([encode_board(chess.Board())] * 3)})
    assert policy.shape == (3, 8, 8, 73) and value.shape == (3, 1)
    assert np.isfinite(policy).all()
    from play.runner import load_runner
    runner = load_runner(int8)
    assert runner.select_move(chess.Board(), deterministic=True) in chess.Board().legal_moves
    assert runner.forward_calls == 1
    assert check_model(fp16, num_runs=2, num_warmup=0)['size_ok']
    with pytest.raises(ValueError, match='FP32'):
        quantize_model(str(fp16), str(int8), calibration_data=calibration)


def test_endgame_metrics_use_mover_perspective(tmp_path):
    from eval.endgames import evaluate_endgames
    from chessio.policy_map import build_policy_target
    board = chess.Board('7k/8/6K1/8/8/8/8/Q7 w - - 0 1')
    moves = list(board.legal_moves)
    chosen = moves[0]
    row = {'fen': board.fen(), 'q_values': {m.uci(): float(m == chosen) for m in moves}, 'value': 1.}
    class Runner:
        def policy(self, current):
            return build_policy_target(chosen, current), .8
    results = evaluate_endgames(Runner(), write_rows(tmp_path / 'endgames.jsonl', [row]))
    assert results[3]['optimal_move_rate'] == 1
    assert results[3]['value_mae'] == pytest.approx(.2)
    assert results[3]['forced_choice_total'] == 1


def test_pretrained_download_verifies_before_replacing(tmp_path):
    import hashlib
    import io
    from scripts.fetch_pretrained import fetch
    data = b'checkpoint bytes'
    with patch('scripts.fetch_pretrained.NETWORKS', {'test': ('test.pb.gz', hashlib.sha256(data).hexdigest())}), \
         patch('scripts.fetch_pretrained.urllib.request.urlopen', return_value=io.BytesIO(data)):
        output = fetch('test', tmp_path)
    assert output.read_bytes() == data
    original = output.read_bytes()
    with patch('scripts.fetch_pretrained.NETWORKS', {'test': ('test.pb.gz', 'wrong')}), \
         patch('scripts.fetch_pretrained.urllib.request.urlopen', return_value=io.BytesIO(b'corrupt')):
        with pytest.raises(ValueError, match='checksum'):
            fetch('test', tmp_path)
    assert output.read_bytes() == original
    assert not list(tmp_path.glob('*.part'))


def test_interactive_plays_one_response_then_quits():
    from play.runner import play_interactive
    runner = ScriptRunner(['e7e5'])
    moves = iter(['invalid', 'e2e4', 'quit'])
    board = chess.Board()
    play_interactive(runner, board, input_fn=lambda _: next(moves))
    assert [m.uci() for m in board.move_stack] == ['e2e4', 'e7e5']
    assert len(runner.calls) == 1


def test_selfplay_batches_positions_without_lookahead(tmp_path):
    from train.selfplay import run_selfplay
    model = CountingModel()
    runner = ChessRunner(model, device='cpu')
    output = tmp_path / 'selfplay.jsonl'
    with patch('train.selfplay.load_runner', return_value=runner):
        run_selfplay('unused', output, num_games=4, max_moves=3, batch_size=4)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 12
    assert model.calls == 3
    for row in rows:
        assert chess.Move.from_uci(row['move']) in chess.Board(row['fen']).legal_moves
    assert len({row['game_id'] for row in rows}) == 4
