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


def test_main_preset_fits_the_six_million_parameter_budget():
    import yaml
    from model import create_model, create_model_main
    from pathlib import Path
    config = yaml.safe_load(Path('configs/main.yaml').read_text())
    counts = [create_model(config).get_num_params(), create_model('main').get_num_params(), create_model_main().get_num_params()]
    assert counts == [5827283] * 3


def run_cli(args):
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


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
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
        board.push_uci(move)
    assert board.can_claim_threefold_repetition() and not board.is_game_over()
    assert runner.select_move(board, deterministic=True) in board.legal_moves
    assert model.calls == 2
    assert runner.select_move(chess.Board('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')) is None
    assert model.calls == 2


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


def test_puzzle_bands_keep_counts_and_sampling_uncertainty(tmp_path):
    from eval.puzzles import evaluate_runner
    rows = [{'fen': chess.STARTING_FEN, 'moves': ['e2e4', 'e7e5'], 'rating': r}
            for r in (1500, 1799, 1800, 2100, 2400, 3100)]
    with patch('eval.puzzles.solve_puzzle', side_effect=[True, False, True, True, True, True]):
        result = evaluate_runner(None, write_rows(tmp_path / 'puzzles.jsonl', rows))
    band = result['bands']['1500-1800']
    assert band['correct'] == 1 and band['total'] == 2 and band['accuracy'] == .5
    assert 0 < band['clopper_pearson_simultaneous_95'][0] < .5 < band['clopper_pearson_simultaneous_95'][1] < 1
    assert result['bands']['2700+']['total'] == 1
    assert result['overall']['correct'] == 5 and result['overall']['total'] == 6


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
    run_cli([sys.executable, '-m', 'train.train', '--config', str(path), '--data', data,
                    '--epochs', '1', '--batch-size', '2', '--device', 'cpu', '--amp', 'none',
                    '--resume', str(initial), '--ckpt-dir', str(tmp_path / 'ckpts')])
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
    run_cli([sys.executable, '-m', 'pipelines.phase_a_pipeline', '--dataset-path', data,
                    '--skip-download', *common])
    latest = ckpts / 'latest.pt'
    assert torch.load(latest, weights_only=True)['epoch'] == 0
    run_cli([sys.executable, '-m', 'pipelines.phase_b_pipeline', '--labels-path', data,
                    '--q-values', *common])
    assert torch.load(latest, weights_only=True)['epoch'] == 1
    import shutil
    if shutil.which('stockfish'):
        run_cli([sys.executable, '-m', 'pipelines.phase_c_pipeline', '--selfplay-path', str(tmp_path / 'selfplay.jsonl'),
                        '--labels-path', str(tmp_path / 'labels.jsonl'), '--games', '1', '--max-moves', '2',
                        '--max-label-positions', '2', '--selfplay-device', 'cpu', *common])
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
        from chessio.records import record_board
        board = record_board(row)
        assert chess.Move.from_uci(row['move']) in board.legal_moves
        assert len(board.move_stack) == len(row['history'])
    assert len({row['game_id'] for row in rows}) == 4


def test_training_records_keep_repetition_history():
    from chessio.records import record_board
    board = chess.Board()
    moves = ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2
    for move in moves:
        board.push_uci(move)
    row = {'root_fen': chess.STARTING_FEN, 'history': moves, 'fen': board.fen()}
    restored = record_board(row)
    assert restored.can_claim_threefold_repetition()
    assert restored.move_stack == board.move_stack
    row['fen'] = chess.STARTING_FEN
    with pytest.raises(ValueError, match='disagree'):
        record_board(row)


def test_failed_pgn_stream_preserves_existing_datasets(tmp_path):
    from train.dataset import PGNStreamDataset, build_jsonl_from_pgn
    train, val = tmp_path / 'train.jsonl', tmp_path / 'val.jsonl'
    train.write_text('old training data')
    val.write_text('old validation data')
    def broken():
        yield {'fen': chess.STARTING_FEN, 'move': 'e2e4', 'game_id': '0'}
        raise OSError('stream interrupted')
    with patch.object(PGNStreamDataset, '_parse_pgn', return_value=broken()):
        with pytest.raises(OSError, match='interrupted'):
            build_jsonl_from_pgn('https://example.org/games.pgn.zst', train, val_path=val)
    assert train.read_text() == 'old training data' and val.read_text() == 'old validation data'
    assert set(tmp_path.iterdir()) == {train, val}
    with pytest.raises(ValueError, match='must differ'):
        build_jsonl_from_pgn(train, train)


def test_clock_forfeit_is_draw_when_opponent_cannot_mate(tmp_path):
    fen = '7k/8/8/8/8/8/8/1R5K w - - 0 1'
    a, b = ScriptRunner(['b1b2']), ScriptRunner(['b1b2'])
    path = tmp_path / 'flags.pgn'
    with patch('eval.arena.time.perf_counter', side_effect=[0., 1., 2., 3.]):
        result = play_match(a, b, num_games=2, openings=[fen], clock_seconds=.1, pgn_path=path)
    assert result['draws'] == 2
    assert path.read_text().count('time_forfeit') == 2


def test_local_pipeline_uses_the_active_python(tmp_path):
    from pipelines.run import main
    data = write_rows(tmp_path / 'rows.jsonl', [{'fen': chess.STARTING_FEN, 'move': 'e2e4'}])
    arguments = ['phase-a', '--dataset-path', data, '--skip-download', '--skip-tests', '--skip-puzzles', '--skip-agreement']
    with patch('sys.argv', arguments), patch('pipelines.run.PipelineRunner.run') as run:
        main('A')
    assert run.call_args_list[0].args[0][0] == sys.executable


def test_remote_training_keeps_local_validation_on_active_python(tmp_path):
    from pipelines.run import main
    data = write_rows(tmp_path / 'rows.jsonl', [{'fen': chess.STARTING_FEN, 'move': 'e2e4'}])
    arguments = ['phase-a', '--dataset-path', data, '--skip-download', '--skip-puzzles', '--skip-agreement',
                 '--train-remote', '--ssh-host', 'host', '--ssh-user', 'user', '--remote-workdir', '/remote/chess']
    with patch('sys.argv', arguments), patch('pipelines.run.PipelineRunner.run') as run, \
         patch('pipelines.run.PipelineRunner.sync_repository'), patch('pipelines.run.PipelineRunner.sync_to_remote'), \
         patch('pipelines.run.PipelineRunner.fetch_from_remote'):
        main('A')
    assert run.call_args_list[0].args[0][0] == '.venv/bin/python'
    assert run.call_args_list[0].kwargs['remote'] is True
    assert run.call_args_list[1].args[0][0] == sys.executable


def test_latency_benchmark_retains_history_and_counts_forwards(tmp_path):
    import chess.pgn
    from eval.latency import benchmark, sample_positions
    game = chess.pgn.Game()
    board, node = game.board(), game
    for uci in ['e2e4', 'e7e5', 'g1f3', 'b8c6']:
        move = board.parse_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    path = tmp_path / 'games.pgn'
    path.write_text(str(game) + '\n')
    boards = sample_positions(path, count=10)
    assert [len(b.move_stack) for b in boards] == [0, 1, 2, 3]
    before = [b.fen() for b in boards]
    runner = ChessRunner(CountingModel(), device='cpu')
    result = benchmark(runner, boards, runs=4, warmup=1)
    assert result['neural_forward_calls'] == 5
    assert 0 < result['p50_ms'] <= result['p95_ms']
    assert [b.fen() for b in boards] == before


def test_arena_records_neural_calls_and_rejects_invalid_controls():
    a, b = [ChessRunner(CountingModel(), device='cpu') for _ in range(2)]
    result = play_match(a, b, num_games=2, max_plies=2)
    assert result['neural_forward_calls_a'] == result['moves_a'] == 2
    assert result['neural_forward_calls_b'] == result['moves_b'] == 2
    for kwargs in ({'clock_seconds': -1}, {'clock_seconds': float('nan')}, {'increment': -1},
                   {'openings': ['8/8/8/8/8/8/8/8 w - - 0 1']}):
        with pytest.raises(ValueError):
            play_match(a, b, num_games=2, **kwargs)
