import io
import json
import stat

import chess
import chess.pgn
import pytest
import yaml

from play.uci import uci_loop
from scripts.lichess_access import account, save_token


@pytest.mark.parametrize('contents', ['', '{}', 'invalid json'])
def test_live_play_hold_blocks_launch_before_reading_credentials(tmp_path, contents):
    import os
    import subprocess
    import sys

    hold = tmp_path / '.config/sonofmagnus/rated-play-hold.json'
    hold.parent.mkdir(parents=True)
    hold.write_text(contents)
    output = tmp_path / 'run'
    result = subprocess.run([sys.executable, '-m', 'scripts.lichess_run', '--ckpt', str(tmp_path / 'absent.mlx'),
                             '--account', 'example-bot', '--out', str(output)],
                            env={**os.environ, 'HOME': str(tmp_path)}, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert 'New rated games are disabled by the user' in result.stderr
    assert 'FileNotFoundError' not in result.stderr
    assert not output.exists()


def test_live_play_without_a_hold_can_reach_the_existing_preflight(tmp_path, monkeypatch):
    from scripts import lichess_run
    monkeypatch.setattr(lichess_run, 'TOKEN_FILE', tmp_path / 'lichess.token')
    lichess_run.require_live_play_allowed()


def test_only_one_runner_can_hold_the_account_evaluation_lock(tmp_path):
    from scripts.lichess_run import acquire_evaluation_lock
    path = tmp_path / 'evaluation.lock'
    first = acquire_evaluation_lock(path)
    with pytest.raises(RuntimeError, match='already running'):
        acquire_evaluation_lock(path)
    first.close()
    second = acquire_evaluation_lock(path)
    second.close()


def test_bridge_inherits_the_lock_after_its_controller_closes(tmp_path):
    import subprocess
    import sys
    from scripts.lichess_run import acquire_evaluation_lock
    path = tmp_path / 'evaluation.lock'
    owner = acquire_evaluation_lock(path)
    child = subprocess.Popen([sys.executable, '-c', 'import sys; print("ready", flush=True); sys.stdin.read()'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, pass_fds=(owner.fileno(),))
    try:
        assert child.stdout.readline().strip() == 'ready'
        owner.close()
        with pytest.raises(RuntimeError, match='already running'):
            acquire_evaluation_lock(path)
    finally:
        owner.close()
        child.communicate('', timeout=10)
    released = acquire_evaluation_lock(path)
    released.close()


def test_five_game_controller_stops_after_fifth_finished_pgn(tmp_path, monkeypatch):
    from scripts import lichess_run

    def game(index, result):
        record = chess.pgn.Game()
        record.headers['Result'] = result
        (tmp_path / f'{index}.pgn').write_text(str(record) + '\n')

    for index in range(4):
        game(index, '1-0')
    game(4, '*')

    class Child:
        returncode = None
        signal = None

        def poll(self):
            return self.returncode

        def send_signal(self, value):
            self.signal = value

        def wait(self):
            self.returncode = 0
            return 0

    child = Child()
    monkeypatch.setattr(lichess_run.time, 'sleep', lambda _: game(4, '1-0'))
    assert lichess_run.finished_games(tmp_path) == 4
    assert lichess_run.wait_for_games(child, tmp_path, 5) == 0
    assert child.signal == lichess_run.signal.SIGINT
    assert lichess_run.finished_games(tmp_path) == 5


def test_prepared_run_freezes_serving_code_with_the_model(tmp_path, monkeypatch):
    from model.artifacts import artifact_info
    from scripts import lichess_run
    root = tmp_path / 'repo'
    for package in ('play', 'chessio', 'model'):
        folder = root / package
        folder.mkdir(parents=True)
        (folder / '__init__.py').write_text('value = 1\n')
    monkeypatch.setattr(lichess_run, 'ROOT', root)
    monkeypatch.setattr(lichess_run.subprocess, 'check_output',
                        lambda command, **kwargs: lichess_run.REVISION if command[-1] == 'HEAD' and 'rev-parse' in command else '')
    checkpoint = tmp_path / 'model.mlx'
    checkpoint.mkdir()
    (checkpoint / 'weights.safetensors').write_bytes(b'frozen model')
    output = tmp_path / 'run'
    manifest = lichess_run.prepare(checkpoint, output, tmp_path / 'bridge')
    runtime = output / manifest['runtime_directory']
    for source, digest in manifest['runtime_source'].items():
        assert artifact_info(runtime / source)['sha256'] == digest
    (root / 'play/__init__.py').write_text('value = 2\n')
    assert (runtime / 'play/__init__.py').read_text() == 'value = 1\n'
    assert repr(str(runtime)) in (output / 'engine.py').read_text()
    assert manifest['model'] == artifact_info(output / checkpoint.name)
    config = yaml.safe_load((output / 'config.yml').read_text())
    minutes, increment = map(int, manifest['clock'].split('+'))
    assert config['challenge']['min_base'] == config['challenge']['max_base'] == minutes * 60
    assert config['challenge']['min_increment'] == config['challenge']['max_increment'] == increment
    assert config['matchmaking']['challenge_initial_time'] == [minutes * 60]
    assert config['matchmaking']['challenge_increment'] == [increment]
    assert config['matchmaking']['challenge_filter'] == 'fine'


def test_token_is_hidden_private_and_never_overwrites_existing(tmp_path, monkeypatch, capsys):
    secret = 'fake-token-not-a-credential'
    monkeypatch.setattr('scripts.lichess_access.getpass.getpass', lambda _: secret)
    path = tmp_path / 'config/lichess.token'
    save_token(path)
    assert secret not in capsys.readouterr().out
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text().strip() == secret
    with pytest.raises(FileExistsError):
        save_token(path)
    assert path.read_text().strip() == secret


def test_access_refuses_a_different_account_or_malformed_token(monkeypatch):
    received = []

    def respond(request, timeout, context):
        assert context.check_hostname
        received.append(request)
        return io.BytesIO(json.dumps({'id': 'mybot', 'title': 'BOT'}).encode())

    monkeypatch.setattr('scripts.lichess_access.urllib.request.urlopen', respond)
    assert account('fake-token', 'MyBot')['title'] == 'BOT'
    with pytest.raises(ValueError, match='different'):
        account('fake-token', 'otherbot')
    with pytest.raises(ValueError, match='Invalid token'):
        account('fake\nsecret', 'mybot')
    assert len(received) == 2
    assert received[0].full_url == 'https://lichess.org/api/account'


@pytest.mark.parametrize('forwards', [1, 2])
def test_uci_audit_reconstructs_history_and_rejects_extra_neural_passes(forwards):
    class Runner:
        forward_calls = 0
        last_latency_ms = 2.5

        def select_move(self, board, deterministic):
            assert deterministic
            self.forward_calls += forwards
            return chess.Move.from_uci('g1f3')

    audit, output = io.StringIO(), io.StringIO()
    commands = io.StringIO('position startpos moves e2e4 e7e5\ngo wtime 180000 btime 180000\nquit\n')
    if forwards != 1:
        with pytest.raises(ValueError, match='one network pass'):
            uci_loop(Runner(), commands, output, audit)
        assert not output.getvalue() and not audit.getvalue()
    else:
        uci_loop(Runner(), commands, output, audit)
        row = json.loads(audit.getvalue())
        board = chess.Board(row['root_fen'])
        for move in row['history']:
            board.push_uci(move)
        assert board.fen() == row['fen']
        assert board.parse_uci(row['move']) in board.legal_moves
        assert row['neural_forward_calls'] == 1
        assert row['latency_ms'] == 2.5
        assert 'bestmove g1f3' in output.getvalue()
