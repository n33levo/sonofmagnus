import argparse
from datetime import datetime, timezone
import json
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import chess.pgn
import yaml

from model.artifacts import artifact_info
from scripts.lichess_access import TOKEN_FILE, account


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = Path.home() / '.cache/sonofmagnus/lichess-bot'
REVISION = 'df7e730de58cc3ef2f1415a0dc2eeda842d39167'


def require_live_play_allowed():
    hold = TOKEN_FILE.with_name('rated-play-hold.json')
    if hold.exists():
        raise RuntimeError(f'New rated games are disabled by the user: {hold}')


def acquire_evaluation_lock(path):
    handle = Path(path).open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('A rated evaluation is already running for this token file') from None
    return handle


def finished_games(directory):
    count = 0
    for path in Path(directory).glob('*.pgn'):
        with path.open() as stream:
            game = chess.pgn.read_game(stream)
        if game and not game.errors and game.headers.get('Result') in ('1-0', '0-1', '1/2-1/2'):
            count += 1
    return count


def wait_for_games(child, directory, target):
    while child.poll() is None:
        if finished_games(directory) >= target:
            child.send_signal(signal.SIGINT)
            return child.wait()
        time.sleep(.25)
    return child.returncode


def prepare(checkpoint, output, bridge=BRIDGE):
    checkpoint, output, bridge = Path(checkpoint).resolve(), Path(output).resolve(), Path(bridge).resolve()
    revision = subprocess.check_output(['git', '-C', str(bridge), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION or subprocess.check_output(['git', '-C', str(bridge), 'diff', 'HEAD'], text=True):
        raise ValueError('Use the pinned, unchanged lichess-bot checkout')
    frozen = artifact_info(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    model = output / checkpoint.name
    if checkpoint.is_dir():
        shutil.copytree(checkpoint, model)
    else:
        shutil.copy2(checkpoint, model)
    if artifact_info(model) != frozen or artifact_info(checkpoint) != frozen:
        raise ValueError('Model changed while freezing the online run')
    sources = {str(path.relative_to(ROOT)): artifact_info(path)['sha256'] for package in ('play', 'chessio', 'model')
               for path in (ROOT / package).glob('*.py')}
    runtime = output / 'runtime'
    for source, digest in sources.items():
        destination = runtime / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, destination)
        if artifact_info(destination)['sha256'] != digest or artifact_info(ROOT / source)['sha256'] != digest:
            raise ValueError('Runtime source changed while freezing the online run')
    wrapper = output / 'engine.py'
    wrapper.write_text(f'import sys\nsys.path.insert(0, {str(runtime)!r})\nfrom play.uci import main\nmain()\n')
    wrapper.chmod(0o700)
    # Disable all chat responses as well as configured greetings.
    bot = output / 'bot.py'
    bot.write_text(f"import sys\nsys.path.insert(0, {str(bridge)!r})\nfrom lib.lichess import Lichess\nLichess.chat = lambda *args, **kwargs: None\nfrom lib.lichess_bot import start_program\nif __name__ == '__main__':\n    start_program()\n")
    config = {
        'token': 'loaded-from-local-environment', 'url': 'https://lichess.org/',
        'engine': {'dir': str(output), 'name': wrapper.name, 'working_dir': str(ROOT), 'protocol': 'uci',
            'interpreter': sys.executable, 'ponder': False, 'uci_ponder': False,
            'engine_options': {'ckpt': str(model), 'audit-log': str(output / 'moves.jsonl')},
            'polyglot': {'enabled': False}, 'uci_options': {},
            'online_moves': {name: {'enabled': False} for name in
                ('chessdb_book', 'lichess_cloud_analysis', 'lichess_opening_explorer', 'online_egtb')},
            'lichess_bot_tbs': {name: {'enabled': False} for name in ('syzygy', 'gaviota')},
            'draw_or_resign': {'resign_enabled': False, 'offer_draw_enabled': False}},
        'challenge': {'concurrency': 1, 'accept_bot': True, 'only_bot': False,
            'variants': ['standard'], 'time_controls': ['blitz'], 'modes': ['rated'],
            'min_base': 180, 'max_base': 180, 'min_increment': 2, 'max_increment': 2,
            'max_takebacks_accepted': 0},
        'matchmaking': {'allow_matchmaking': True, 'allow_during_games': False, 'challenge_timeout': 1,
            'challenge_initial_time': [180], 'challenge_increment': [2], 'challenge_variant': 'standard',
            'challenge_mode': 'rated', 'opponent_rating_difference': 400, 'rating_preference': 'none',
            'challenge_filter': 'fine'},
        'greeting': {name: '' for name in ('hello', 'goodbye', 'hello_spectators', 'goodbye_spectators')},
        'max_takebacks_accepted': 0, 'fake_think_time': False, 'quit_after_all_games_finish': True,
        'pgn_directory': str(output / 'games'), 'pgn_file_grouping': 'game'
    }
    (output / 'config.yml').write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {'model': frozen, 'source_checkpoint': str(checkpoint), 'bridge_revision': revision,
                'bridge_source': 'https://github.com/lichess-bot-devs/lichess-bot', 'bridge_license': 'AGPL-3.0',
                'runtime_source': sources, 'runtime_directory': 'runtime', 'clock': '3+2', 'rated': True, 'variant': 'standard',
                'protocol': 'One deterministic neural forward pass per move; no search or external move sources',
                'status': 'Prepared locally; no online games played'}
    manifest['launch_files'] = {path.name: artifact_info(path) for path in (wrapper, bot, output / 'config.yml')}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the frozen searchless engine through the Lichess BOT bridge')
    parser.add_argument('--ckpt', required=True, type=Path)
    parser.add_argument('--account', required=True)
    parser.add_argument('--token-file', type=Path, default=TOKEN_FILE)
    parser.add_argument('--bridge', type=Path, default=BRIDGE)
    parser.add_argument('--out', type=Path, default=ROOT / 'out' / ('lichess-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    parser.add_argument('--prepare', action='store_true', help='Freeze and validate the configuration without contacting Lichess')
    parser.add_argument('--max-games', type=int, help='Stop after this many finished local game records')
    args = parser.parse_args()
    if args.max_games is not None and args.max_games < 1:
        parser.error('max-games must be positive')
    args.out, args.bridge = args.out.resolve(), args.bridge.resolve()
    token, before = None, None
    if not args.prepare:
        require_live_play_allowed()
        evaluation_lock = acquire_evaluation_lock(args.token_file.with_name('lichess-run.lock'))
        token = args.token_file.read_text().strip()
        before = account(token, args.account)
        if before.get('title') != 'BOT':
            parser.error('The specified account must be upgraded to BOT first')
    manifest = prepare(args.ckpt, args.out, args.bridge)
    manifest.update(account=args.account, rating_before=before.get('perfs', {}).get('blitz') if before else None,
                    games_requested=args.max_games)
    path = args.out / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.prepare:
        print(f'Prepared local run: {args.out}')
    else:
        manifest['status'] = 'Online evaluation running; rating not yet established'
        manifest['started_ms'] = int(datetime.now(timezone.utc).timestamp() * 1000)
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        environment = {**os.environ, 'LICHESS_BOT_TOKEN': token}
        command = [str(args.bridge / '.venv/bin/python'), str(args.out / 'bot.py'), '--config', str(args.out / 'config.yml'),
                   '--logfile', str(args.out / 'bridge.log')]
        try:
            child = subprocess.Popen(command, cwd=args.bridge, env=environment, pass_fds=(evaluation_lock.fileno(),))
            try:
                returncode = (wait_for_games(child, args.out / 'games', args.max_games)
                              if args.max_games else child.wait())
            except KeyboardInterrupt:
                child.send_signal(signal.SIGINT)
                print('Waiting for current games to finish before stopping')
                returncode = child.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
            if args.max_games and finished_games(args.out / 'games') != args.max_games:
                raise ValueError('Rated run did not finish exactly the requested number of local games')
        finally:
            unchanged = all(artifact_info(args.out / manifest['runtime_directory'] / source)['sha256'] == digest for source, digest in manifest['runtime_source'].items())
            manifest.update(status='Runner stopped; games and rating require review', runtime_source_unchanged=unchanged,
                            model_unchanged=artifact_info(args.out / args.ckpt.name) == manifest['model'])
            try:
                manifest['rating_after'] = account(token, args.account).get('perfs', {}).get('blitz')
            except OSError:
                manifest['rating_after'] = None
            path.write_text(json.dumps(manifest, indent=2) + '\n')
