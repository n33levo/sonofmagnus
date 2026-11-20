"""Fetch tournament PGNs and extract clean GM-versus-GM decisions."""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path

import chess.pgn
import zstandard

from chessio.records import holdout_positions
from model.artifacts import artifact_info
from scripts.fetch_data import fetch


def eligible(headers, min_elo):
    try:
        return (headers.get('Variant', 'Standard') == 'Standard' and
                headers.get('Result') in ('1-0', '0-1', '1/2-1/2') and
                all(headers.get(side + 'Title') == 'GM' and
                    int(headers.get(side + 'Elo', 0)) >= min_elo
                    for side in ('White', 'Black')))
    except ValueError:
        return False


def eval_drop(parent, child, turn):
    before, after = parent.eval(), child.eval()
    if before is None or after is None:
        return None
    return max(0, before.pov(turn).score(mate_score=10000) -
               after.pov(turn).score(mate_score=10000))


def build(archives, train, validation, exclude=(), min_elo=2500, max_positions=60000):
    train, validation = Path(train), Path(validation)
    manifest = train.with_suffix('.json')
    outputs = [train, validation, manifest]
    if min_elo < 0 or max_positions <= 0 or len(set(map(lambda p: p.resolve(), outputs))) != 3:
        raise ValueError('Use distinct outputs and positive settings')
    if any(path.exists() for path in outputs):
        raise FileExistsError('Corpus already exists')
    sources = {str(path): artifact_info(path) for path in archives}
    frozen_tests = {str(path): artifact_info(path) for path in exclude}
    blocked, seen, games = holdout_positions(exclude), set(), set()
    counts, players = Counter(), Counter()
    temporary = [path.with_suffix(path.suffix + '.part') for path in (train, validation)]
    class Grandmasters(chess.pgn.GameBuilder):
        def end_headers(self):
            if not eligible(self.game.headers, min_elo):
                return chess.pgn.SKIP
    try:
        with ExitStack() as stack:
            streams = [stack.enter_context(path.open('x')) for path in temporary]
            for archive in archives:
                with Path(archive).open('rb') as compressed:
                    raw = zstandard.ZstdDecompressor().stream_reader(compressed)
                    with io.TextIOWrapper(raw, encoding='utf-8-sig') as stream:
                        while (game := chess.pgn.read_game(stream, Visitor=Grandmasters)) is not None:
                            if counts['positions'] >= max_positions:
                                break
                            nodes = list(game.mainline())
                            if game.errors or len(nodes) < 16 or not eligible(game.headers, min_elo):
                                continue
                            root = game.board().fen()
                            game_id = hashlib.sha256((root + ' ' + ' '.join(
                                node.move.uci() for node in nodes)).encode()).hexdigest()
                            if game_id in games:
                                continue
                            games.add(game_id)
                            split = int(game_id[:8], 16) % 10 == 0
                            counts['games'] += 1
                            players.update(game.headers[side] for side in ('White', 'Black'))
                            board, history = game.board(), []
                            for node in nodes:
                                key = ' '.join(board.fen().split()[:4])
                                drop = eval_drop(node.parent, node, board.turn)
                                if key in blocked:
                                    counts['excluded'] += 1
                                elif key not in seen and board.fullmove_number >= 8 and not board.is_game_over(claim_draw=True):
                                    seen.add(key)
                                    if drop is not None and drop > 75:
                                        counts['recorded_eval_errors_skipped'] += 1
                                    else:
                                        row = {'fen': board.fen(), 'root_fen': root, 'history': history.copy(),
                                               'move': node.move.uci(), 'game_id': game_id, 'weight': 1.,
                                               'source': 'grandmaster-broadcast',
                                               'game_url': game.headers.get('GameURL'),
                                               'players': {side: {key: game.headers.get(side + key) for key in
                                                           ('Title', 'Elo', 'FideId')} | {'name': game.headers[side]}
                                                           for side in ('White', 'Black')}}
                                        if drop is not None:
                                            row['recorded_eval_drop_cp'] = drop
                                            counts['recorded_eval_pairs'] += 1
                                        streams[split].write(json.dumps(row, separators=(',', ':')) + '\n')
                                        counts['validation' if split else 'training'] += 1
                                        counts['positions'] += 1
                                board.push(node.move)
                                history.append(node.move.uci())
                                if counts['positions'] >= max_positions:
                                    break
                            if counts['games'] % 100 == 0:
                                print(json.dumps(dict(counts)), flush=True)
        if not counts['training'] or not counts['validation']:
            raise ValueError('Need both training and validation games')
        if any(artifact_info(path) != digest for path, digest in {**sources, **frozen_tests}.items()):
            raise ValueError('Archive or holdout changed')
        for partial, output in zip(temporary, (train, validation)):
            partial.replace(output)
        report = {'counts': dict(counts), 'min_elo': min_elo, 'titles': ['GM', 'GM'],
                  'license': 'CC-BY-SA-4.0', 'attribution': 'Lichess tournament broadcasts',
                  'source_page': 'https://database.lichess.org/#broadcasts', 'sources': sources,
                  'excluded': frozen_tests, 'split': 'SHA256 of complete game; 10% validation',
                  'position_duplicates': 'First occurrence only across both splits',
                  'recorded_eval_filter': 'Skip observed centipawn drops above 75; missing evaluations remain unverified',
                  'players': dict(players.most_common()), 'producer': artifact_info(__file__),
                  'training': artifact_info(train), 'validation': artifact_info(validation)}
        manifest.write_text(json.dumps(report, indent=2) + '\n')
        return report
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', default='2026-09')
    parser.add_argument('--out', default='data/gm_train.jsonl')
    parser.add_argument('--validation', default='data/gm_validation.jsonl')
    parser.add_argument('--exclude-data', required=True)
    parser.add_argument('--min-elo', type=int, default=2500)
    parser.add_argument('--positions', type=int, default=60000)
    args = parser.parse_args()
    name = f'lichess_db_broadcast_{args.month}.pgn.zst'
    downloads = json.loads(Path('data/gm_downloads.json').read_text())
    if name not in downloads:
        parser.error('Month is not in the pinned download manifest')
    archive = fetch(name, downloads[name], 'data')
    report = build([archive], args.out, args.validation, args.exclude_data.split(','), args.min_elo, args.positions)
    print(json.dumps(report['counts']), flush=True)
