import json
import shutil

import chess
import pytest

from model.artifacts import artifact_info
from scripts.check_relabels import check
from scripts.relabel_selfplay import relabel


def test_interrupted_labels_resume_and_refuse_changed_sources(tmp_path, monkeypatch):
    import scripts.relabel_selfplay as module
    source, output = tmp_path / 'source.jsonl', tmp_path / 'labels.jsonl'
    engine = tmp_path / 'engine'
    engine.write_bytes(b'pinned engine')
    board, rows = chess.Board(), []
    for move in ['e2e4', 'e7e5', 'g1f3']:
        rows.append({'fen': board.fen(), 'root_fen': chess.STARTING_FEN,
                     'history': [m.uci() for m in board.move_stack], 'predicted_value': .9})
        board.push_uci(move)
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    class Engine:
        id = {'name': 'test engine'}
        def configure(self, options): pass
        def quit(self): pass

    monkeypatch.setattr(module.chess.engine.SimpleEngine, 'popen_uci', lambda _: Engine())
    calls = []
    def labels(board, engine, *args):
        calls.append(board.fen())
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        legal = [m.uci() for m in board.legal_moves]
        return {'fen': board.fen(), 'value': .2, 'q_values': dict.fromkeys(legal, .6),
                'move_probs': {legal[0]: 1.}, 'label_depth_min': 12, 'label_depth_max': 12}
    monkeypatch.setattr(module, 'label_position', labels)
    with pytest.raises(RuntimeError, match='interrupted'):
        relabel(source, output, engine, workers=1)
    partial = output.with_suffix('.jsonl.part')
    assert len(partial.read_text().splitlines()) == 1
    with partial.open('a') as stream:
        stream.write('{"crash_truncated":')
    calls.clear()
    def completed(board, engine, *args):
        calls.append(board.fen())
        legal = [m.uci() for m in board.legal_moves]
        return {'fen': board.fen(), 'value': .2, 'q_values': dict.fromkeys(legal, .6),
                'move_probs': {legal[0]: 1.}, 'label_depth_min': 12, 'label_depth_max': 12}
    monkeypatch.setattr(module, 'label_position', completed)
    result = relabel(source, output, engine, workers=1)
    assert result['positions'] == 3 and len(calls) == 2
    assert result['artifact'] == artifact_info(output)
    assert all(row['value'] == .2 for row in map(json.loads, output.read_text().splitlines()))
    assert relabel(source, output, engine, workers=1) == result
    source.write_text(source.read_text() + json.dumps(rows[0]) + '\n')
    with pytest.raises(ValueError, match='changed'):
        relabel(source, output, engine, workers=1)


def test_relabel_real_stockfish_covers_all_actions_and_history(tmp_path):
    engine = shutil.which('stockfish')
    if not engine:
        pytest.skip('Stockfish not installed')
    board = chess.Board()
    board.push_uci('e2e4')
    source, output = tmp_path / 'source.jsonl', tmp_path / 'labels.jsonl'
    source.write_text(json.dumps({'fen': board.fen(), 'root_fen': chess.STARTING_FEN,
                                 'history': ['e2e4']}) + '\n')
    result = relabel(source, output, engine, depth=2, workers=1)
    row = json.loads(output.read_text())
    assert result['positions'] == 1 and row['history'] == ['e2e4']
    assert set(row['q_values']) == {m.uci() for m in board.legal_moves}
    assert row['label_depth_min'] >= 2
    assert row['value'] == pytest.approx(2 * max(row['q_values'].values()) - 1)
    assert sum(row['move_probs'].values()) == pytest.approx(1)
    assert check(source, output)['positions'] == 1
    row['q_values'].pop(next(iter(row['q_values'])))
    output.write_text(json.dumps(row) + '\n')
    manifest = json.loads(output.with_suffix('.json').read_text())
    manifest['artifact'] = artifact_info(output)
    output.with_suffix('.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incomplete legal'):
        check(source, output)
