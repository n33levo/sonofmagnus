import json

import chess
import pytest

from model.artifacts import artifact_info
from scripts.select_selfplay import select


def test_selective_labeling_keeps_exact_budget_and_most_uncertain_distinct_inputs(tmp_path):
    source, output = tmp_path / 'selfplay.jsonl', tmp_path / 'selected.jsonl'
    rows = []
    for index in range(100):
        board = chess.Board()
        board.halfmove_clock = index
        rows.append({'fen': board.fen(), 'policy_entropy': index / 100})
    source.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    source.with_suffix('.json').write_text(json.dumps({'artifact': artifact_info(source), 'positions': 100}))
    result = select(source, output)
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert result['selected_positions'] == len(selected) == 15
    assert min(row['policy_entropy'] for row in selected) == .85
    with pytest.raises(ValueError, match='new output'):
        select(source, output)
    source.write_text(source.read_text() + json.dumps(rows[0]) + '\n')
    with pytest.raises(ValueError, match='differs'):
        select(source, tmp_path / 'changed.jsonl')


def test_selection_preserves_phases_when_endgame_entropy_is_higher(tmp_path):
    source, output = tmp_path / 'source.jsonl', tmp_path / 'selected.jsonl'
    rows = []
    for index in range(20):
        opening = chess.Board()
        ending = chess.Board('8/8/8/8/8/2k5/8/K6R w - - 0 1')
        opening.halfmove_clock = ending.halfmove_clock = index
        rows.extend([{'fen': opening.fen(), 'policy_entropy': .1 + index / 100},
                     {'fen': ending.fen(), 'policy_entropy': .8 + index / 100}])
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    source.with_suffix('.json').write_text(json.dumps({'artifact': artifact_info(source), 'positions': 40}))
    result = select(source, output, fraction=.15)
    selected = [json.loads(line) for line in output.read_text().splitlines()]
    assert result['selected_by_phase'] == {0: 3, 3: 3}
    assert sorted(row['policy_entropy'] for row in selected) == pytest.approx([.27, .28, .29, .97, .98, .99])
