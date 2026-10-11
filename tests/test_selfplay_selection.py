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
