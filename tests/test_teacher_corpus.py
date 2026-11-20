import io
import json

import chess
import pytest
import zstandard

from scripts import build_teacher_corpus
from scripts.check_teacher import review
from train.dataset import position_sample
from train.distill_labeler import cp_to_q_value


def source_row(board, depth=16):
    moves = list(board.legal_moves)[:5]
    return {'fen': board.fen(), 'evals': [{'depth': depth, 'pvs': [
        {'line': m.uci(), 'cp': (300 - i * 100) * (1 if board.turn else -1)}
        for i, m in enumerate(moves)]}]}


@pytest.mark.parametrize('black', [False, True])
def test_partial_teacher_keeps_unknown_actions_masked_and_relative_scores(black):
    board = chess.Board()
    if black:
        board.push_uci('e2e4')
    row = source_row(board)
    label = build_teacher_corpus.teacher_label(row)
    best = row['evals'][0]['pvs'][0]['line']
    assert not label['complete_action_labels']
    assert len(label['q_values']) == 5
    assert label['centipawns'][best] == 300
    assert label['q_values'][best] == cp_to_q_value(300)
    assert max(label['move_probs'], key=label['move_probs'].get) == best
    sample = position_sample(label)
    assert (sample['q_value_target'] >= 0).sum() == 5
    assert sample['legal_mask'].sum() == board.legal_moves.count()
    assert sample['policy_target'].sum() == pytest.approx(1)


@pytest.mark.parametrize('failure', ['shallow', 'few_pvs', 'duplicate', 'illegal', 'nonfinite', 'low_material'])
def test_teacher_rejects_invalid_or_low_coverage_evaluations(failure):
    row = source_row(chess.Board())
    evaluation = row['evals'][0]
    if failure == 'shallow':
        evaluation['depth'] = 15
    elif failure == 'few_pvs':
        evaluation['pvs'].pop()
    elif failure == 'duplicate':
        evaluation['pvs'][-1] = evaluation['pvs'][0]
    elif failure == 'illegal':
        evaluation['pvs'][0]['line'] = 'a1a8'
    elif failure == 'nonfinite':
        evaluation['pvs'][0]['cp'] = float('inf')
    else:
        row['fen'] = '4k3/8/8/8/8/8/8/4K3 w - - 0 1'
    assert build_teacher_corpus.teacher_label(row) is None


def test_real_compressed_stream_excludes_continuations_and_deduplicates(tmp_path, monkeypatch):
    board = chess.Board()
    blocked = board.copy()
    board.push_uci('e2e4')
    continuation = board.copy()
    board.push_uci('e7e5')
    rows = [source_row(b) for b in (blocked, continuation, board, board)]
    board.push_uci('g1f3')
    rows.append(source_row(board))
    exclude = tmp_path / 'puzzle.jsonl'
    exclude.write_text(json.dumps({'fen': blocked.fen(), 'moves': ['e2e4']}) + '\n')
    packed = zstandard.ZstdCompressor().compress(''.join(json.dumps(r) + '\n' for r in rows).encode())

    class Source(io.BytesIO):
        headers = {'ETag': 'frozen-test'}

    monkeypatch.setattr(build_teacher_corpus, 'RangeReader', lambda _: Source(packed))
    output = tmp_path / 'teacher.jsonl'
    result = build_teacher_corpus.collect(output, 2, [exclude])
    assert result['heldout_excluded'] == 2
    assert result['duplicates_excluded'] == 1
    assert result['complete_action_rows'] == 0
    assert len(output.read_text().splitlines()) == 2
    assert review(output, [exclude], 2)['positions'] == 2
    before = output.read_bytes()
    with pytest.raises(ValueError):
        build_teacher_corpus.collect(output, 2, [exclude])
    assert output.read_bytes() == before


@pytest.mark.parametrize('failure', ['score', 'complete', 'policy', 'duplicate', 'count'])
def test_independent_review_rejects_corrupt_labels(tmp_path, failure):
    row = build_teacher_corpus.teacher_label(source_row(chess.Board()))
    first = next(iter(row['q_values']))
    if failure == 'score':
        row['q_values'][first] = .1
    elif failure == 'complete':
        row['complete_action_labels'] = True
    elif failure == 'policy':
        row['move_probs'][first] = .1
    path = tmp_path / 'labels.jsonl'
    path.write_text((json.dumps(row) + '\n') * (2 if failure == 'duplicate' else 1))
    with pytest.raises(ValueError):
        review(path, [], 2 if failure == 'count' else 1)
