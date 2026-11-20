import chess
import chess.engine
import pytest

from train.uci_info import complete_analysis


def info(depth, rank, move, **extra):
    return dict(depth=depth, multipv=rank, pv=[chess.Move.from_uci(move)],
                score=chess.engine.PovScore(chess.engine.Cp(depth), chess.WHITE),
                wdl=chess.engine.PovWdl(chess.engine.Wdl(300, 500, 200), chess.WHITE), **extra)


class Engine:
    def __init__(self, events): self.events = events
    def analysis(self, board, limit, **kwargs):
        assert board.fen() == chess.STARTING_FEN and limit.depth == 22
        class Stream:
            def __enter__(stream): return stream
            def __exit__(stream, *args): return False
            def __iter__(stream): return iter(self.events)
        return Stream()


def run(events, multipv=None):
    return complete_analysis(Engine(events), chess.Board(), chess.engine.Limit(depth=22), multipv=multipv)


def test_stale_bound_flag_does_not_poison_a_later_exact_observation():
    bounded = info(22, 1, 'e2e4', lowerbound=True)
    exact = info(22, 1, 'd2d4')
    assert {**bounded, **exact}['lowerbound']  # The library's aggregate retains it.
    result = run([bounded, exact])
    assert result == exact and 'lowerbound' not in result


def test_partial_deeper_iteration_does_not_mix_with_earlier_candidates():
    complete = [info(21, 1, 'e2e4'), info(21, 2, 'd2d4')]
    assert run([*complete, info(22, 1, 'd2d4')], 2) == complete


def test_all_candidates_come_from_one_complete_exact_iteration():
    complete = [info(22, 1, 'd2d4'), info(22, 2, 'e2e4')]
    result = run([info(21, 1, 'e2e4'), info(21, 2, 'd2d4'), *complete,
                  info(23, 1, 'g1f3', upperbound=True)], 2)
    assert result == complete


def test_missing_wdl_is_not_filled_from_a_previous_depth():
    earlier, later = info(21, 1, 'e2e4'), info(22, 1, 'd2d4')
    later.pop('wdl')
    assert run([earlier, later]) == earlier


def test_repeated_first_rank_starts_a_new_iteration():
    result = run([info(22, 1, 'e2e4'), info(22, 2, 'd2d4', lowerbound=True),
                  info(22, 1, 'd2d4')], 2)
    assert all(r['incomplete'] and not r.get('pv') for r in result)


def test_duplicate_candidate_moves_do_not_make_a_complete_iteration():
    result = run([info(22, 1, 'e2e4'), info(22, 2, 'e2e4')], 2)
    assert all(r['incomplete'] for r in result)


def test_empty_stream_reports_no_scored_observation():
    assert run([]) == dict(depth=0, incomplete=True)


def test_unrequested_candidate_is_not_accepted():
    result = complete_analysis(Engine([info(22, 1, 'e2e4')]), chess.Board(),
        chess.engine.Limit(depth=22), root_moves=[chess.Move.from_uci('d2d4')])
    assert result['incomplete']


@pytest.mark.parametrize('multipv', [0, -1, 1.5, True])
def test_invalid_multipv_is_rejected(multipv):
    with pytest.raises(ValueError, match='positive integer'):
        run([], multipv)
