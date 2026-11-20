import json
import sys

import chess
import chess.engine
import pytest

from model.artifacts import artifact_info
from train.elite_labels import review_label
from train.label_actions import main


@pytest.mark.parametrize('mode', ['complete', 'incomplete-action', 'bounded', 'missing-wdl', 'short-depth'])
@pytest.mark.parametrize('unbounded', [False, True])
def test_external_label_admission(tmp_path, monkeypatch, mode, unbounded):
    root = chess.Board()
    requests = tmp_path/'requests.jsonl'
    requests.write_text(json.dumps(dict(fen=root.fen(), move='e2e4', screened_teacher_move='e2e4',
                                        actor_choices=['d2d4']))+'\n')
    requests.with_suffix('.json').write_text(json.dumps(dict(artifact=artifact_info(requests), exclusions=[])))
    engine_path = tmp_path/'stockfish'
    engine_path.write_text('fixture')
    output = tmp_path/'labels.jsonl'

    class Engine:
        closed = False
        @property
        def id(self):
            if self.closed:
                raise chess.engine.EngineTerminatedError('engine event loop dead')
            return dict(name='Stockfish 19')
        def __enter__(self): return self
        def __exit__(self, *args): self.closed = True; return False
        def configure(self, options):
            assert options == {'Threads':1, 'Hash':256 if unbounded else 64, 'UCI_ShowWDL':True}
        def analysis(self, board, limit, **kwargs):
            if kwargs.get('multipv') is None:kwargs.pop('multipv')
            if kwargs.get('root_moves') is None:kwargs.pop('root_moves')
            raw = self.analyse(board, limit, **kwargs)
            events = [dict(v, multipv=i+1) for i, v in enumerate(raw)] if isinstance(raw, list) else [raw]
            class Stream:
                def __enter__(self): return self
                def __exit__(self, *args): return False
                def __iter__(self): return iter(events)
            return Stream()
        def analyse(self, board, limit, **kwargs):
            assert board.fen() == root.fen() and limit.depth == 22
            assert limit.time is None if unbounded else limit.time in (45, 90, 180)
            def info(move):
                value = dict(pv=[chess.Move.from_uci(move)], depth=21 if mode == 'short-depth' else 22,
                    score=chess.engine.PovScore(chess.engine.Cp(100 if move == 'e2e4' else -100), board.turn),
                    wdl=chess.engine.PovWdl(chess.engine.Wdl(700, 200, 100), board.turn))
                if mode == 'bounded': value['lowerbound'] = True
                if mode == 'missing-wdl': value.pop('wdl')
                return value
            if 'multipv' not in kwargs: return info('e2e4')
            assert {m.uci() for m in kwargs['root_moves']} == {'e2e4', 'd2d4'}
            return [info('e2e4')] if mode == 'incomplete-action' else [info('e2e4'), info('d2d4')]

    monkeypatch.setattr(chess.engine.SimpleEngine, 'popen_uci', lambda _:Engine())
    budget = ['--unbounded', '--hash-mib', '256'] if unbounded else []
    monkeypatch.setattr(sys, 'argv', ['label_actions', '--requests', str(requests), '--out', str(output), '--engine', str(engine_path), *budget])
    main()
    saved = list(map(json.loads, output.open()))
    report = json.loads(output.with_suffix('.json').read_text())
    assert report['engine_id'] == {'name':'Stockfish 19'}
    assert report['depth'] == 22 and report['wall_clock_cap'] == (not unbounded)
    assert report['uci_options']['Hash'] == (256 if unbounded else 64)
    if mode == 'complete':
        assert len(saved) == 1 and review_label(saved[0]) == 'confirmed-correction'
        assert saved[0]['actor_costs'] == {'d2d4':200}
        assert saved[0]['reward_supervision']['known_costs'] == [0, 200]
        def no_analysis(*args, **kwargs):
            raise AssertionError('Completed observations must be reused without engine work')
        monkeypatch.setattr(Engine, 'analyse', no_analysis)
        for part in (0, 1):
            reused = tmp_path/f'reused-{part}.jsonl'
            monkeypatch.setattr(sys, 'argv', ['label_actions', '--requests', str(requests), '--out', str(reused),
                '--engine', str(engine_path), '--reuse', str(output), '--part', str(part), '--parts', '2', *budget])
            main()
            assert list(map(json.loads, reused.open())) == (saved if part == 0 else [])
            assert json.loads(reused.with_suffix('.json').read_text())['reused'] == (1 if part == 0 else 0)
    else:
        assert saved == [] and sum(report['counts'].values()) == 1
        assert next(iter(report['counts'])).startswith('incomplete-')
