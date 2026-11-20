from math import isfinite

from chessio.records import record_board


def review_label(row):
    """Admit completed external observations, including every requested root move."""
    board = record_board(row)
    legal = {move.uci() for move in board.legal_moves}
    actors = row['actor_choices']
    passes = row['teacher_observations']
    if not 1 <= len(actors) <= 2 or not set(actors) <= legal or not 1 <= len(passes) <= 2:
        raise ValueError('Invalid actor choices or teacher passes')
    first = passes[0]
    if first['kind'] != 'unrestricted' or first['observed'] is None:
        raise ValueError('Missing unrestricted teacher observation')
    scored = [first['observed']]
    if len(passes) == 2:
        restricted = passes[1]
        requested, scored = restricted['requested'], restricted['observed']
        candidate = row.get('played_grandmaster_move', row.get('screened_teacher_move'))
        expected = {first['observed']['move'], *actors, candidate}
        if (restricted['kind'] != 'restricted-actions' or set(requested) != expected or
                len(requested) != len(set(requested)) or len(scored) != len(requested) or
                any(info is None for info in scored) or {info['move'] for info in scored} != expected):
            raise ValueError('Incomplete requested root actions')
    elif actors != [first['observed']['move']]:
        raise ValueError('Unjudged actor action')
    for info in [first['observed'], *scored]:
        raw, wdl = info['raw'], info['wdl']
        cp = raw.get('cp')
        if 'mate' in raw:
            mate = raw['mate']
            if type(mate) is not int or not mate:
                raise ValueError('Invalid raw mate score')
            cp = (1 if mate > 0 else -1) * (100000 - 100 * min(abs(mate), 999))
        if (info['depth'] < 22 or info['move'] not in legal or type(cp) is not int or cp != info['cp'] or
                len(wdl) != 3 or any(type(v) is not int or v < 0 for v in wdl) or sum(wdl) != 1000):
            raise ValueError('Invalid depth, score or actual teacher WDL')
    scores = {info['move']: info['cp'] for info in scored}
    winner = max(scores, key=scores.get)
    costs = {move: scores[winner] - cp for move, cp in scores.items()}
    best = next(info for info in scored if info['move'] == winner)
    expected_status = ('below-recovery' if scores[winner] < -300 else
                       'source-safe' if costs[actors[0]] <= 40 else 'confirmed-correction')
    if (row['centipawns'] != scores or row['move'] != winner or row['alignment_status'] != expected_status or
            row['actor_costs'] != {move: costs[move] for move in actors} or row['teacher_wdl'] != best['wdl'] or
            not isfinite(row['value']) or row['value'] != (best['wdl'][0] - best['wdl'][2]) / 1000):
        raise ValueError('Saved labels disagree with raw external observations')
    if row['complete_action_labels'] != (set(scores) == legal):
        raise ValueError('Incorrect full-action label claim')
    rewards = row.get('reward_supervision')
    if expected_status == 'below-recovery':
        if rewards is not None:
            raise ValueError('Lost roots must not contribute policy costs')
    elif (not rewards or rewards['expert_depth_min'] < 22 or
          len(rewards['labelled_moves']) != len(set(rewards['labelled_moves'])) or
          len(rewards['labelled_moves']) != len(rewards['known_costs']) or
          dict(zip(rewards['labelled_moves'], rewards['known_costs'])) != costs):
        raise ValueError('Incorrect cost supervision')
    return expected_status
