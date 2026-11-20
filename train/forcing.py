"""Legal off-line branches for supervised replies to checks and threats."""
from collections import Counter
import hashlib
import json
import random
import re

import chess

from chessio.records import record_board


def game_ids(row):
    identifiers = set()
    for name in ('game_id', 'origin_game_id', 'source_game', 'game_url'):
        value = row.get(name)
        if not isinstance(value, str):
            continue
        identifiers.add(value)
        match = re.fullmatch(r'external-([A-Za-z0-9]{8}|[a-f0-9]{64})-[01]', value)
        if match:
            identifiers.add(match[1])
        identifiers.update(re.findall(r'https://lichess.org/([A-Za-z0-9]{8})(?:[A-Za-z0-9]{4})?(?:$|[#/?])', value))
    return identifiers


def position_key(board):
    return ' '.join(board.fen().split()[:4])


def branches(rows, blocked, denied, limit, seed, max_per_game=2):
    if type(limit) is not int or limit < 1 or type(max_per_game) is not int or max_per_game < 1:
        raise ValueError('Positive integer limits required')
    rng, candidates, counts = random.Random(seed), [], Counter()
    ordered = list(rows)
    rng.shuffle(ordered)
    for row in ordered:
        if game_ids(row) & denied:
            counts['protected-source-game'] += 1
            continue
        board = record_board(row)
        legal = list(board.legal_moves)
        rng.shuffle(legal)
        kept = 0
        for move in legal:
            if move.uci() == row['move']:
                continue
            check, capture = board.gives_check(move), board.is_capture(move)
            child = board.copy()
            child.push(move)
            kind = 'check' if check else 'promotion' if move.promotion else 'capture' if capture else None
            if kind is None:
                targets = child.pieces_mask(chess.QUEEN, not board.turn) | child.pieces_mask(chess.ROOK, not board.turn)
                newly_attacked = (int(child.attacks(move.to_square)) & targets) & ~int(board.attacks(move.from_square))
                if newly_attacked:
                    kind = 'direct-attack'
            if kind is None or child.is_game_over(claim_draw=True) or len(child.piece_map()) < 6 or child.legal_moves.count() < 2:
                continue
            if {position_key(child), position_key(child.mirror())} & blocked:
                counts['protected-position'] += 1
                continue
            history = [*row['history'], move.uci()]
            identity = hashlib.sha256(json.dumps([row['root_fen'], history], separators=(',', ':')).encode()).hexdigest()
            item = dict(game_id='forcing-'+identity[:24], origin_game_id=row['game_id'],
                fen=child.fen(), root_fen=row['root_fen'], history=history,
                history_origin='legal-forcing-branch', branch_move=move.uci(), branch_kind=kind,
                parent_fen=row['fen'], parent_supervised_move=row['move'],
                parent_record=row['parent_record'], origin_ratings=row['origin_ratings'],
                source='forcing-recovery-proposal', proposal_only=True, policy_supervision=False,
                actual_gpu_actor_review_pending=True)
            candidates.append(item)
            kept += 1
            if kept == 4:
                break
    rng.shuffle(candidates)
    selected, seen, per_game = [], set(), Counter()
    for item in candidates:
        group = item['origin_game_id']
        if item['game_id'] in seen or per_game[group] >= max_per_game:
            continue
        selected.append(item)
        per_game[group] += 1
        seen.add(item['game_id'])
        if len(selected) == limit:
            break
    counts.update({'source-rows': len(rows), 'candidate-branches': len(candidates),
        'selected': len(selected), 'source-games': len(per_game)})
    return selected, dict(counts)
