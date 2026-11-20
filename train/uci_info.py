def complete_analysis(engine, board, limit, *, multipv=None, root_moves=None, game=None):
    """Keep atomic, exact UCI records from the last complete MultiPV iteration."""
    count = 1 if multipv is None else multipv
    if type(count) is not int or count < 1:
        raise ValueError('MultiPV must be a positive integer')
    allowed = set(root_moves) if root_moves is not None else None
    pending, reported = {}, {}
    pending_depth, complete = None, None
    with engine.analysis(board, limit, multipv=multipv, root_moves=root_moves, game=game) as analysis:
        for info in analysis:
            rank, depth = info.get('multipv', 1), info.get('depth', 0)
            if not 1 <= rank <= count:
                continue
            reported[rank] = max(depth, reported.get(rank, 0))
            if 'score' not in info or not info.get('pv'):
                continue
            if rank == 1:
                pending, pending_depth = {}, depth
            if depth != pending_depth:
                continue
            if info.get('lowerbound') or info.get('upperbound') or 'wdl' not in info:
                pending.pop(rank, None)
                continue
            pending[rank] = dict(info, pv=list(info['pv']))
            if len(pending) == count:
                moves = {r['pv'][0] for r in pending.values()}
                if len(moves) == count and (allowed is None or moves <= allowed):
                    complete = [dict(pending[r]) for r in range(1, count+1)]
    if complete is None:
        complete = [dict(depth=reported.get(r, 0), incomplete=True) for r in range(1, count+1)]
    return complete[0] if multipv is None else complete
