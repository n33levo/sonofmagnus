"""Verify aggregate checksums and recompute the published local point estimates."""
import hashlib
import json
from pathlib import Path

from eval.rating import fit_rating

ROOT=Path(__file__).resolve().parents[1]/'results'


def verify(directory=ROOT):
    for line in (directory/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,('Changed evidence',name)
    pools=json.loads((directory/'local-pool.json').read_text())
    for row in pools.values():
        calculated=float(fit_rating(row['score_by_opponent'],[engine['reference_elo'] for engine in row['anchors']]))
        assert abs(calculated-row['local_elo'])<1e-9
        assert row['elo_opening_bootstrap_95'][0]<calculated<row['elo_opening_bootstrap_95'][1]
        assert row['games']==400 and row['independent_opening_clusters']==50
    deployment=json.loads((directory/'deployment.json').read_text())
    assert deployment['completed_controlled_games']==10 and deployment['neural_moves_matched']==492
    assert deployment['unmatched_decisions']==0
    assert all(key not in deployment for key in ('username','games_reviewed','wins','losses','players'))
    return dict(local_estimates_recomputed={key:row['local_elo'] for key,row in pools.items()},
        checksums_verified=True,individual_game_logs='Private; public verification checks aggregates only')


if __name__=='__main__':print(json.dumps(verify(),indent=2))
