import json,math,time
from collections import Counter
from pathlib import Path
import chess
from model.artifacts import artifact_info
from chessio.records import holdout_positions
path=Path('data/full_q_2500k.jsonl')
start=time.perf_counter();piece_counts=Counter();moves=Counter();depths=Counter();new_overlap=0;count=0
frozen=artifact_info(path)
blocked=holdout_positions(['data/puzzles_depth3000.jsonl'])
for count,line in enumerate(path.open(),1):
 row=json.loads(line);b=chess.Board(row['fen']);legal={m.uci() for m in b.legal_moves}
 assert b.is_valid() and len(legal)>=2
 assert set(row['q_values'])==legal
 assert all(math.isfinite(v) and 0<=v<=1 for v in row['q_values'].values())
 assert set(row['move_probs'])<=legal and all(math.isfinite(v) and 0<=v<=1 for v in row['move_probs'].values())
 assert abs(sum(row['move_probs'].values())-1)<1e-10
 assert abs(row['value']-(2*max(row['q_values'].values())-1))<1e-10
 assert row['label_depth_min']>=12 and row['label_depth_max']>=row['label_depth_min']
 moves[len(legal)]+=1;piece_counts[len(b.piece_map())]+=1;depths[row['label_depth_min']]+=1
 new_overlap += ' '.join(b.fen().split()[:4]) in blocked
 if count%250000==0:print(json.dumps({'validated':count}),flush=True)
assert count==2500000
assert artifact_info(path)==frozen
result={'positions_validated':count,'all_legal_action_values_verified':True,'side_to_move_value_consistency_verified':True,'probability_mass_verified':True,'board_validity_verified':True,'minimum_legal_moves':min(moves),'minimum_depth':min(depths),'legal_move_count_histogram':dict(moves),'piece_count_histogram':dict(piece_counts),'depth_histogram':dict(depths),'new_depth_holdout_overlap':new_overlap,'required_training_action':'Exclude the newer depth-conditioned test through --exclude-data; it was frozen after this extraction started','artifact':artifact_info(path),'seconds':time.perf_counter()-start}
Path('out/full_q_2500k_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
