import argparse
import json
import math
from pathlib import Path

from play.runner import load_runner
from chessio.records import record_board
from model.artifacts import artifact_info


def evaluate_agreement(model_path, labels_path, config='main', device='auto', max_samples=None, top_k=(1, 3, 5)):
    runner = load_runner(model_path, config, device)
    correct = dict.fromkeys(top_k, 0)
    total, divergences = 0, []
    with open(labels_path) as source:
        for line in source:
            if max_samples is not None and total >= max_samples:
                break
            item = json.loads(line)
            board = record_board(item)
            teacher = item.get('move_probs', {item.get('move'): 1.})
            best = max(teacher, key=teacher.get)
            probs = {move.uci(): p for move, p in runner.move_probs(board).items()}
            if best not in probs:
                raise ValueError(f'Illegal teacher move {best} at {board.fen()}')
            ranking = sorted(probs, key=probs.get, reverse=True)
            for k in correct:
                correct[k] += best in ranking[:k]
            mass = sum(teacher.values())
            divergences.append(sum((p / mass) * math.log((p / mass) / max(probs.get(move, 0), 1e-300))
                                   for move, p in teacher.items() if p > 0))
            total += 1
    if not total:
        raise ValueError('Empty agreement dataset')
    return {**{f'top{k}_agreement': n / total for k, n in correct.items()},
            'total': total, 'mean_kl_divergence': sum(divergences) / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--config', choices=['main', 'mini'], default='main')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max', type=int)
    parser.add_argument('--top-k', default='1,3,5')
    parser.add_argument('--out')
    args = parser.parse_args()
    model, dataset = artifact_info(args.ckpt), artifact_info(args.labels)
    results = evaluate_agreement(args.ckpt, args.labels, args.config, args.device, args.max,
                                 tuple(int(k) for k in args.top_k.split(',')))
    if model != artifact_info(args.ckpt) or dataset != artifact_info(args.labels):
        raise ValueError('Agreement artifacts changed during evaluation')
    results.update(model=model, dataset=dataset, runtime_source={path: artifact_info(path)['sha256'] for path in
        ('eval/agree.py', 'play/runner.py', 'play/mlx.py', 'chessio/records.py', 'chessio/lc0.py')})
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
