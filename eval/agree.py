"""
Move agreement evaluation.

Measures how often the model agrees with teacher labels.
"""

import argparse
import json

import chess
import numpy as np
import torch

from chessio.encode import encode_board
from chessio.mask import get_legal_move_probs
from model import create_model_main, create_model_mini


def evaluate_agreement(
    model_path: str,
    labels_path: str,
    config: str = "main",
    device: str = "cpu",
    max_samples: int = None,
) -> dict:
    """
    Evaluate move agreement with teacher labels.

    Args:
        model_path: Path to model checkpoint
        labels_path: Path to teacher labels JSONL
        config: Model config
        device: Device
        max_samples: Max samples to evaluate

    Returns:
        Dictionary with agreement metrics
    """
    # Load model
    if config == "main":
        model = create_model_main()
    else:
        model = create_model_mini()

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load labels
    labels = []
    with open(labels_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            labels.append(json.loads(line))

    print(f"Evaluating agreement on {len(labels)} positions...")

    top1_agree = 0
    top3_agree = 0
    total = 0

    kl_divs = []

    with torch.no_grad():
        for i, item in enumerate(labels):
            board = chess.Board(item["fen"])

            # Encode and predict
            board_tensor = encode_board(board)
            board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(device)

            policy_logits, _ = model(board_tensor)
            policy_logits = policy_logits[0].cpu().numpy()

            # Get predicted move probabilities
            pred_probs = get_legal_move_probs(policy_logits, board)

            # Get teacher move(s)
            if "move" in item:
                teacher_move = chess.Move.from_uci(item["move"])
                teacher_moves = [teacher_move]
            elif "move_probs" in item:
                # Top-3 teacher moves
                teacher_move_probs = {
                    chess.Move.from_uci(uci): prob
                    for uci, prob in item["move_probs"].items()
                }
                teacher_moves = sorted(
                    teacher_move_probs.keys(),
                    key=lambda m: teacher_move_probs[m],
                    reverse=True
                )[:3]

                # Compute KL divergence
                kl = 0.0
                for move in board.legal_moves:
                    p_teacher = teacher_move_probs.get(move, 0.0)
                    p_pred = pred_probs.get(move, 1e-9)
                    if p_teacher > 0:
                        kl += p_teacher * np.log(p_teacher / p_pred)
                kl_divs.append(kl)
            else:
                continue

            # Predicted top moves
            pred_top_moves = sorted(pred_probs.keys(), key=lambda m: pred_probs[m], reverse=True)

            # Top-1 agreement
            if pred_top_moves[0] == teacher_moves[0]:
                top1_agree += 1

            # Top-3 agreement
            if any(pm in teacher_moves for pm in pred_top_moves[:3]):
                top3_agree += 1

            total += 1

            if (i + 1) % 100 == 0:
                print(f"  Evaluated {i + 1}/{len(labels)} positions...")

    results = {
        "top1_agreement": top1_agree / total if total > 0 else 0.0,
        "top3_agreement": top3_agree / total if total > 0 else 0.0,
        "total": total,
    }

    if kl_divs:
        results["mean_kl_divergence"] = float(np.mean(kl_divs))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate move agreement")
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--labels", type=str, required=True, help="Teacher labels JSONL")
    parser.add_argument("--config", type=str, choices=["main", "mini"], default="main")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max", type=int, help="Max samples to evaluate")
    args = parser.parse_args()

    results = evaluate_agreement(
        model_path=args.ckpt,
        labels_path=args.labels,
        config=args.config,
        device=args.device,
        max_samples=args.max,
    )

    print("\nAgreement Results:")
    print(f"  Top-1 agreement: {results['top1_agreement']:.2%}")
    print(f"  Top-3 agreement: {results['top3_agreement']:.2%}")
    if "mean_kl_divergence" in results:
        print(f"  Mean KL divergence: {results['mean_kl_divergence']:.4f}")
    print(f"  Total positions: {results['total']}")


if __name__ == "__main__":
    main()
