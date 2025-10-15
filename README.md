# ChessHacks Bot - Searchless Neural Chess

A **single-pass neural chess bot** (no search) with **AlphaZero-style policy+value network** and **8×8×73 move head**. Designed for the [ChessHacks hackathon](https://chesshacks.dev/).

## Features

- **No search at inference** - Single forward pass per move
- **Legal move masking** - Zero illegal moves guaranteed
- **AlphaZero-style architecture** - Policy head (8×8×73) + Value head
- **3-phase training** - Behavioral cloning → Engine distillation → DAgger self-play
- **ONNX export + INT8 quantization** - Target ≤10 MB for Knight's Edge track
- **Comprehensive tests** - Legality, endings, no-search compliance

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+
- python-chess
- ONNX Runtime
- Stockfish (for offline distillation only)

### 2. Download Data

```bash
bash scripts/download_lichess.sh
```

This downloads a sample of Lichess CC0 games. Extract positions:

```bash
python -m train.dataset \
  --pgn data/lichess_2024-01.pgn.zst \
  --out data/positions.jsonl \
  --elo-min 1800 \
  --sample-rate 0.1 \
  --max-positions 1000000
```

### 3. Run Tests

```bash
make test
```

Verifies:
- ✓ Zero illegal moves (Gate G1)
- ✓ Ending conditions (stalemate, 50-move, threefold, promotions)
- ✓ No search in play/ module (Gate G5)

### 4. Train

#### Phase A: Behavioral Cloning

```bash
python -m train.train \
  --config configs/main.yaml \
  --data data/positions.jsonl \
  --epochs 2 \
  --amp bf16
```

#### Phase B: Engine Distillation

Curate hard positions:

```bash
python scripts/curate_hards.py \
  --pgn data/lichess_2024-01.pgn.zst \
  --out data/hard_positions.jsonl \
  --max-positions 50000
```

Generate engine labels:

```bash
python -m train.distill_labeler \
  --fens data/hard_positions.jsonl \
  --out data/labels_fast.jsonl \
  --preset fast \
  --max 50000
```

Train on labels:

```bash
python -m train.train \
  --config configs/main.yaml \
  --data data/labels_fast.jsonl \
  --epochs 1 \
  --amp bf16
```

#### Phase C: DAgger Self-Play

Generate self-play data:

```bash
python -m train.selfplay \
  --ckpt ckpts/latest.pt \
  --out data/selfplay.jsonl \
  --games 2000
```

Relabel with engine:

```bash
python -m train.distill_labeler \
  --fens data/selfplay.jsonl \
  --out data/selfplay_labels.jsonl \
  --preset fast
```

Fine-tune:

```bash
python -m train.train \
  --config configs/main.yaml \
  --data data/selfplay_labels.jsonl \
  --epochs 1 \
  --amp bf16 \
  --resume ckpts/latest.pt
```

### 5. Evaluate

#### Puzzles

```bash
python -m eval.puzzles \
  --ckpt ckpts/latest.pt \
  --puzzles data/puzzles.jsonl \
  --config main
```

#### Arena (vs baseline)

```bash
python -m eval.arena \
  --a ckpts/latest.pt \
  --b ckpts/baseline.pt \
  --games 100 \
  --config main
```

#### Move Agreement

```bash
python -m eval.agree \
  --ckpt ckpts/latest.pt \
  --labels data/holdout_labels.jsonl \
  --config main
```

### 6. Export & Submit

Promote a snapshot:

```bash
bash scripts/promote_snapshot.sh ckpts/latest.pt main
```

For **Knight's Edge** (≤10 MB):

```bash
bash scripts/promote_snapshot.sh ckpts/latest.pt mini
```

This will:
1. Export to ONNX (FP16)
2. Quantize to INT8 (mini only)
3. Verify size ≤10 MB and latency
4. Output: `out/latest_int8.onnx`

## Project Structure

```
chesshacks-bot/
├── io/                     # Board encoding & policy mapping
│   ├── encode.py           # board → 18-channel tensor
│   ├── policy_map.py       # 8×8×73 move encoding
│   └── mask.py             # Legality masking
├── model/                  # Neural network
│   ├── resnet.py           # ResNet backbone
│   ├── heads.py            # Policy + value heads
│   └── __init__.py         # Complete ChessNet
├── play/                   # Inference (NO SEARCH)
│   └── runner.py           # Single-pass move selection
├── train/                  # Training pipeline
│   ├── dataset.py          # Lichess data loading
│   ├── train.py            # AMP training loop
│   ├── distill_labeler.py  # Stockfish labeling (offline only)
│   └── selfplay.py         # DAgger self-play
├── eval/                   # Evaluation
│   ├── puzzles.py          # Tactical puzzles
│   ├── agree.py            # Move agreement
│   └── arena.py            # Head-to-head matches
├── deploy/                 # Export & quantization
│   ├── export_onnx.py      # PyTorch → ONNX
│   ├── quantize_int8.py    # INT8 quantization
│   └── size_latency_check.py
├── tests/                  # Compliance tests
│   ├── test_legality.py    # Gate G1
│   ├── test_endings.py     # Stalemate, 50-move, etc.
│   └── test_nosearch.py    # Gate G5
├── configs/                # Model configs
│   ├── main.yaml           # ~6-8M params
│   └── mini.yaml           # ~2-4M params (≤10 MB)
├── scripts/                # Helper scripts
│   ├── download_lichess.sh
│   ├── curate_hards.py
│   └── promote_snapshot.sh
├── Makefile
├── requirements.txt
└── README.md
```

## Model Architecture

### Input Encoding (18 channels)
- 12 piece planes (own/opponent × K,Q,R,B,N,P)
- 1 side-to-move
- 4 castling rights
- 1 halfmove clock (normalized)

### Network
- **Backbone**: 12 ResNet blocks, 128 channels (main) / 8 blocks, 64 channels (mini)
- **Policy Head**: Conv → 8×8×73 logits
- **Value Head**: Conv → GlobalAvgPool → FC → tanh (-1..1)

### Loss
```
L = CE(policy) + λ * MSE(value)    where λ = 0.25
```

### 8×8×73 Move Encoding
- Planes 0-55: Queen moves (8 directions × 7 distances)
- Planes 56-63: Knight moves (8 L-shaped)
- Planes 64-72: Underpromotions (N, NW, NE × Knight, Bishop, Rook)

## Acceptance Gates

✅ **G1 - Legality**: Illegal move rate = 0 on 10k random boards
✅ **G2 - Latency**: ≤100ms/move
✅ **G3 - Strength**: snapshot(N) ≥ snapshot(N-1) by +5% win-rate
✅ **G4 - Size**: ONNX INT8 ≤10 MB (Knight's Edge)
✅ **G5 - Compliance**: No `chess.engine` in runtime path

## Tracks

### Queen's Crown (Best Overall)
- Use `configs/main.yaml` (~6-8M params)
- Export FP16: `bash scripts/promote_snapshot.sh ckpts/latest.pt main`

### Knight's Edge (≤10 MB)
- Use `configs/mini.yaml` (~2-4M params)
- Export INT8: `bash scripts/promote_snapshot.sh ckpts/latest.pt mini`
- Verify: model size ≤10 MB

## Development Workflow

### During Hackathon

**Hour 0-3**: Wire datasets, smoke-test play, start Phase A
**Hour 3-8**: Finish Phase A → v0, quick arena vs baselines
**Hour 8-18**: Curate hards, label fast, train distill → v1
**Hour 18-26**: DAgger burst, fine-tune → v2
**Hour 26-33**: Export, INT8, size/latency gates
**Hour 33-36**: Final gauntlet + submission

### Iteration Loop

1. Train new snapshot
2. Run tests: `make test`
3. Evaluate: `python -m eval.arena --a ckpts/new.pt --b ckpts/old.pt`
4. If better: promote with `scripts/promote_snapshot.sh`
5. Submit ONNX model

## Tips

- **Compute tight?** Use mini config + stronger labels; skip Phase C
- **Quantization hurts?** Quantize trunk only; keep heads FP16
- **Data bottleneck?** Use streaming loaders; downsample games
- **Time crunch?** Fine-tune event starters on hard buckets

## Play a Game

```bash
python -m play.runner \
  --ckpt ckpts/latest.pt \
  --config main \
  --moves 50
```

## License

Code: MIT
Data: [Lichess CC0](https://database.lichess.org/)

## References

1. [ChessHacks Official Site](https://chesshacks.dev/)
2. [Lichess Open Database](https://database.lichess.org/)
3. [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
4. [ChessBench (Grandmaster-level without search)](https://arxiv.org/abs/2402.04494)
5. [python-chess Documentation](https://python-chess.readthedocs.io/)

---

Built for ChessHacks hackathon. Good luck! ♟️
