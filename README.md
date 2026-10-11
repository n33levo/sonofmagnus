# sonofmagnus

A searchless chess engine. We encode the current position and its history, run one network pass, mask illegal moves and choose a move. No tree search, successor-position evaluation or engine calls during play.

We train compact policies and run larger policies on Apple Metal. The GPU runtime uses fused attention and packed matrix weights. Training can use Stockfish offline; inference cannot.

The strength target remains **2500–2650 Elo**, with **2600** for the final model. The compact deployment target remains **8.7 MB at 8 bits / 24 MB FP16**, **<100 ms CPU / <20 ms GPU**, and one forward pass per move. These targets need to hold together for one tested model.

## What we have measured

Frozen datasets, results and game logs live in [data](data) and [out](out).

| Model | Stored weights | Full move-selection p95 | Complete puzzles | Stockfish depth-14 top-1 |
|---|---:|---:|---:|---:|
| sonofmagnus-102M | 205.1 MB, FP16 MLX | 10.9 ms, isolated GPU benchmark | 717/1000 | 62.95%, 2000 positions |
| sonofmagnus-185M | 163.6 MB, 6-bit MLX | 17.6 ms, isolated GPU benchmark | 784/1000 | 63.15%, 2000 positions |
| sonofmagnus-6M, first training pass | 12.09 MB FP16 / 7.41 MB at 8 bits | 15.9 ms, isolated FP16 GPU benchmark | 277/1000 | 38.25%, 2000 positions |

Parameter labels are rounded; export manifests record exact stored scalar counts and bytes. Puzzle scoring requires the whole solution, with one deterministic attempt. The 102M model clears the five original puzzle-band point targets: **95%, 82%, 68%, 47%, 23%**. Sampling uncertainty still applies.

**The 185M checkpoint measured 2853 local Elo** in [400 recorded games](out/bt4_ccrl_pool), with 192 wins, 94 draws and 114 losses at 10+0.1. The paired-opening 95% interval is **2819–2886**; the conservative bound is **2687–3037**. We replayed all 41,863 plies and verified one network call for each of its 20,963 moves, with no forfeits or capped games.

The pool uses published CCRL Blitz ratings as reference points. Its **local Elo estimate** uses different hardware and clocks; it is not an official CCRL, FIDE or Lichess rating. Intervals condition on fixed anchors and exclude those systematic differences. In-game latency remains about **33.3 ms at p95**, above the GPU target.

The original numeric targets remain in [out/targets.json](out/targets.json). [out/status.json](out/status.json) tracks the remaining work. A new **5.96M** policy is training on **341,010** positions with real histories and verified tactical labels. Its strength and the 70M-position training run remain unverified.

## Run it

Python 3.12. For the Apple GPU runtime:

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-mlx.txt
python -m scripts.fetch_pretrained --network bt4
lc0 leela2onnx --input=ckpts/bt4.pb.gz --output=ckpts/bt4.onnx --onnx-data-type=f16
python -m deploy.export_mlx --onnx ckpts/bt4.onnx --source-weights ckpts/bt4.pb.gz --bits 6 --out ckpts/bt4-q6.mlx
python -m play.uci --ckpt ckpts/bt4-q6.mlx
```

For CPU or CUDA training, install `requirements.txt`. The native ResNet uses policy cross-entropy, a separate all-legal-move Q head and a position-value head:

```sh
python -m train.train --config configs/main.yaml --data data/train.jsonl --val-data data/validation.jsonl --epochs 3 --q-value-weight .4
python -m play.runner --ckpt ckpts/latest.pt --deterministic
```

Compress a pretrained attention policy, retrain it, then quantize. Compression alone does not preserve strength:

```sh
python -m deploy.calibrate_mlx --model ckpts/bt3.mlx --data data/train.jsonl --out data/moments.safetensors
python -m deploy.export_mlx --onnx ckpts/bt3.onnx --source-weights ckpts/bt3.pb.gz --rank 16 --io-rank 32 --calibration data/moments.safetensors --out ckpts/compact-init.mlx
python -m train.mlx --student ckpts/compact-init.mlx --teacher ckpts/bt3.mlx --data data/train.jsonl --teacher-cache data/teacher.npy --out ckpts/compact.mlx
python -m deploy.quantize_mlx --model ckpts/compact.mlx --out ckpts/compact-q8.mlx
```

## Check it

```sh
pytest tests -q
python -m eval.agree --ckpt ckpts/bt3.mlx --labels data/stockfish_holdout.jsonl
python -m eval.puzzles --ckpt ckpts/bt3.mlx --puzzles data/puzzles_holdout.jsonl
python -m eval.latency --ckpt ckpts/bt3.mlx --pgn out/bt3_sf2650_clocked.pgn --out out/latency.json
python -m scripts.build_pool
python -m eval.pool --ckpt ckpts/bt3.mlx --games 200 --time-control 30+0.3 --out out/pool-run
```

The arena swaps colours for every opening, supplies real clocks to opponents, resets them between games, records time forfeits and saves every PGN. The rating runner refuses repeated opening pairs and unfinished games. It freezes model and opponent hashes and verifies one network pass per move.

## Learning targets and prior work

`Q = sigmoid(cp / 400)` is an outcome surrogate in **[0, 1]**. It gives **56.2% at +100 cp** and **73.1% at +400 cp**. The centred position value is `2Q - 1`. These are training targets, not calibrated winning probabilities.

[ChessBench](https://arxiv.org/abs/2402.04494) already studied action-value supervision. Our work here is the compact training pipeline, history encoder, GPU execution and reproducible evaluation. The larger baseline policies use [LC0 networks](https://lczero.org/play/networks/bestnets/); source URLs, hashes and runtime provenance are in [ckpts/pretrained.json](ckpts/pretrained.json).

Own code: MIT. Third-party weights, data and the optional LC0 runtime retain their upstream terms. Broadcast-game attribution is in [data/master_openings5000.json](data/master_openings5000.json).
