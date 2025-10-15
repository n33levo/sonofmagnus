# ChessHacks Plan — Searchless Neural Chess Bot (Policy+Value)

## 0) TL;DR
Build a **single-pass neural chess bot** (no search at play time) with an **AlphaZero-style policy+value network** and an **8×8×73** move head. Train in three steps: **(A)** human behavioral cloning, **(B)** **engine distillation** on hard positions, **(C)** **DAgger-lite** self-play relabeling. Export with **ONNX** and (optionally) INT8 quantization to target **≤10 MB** for the *Knight’s Edge* track. Tournament move selection = **forward pass → mask illegal moves → (temp-annealed) softmax/argmax**.

---

## 1) What this is
**ChessHacks** is a **36-hour** hackathon where teams (≤3) build an **AI model that plays chess**. Models **compete head-to-head** during and after the event. Tracks include **Queen’s Crown** (best overall), **Rook’s Rampage** (Elo king of the hill), **Bishop’s Gambit** (first to beat Waterloo’s strongest human), **Knight’s Edge** (**best mini-model ≤10 MB**), and **Pawn’s Rebellion** (beat the baseline). Organizers provide dev tools, a submission dashboard, live matches, and a few **pre-trained starters** you may extend.  
**Rule highlight:** “Cold hard neural networks: **no search algorithms**, no fine-tuned GPTs.” You **must** build your own model during the event; public datasets and **distilling from engines (for training)** are allowed.

**Success metric:** Win games at low latency; for mini-model, also satisfy **≤10 MB**.

---

## 2) What it’s for (our objectives)
- **Primary:** A robust **search-free** policy+value model that (a) never plays illegal moves, (b) respects time controls, (c) steadily improves via quick retrains.  
- **Track options:**  
  - **Queen’s Crown:** strongest overall snapshot.  
  - **Knight’s Edge:** ≤10 MB model with competitive strength.  
- **Operational goals:** fully automated train/eval/export pipeline; repeatable runs; crash-proof play loop.

---

## 3) Rules we must satisfy (baked into code/tests)
- **Inference:** exactly **one NN forward pass** per move; **no MCTS/alphabeta/engine calls** in runtime path.  
- **Legality:** mask **illegal moves** from the policy head before sampling/argmax.  
- **Data & training:** use **public data** (e.g., Lichess CC0), and (optionally) **engine-distill** **offline** to create supervised labels.  
- **Submission:** ship a single model file + runner that meets latency and (if targeting mini-model) **≤10 MB**.

---

## 4) System design (frozen interfaces to avoid churn)
### 4.1 Input encoding (board → tensor)
- 12 piece planes (own/opponent × {K,Q,R,B,N,P})  
- Side-to-move (1), castling rights (4), halfmove clock (1)  
- Optional **history** frames (e.g., last 4–6 plies) to reduce horizon blindness

### 4.2 Action encoding (policy head)
- **8×8×73** tensor (≈4,672 logits): each from-square (8×8) × 73 move-type planes (ray directions, knight offsets, promotions, etc.).  
- **Legality mask:** zero logits for illegal actions from `python-chess` enumerations → renormalize → sample/argmax.

### 4.3 Network
- **Backbone:** 10–14 **ResNet** blocks, 128 channels (≈6–8M params), GroupNorm + SiLU.  
- **Heads:**  
  - **Policy:** conv → 1×1 conv → **8×8×73** logits  
  - **Value:** conv → global-avg-pool → 2×FC → **tanh** (−1..1)

### 4.4 Loss & training
- **Loss:** `L = CE(policy) + λ * MSE(value)`; start λ = 0.25 (sweep 0.25–0.5).  
- **Speed:** PyTorch **AMP** (bf16/fp16) + GradScaler.  
- **Augment:** symmetry flips, color swap.

### 4.5 Inference (play loop)
- Encode board → forward pass → apply legality mask → temperature-annealed softmax (e.g., τ=0.6 opening, 0.2 endgame) or argmax → push move.  
- Guards: per-move **timeout**, **NaN** checks, safe **resign/draw** logic (50-move, threefold detection supported by rules API).

---

## 5) Repository layout (ready for Codex to scaffold)

chesshacks-bot/
  io/
    encode.py              # board→tensor planes
    policy_map.py          # 8x8x73 mapping + round-trip tests
    mask.py                # legality mask (python-chess)
  model/
    resnet.py              # trunk (ResNet backbone)
    heads.py               # policy/value heads
  train/
    dataset.py             # Lichess stream + sampling
    train.py               # AMP loop (CE+MSE), logging
    distill_labeler.py     # offline engine labels (top-k + value)
    selfplay.py            # searchless self-play FEN logger
  eval/
    puzzles.py             # CC0 puzzles evaluation
    agree.py               # move-agreement vs teacher
    arena.py               # round-robin matches vs snapshots
  play/
    runner.py              # single-pass move loop (CLI/bot API)
  deploy/
    export_onnx.py         # export trained model to ONNX
    quantize_int8.py       # ONNX Runtime dynamic/static quantization
    size_latency_check.py  # ensure model ≤10 MB & meets latency target
  tests/
    test_legality.py       # zero-illegal, masked softmax sums to 1
    test_endings.py        # stalemate, 50-move, threefold, promotions
    test_nosearch.py       # assert no chess.engine imports in play
  configs/
    main.yaml              # ~6–8M params (default config)
    mini.yaml              # ~2–4M params (≤10 MB after INT8)
  scripts/
    download_lichess.sh    # stream CC0 Lichess data
    curate_hards.py        # build “hard positions” dataset
    promote_snapshot.sh    # one-click export + quantize + gate check
  README.md
  plan.md                  # project plan and build stages
  Makefile                 # standardized commands for build & tests


---

## 6) What to **do now** (safe pre-event work)

> **Note:** Do not ship pre-trained final weights. Keep code, scripts, configs, and tests ready. Use tiny toy runs to verify plumbing.

### 6.1 Environment & deps
- Python 3.10+; `torch`, `onnx`, `onnxruntime`, `python-chess`, `tqdm`, `numpy`, `pandas`, `rich`  
- Make targets: `make test`, `make play_sanity`, `make export`, `make quantize`

### 6.2 Rules-correct play loop
- Implement `runner.py`:
  - build `chess.Board`  
  - `encode(board)` → `net(x)` → `mask_legal(logits, board)` → `pick_move(logits, τ)` → `board.push(move)`  
  - guards: **timeout**, **NaN**, **fallback** (e.g., highest-prob capture if distribution too flat)

### 6.3 IO (frozen)
- Implement 12+ planes encoder, **8×8×73** map (encode/decode UCI), and **unit tests**:
  - every legal move must map to exactly one (from,plane) index  
  - masked softmax sums to 1; illegals = 0 probability

### 6.4 Data pipeline
- `download_lichess.sh` to stream CC0 Lichess PGN (`.zst`) without full decompression.  
- `dataset.py` samplers: (a) uniform midgame positions, (b) Elo>1800 filter, (c) tactical buckets.  
- Fixed **puzzles** eval slice (store FEN+solution) for offline consistency.

### 6.5 Teacher labeler (distillation tooling only)
- `distill_labeler.py`:
  - input: CSV/JSONL of FEN  
  - output: JSONL with **top-k move distribution** + **centipawn/value** from **Stockfish** (depth/time caps)  
  - **Never** called in play. CLI supports `--fast` and `--deep` presets.

### 6.6 Self-play logger
- `selfplay.py`: no-search self-play using current policy; logs visited FENs to JSONL for **DAgger** relabeling.

### 6.7 Trainer (dry-run only now)
- `train.py` with AMP (`autocast` + `GradScaler`), cosine LR schedule, checkpoints, metrics: CE(policy), MSE(value), throughput.

### 6.8 Evaluation harness
- `agree.py`: move-agreement vs teacher labels on hold-out; Kendall’s τ optional.  
- `puzzles.py`: accuracy by rating buckets.  
- `arena.py`: N×N matches of snapshot(N) vs snapshot(N-1); metrics: win-rate, draw-rate, **latency**, crash-rate.

### 6.9 Export & size
- `export_onnx.py` (fp32/fp16) → `quantize_int8.py` (dynamic or static with calibration).  
- `size_latency_check.py`: file size and per-move latency report; fail if >10 MB for `mini.yaml`.

### 6.10 Compliance tests (prevent DQs)
- `test_legality.py`: zero illegal moves across random boards; masked softmax sums to 1.  
- `test_endings.py`: threefold/fifty-move/stalemate/promotion coverage.  
- `test_nosearch.py`: forbid `chess.engine` or external engines in `play/`.

### 6.11 Docs & one-clicks
- `README.md`: **exact** commands (download → sample → train → eval → export).  
- `promote_snapshot.sh`: tag checkpoint → export ONNX → quantize → run gates → emit **submission bundle**.

---

## 7) What to **do during the hackathon** (all final training)

### Phase A — Human Behavioral Cloning (stabilize)
- Sample 2–5M positions (midgames, Elo>1800). Train policy+value with AMP, symmetry augments. **Snapshot v0**.

### Phase B — Engine Distillation (strength jump)
- Curate **hard** buckets (tactics, high-branching middlegames, tricky endgames).  
- Label with Stockfish (top-k policies + centipawns/values) using `--fast` then `--deep` for a subset.  
- Train to **imitate** teacher policies/values. **Snapshot v1**.

### Phase C — DAgger-lite (fix your own mistakes)
- Short self-play bursts (no search) → relabel visited FENs with teacher → fine-tune. **Snapshot v2**.

### Track decision + compression
- **Queen’s Crown:** keep **main.yaml** FP16 if strongest.  
- **Knight’s Edge (≤10 MB):** switch to **mini.yaml** or prune channels, quantize INT8, verify gates.

### Tournament hardening
- Temperature schedule (τ higher opening, lower ending), safe resignation, time guards, crash handling.  
- Promote only snapshots that beat previous by **+5–10% win-rate at equal latency**.

### Live iteration
- Use organizer dashboard for A/B vs baselines/others; micro-finetune on observed weaknesses.

---

## 8) Acceptance gates (automated)
- **G1 – Legality:** illegal move rate = 0 on 10k random boards.  
- **G2 – Latency:** ≤ target ms/move on provided machine.  
- **G3 – Strength:** snapshot(N) ≥ snapshot(N−1) by +5–10% win-rate (100–300 games).  
- **G4 – Size (mini-model):** ONNX INT8 ≤ **10 MB**.  
- **G5 – Compliance:** no `chess.engine`/external engine in runtime path.

---

## 9) Risks & fallbacks
- **Compute tight?** Use **mini.yaml** (2–4M params) + stronger labels; skip Phase C if needed.  
- **Quantization hurts?** Quantize trunk only; keep heads FP16.  
- **Data bottleneck?** Use streaming loaders; downsample games; prefer curated hards.  
- **Time crunch?** Fine-tune **event starters** on hard buckets (explicitly allowed).

---

## 10) Hour-by-hour on event weekend (suggested)
- **H 0–3**  Wire datasets; smoke-test play; start Phase A.  
- **H 3–8**  Finish Phase A → **v0**; quick arena vs random/greedy baselines.  
- **H 8–18** Curate hards; label **fast**; train distill → **v1**; arena vs v0.  
- **H 18–26** DAgger burst(s); fine-tune → **v2**; arena vs v1.  
- **H 26–33** Export, INT8 (if mini-model), size/latency gates.  
- **H 33–36** Final gauntlet + submission.

---

## 11) Command snippets (Codex can wire)

# Data
bash scripts/download_lichess.sh
python -m train.dataset --build-splits --elo-min 1800 --midgame-only

# Teacher labels (offline only)
python -m train.distill_labeler --fens data/hard_fens.jsonl --topk 5 --depth 14 --out data/labels_fast.jsonl
python -m train.distill_labeler --fens data/hard_fens_top.jsonl --topk 12 --depth 18 --out data/labels_deep.jsonl

# Train (AMP)
python -m train.train --config configs/main.yaml --labels data/labels_fast.jsonl --epochs 2 --amp bf16
python -m train.train --config configs/main.yaml --labels data/labels_deep.jsonl --epochs 1 --amp bf16

# Self-play + DAgger
python -m train.selfplay --games 2000 --out data/selfplay.jsonl
python -m train.distill_labeler --fens data/selfplay.jsonl --topk 8 --depth 14 --out data/selfplay_labels.jsonl
python -m train.train --config configs/main.yaml --labels data/selfplay_labels.jsonl --epochs 1 --amp bf16

# Eval
python -m eval.agree --labels data/holdout_labels.jsonl --ckpt ckpts/v1.pt
python -m eval.puzzles --puzzles data/puzzles.jsonl --ckpt ckpts/v1.pt
python -m eval.arena --a ckpts/v1.pt --b ckpts/v0.pt --games 300 --tc "0.2"

# Export & quantize
python -m deploy.export_onnx --ckpt ckpts/v2.pt --out out/model.onnx --fp16
python -m deploy.quantize_int8 --in out/model.onnx --out out/model.int8.onnx --calib data/calib.jsonl
python -m deploy.size_latency_check --model out/model.int8.onnx

# Submission bundle
bash scripts/promote_snapshot.sh ckpts/v2.pt

---

## 13) Stretch (only if time permits)

Snapshot ensembling (2–3 checkpoints): average policy logits at inference (still one forward per snapshot; fuse offline if needed).

Skill-aware head (human-like option): add small “skill embedding” to smooth openings vs weaker opponents.

---

## 14) Definition of done

Passes G1–G5 gates; wins > baseline; meets target latency; for mini-model, ≤10 MB; reproducible with one command.


## Sources

1) **ChessHacks (official site & FAQ).**  
   https://chesshacks.dev/ :contentReference[oaicite:0]{index=0}

2) **Lichess Open Database (CC0) & streaming examples.**  
   - Official CC0 database exports: https://database.lichess.org/ :contentReference[oaicite:1]{index=1}  
   - Example streaming/parse pipeline using `python-chess` (HF dataset): https://huggingface.co/datasets/Icannos/lichess_games :contentReference[oaicite:2]{index=2}

3) **`python-chess` documentation (legality, threefold, fifty-move).**  
   https://python-chess.readthedocs.io/en/latest/core.html :contentReference[oaicite:3]{index=3}

4) **AlphaZero move representation & 8×8×73 policy head.**  
   - AlphaZero chess paper: Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (arXiv:1712.01815). https://arxiv.org/abs/1712.01815 :contentReference[oaicite:4]{index=4}  
   - Clear description of 8×8×73 action planes (quoting AlphaZero): PettingZoo Chess docs. https://pettingzoo.farama.org/environments/classic/chess/ :contentReference[oaicite:5]{index=5}

5) **“Grandmaster-level chess without search” / ChessBench (engine-labeled supervised training).**  
   - Ruoss et al., *Amortized Planning with Large-Scale Transformers* (“ChessBench”), arXiv:2402.04494. https://arxiv.org/abs/2402.04494 :contentReference[oaicite:6]{index=6}  
   - Google DeepMind “searchless_chess” / ChessBench repository. https://github.com/google-deepmind/searchless_chess :contentReference[oaicite:7]{index=7}

6) **Deployment references (quantization & mixed precision).**  
   - ONNX Runtime INT8 quantization docs: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html :contentReference[oaicite:8]{index=8}  
   - PyTorch Automatic Mixed Precision (autocast + GradScaler): https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html :contentReference[oaicite:9]{index=9}
