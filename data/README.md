# Data

Git keeps manifests and small arena inputs. Large datasets are ignored; download them when needed:

```sh
python -m scripts.fetch_data
```

This restores the exact seven evaluation files used in `out/`, pinned to a published commit and checked by size and SHA-256. Existing files are verified and never overwritten. Training must exclude every position in these tests, including puzzle continuations.

Build training data from the public sources:

```sh
python -m scripts.build_human_corpus --month 2024-01 --positions 15000000 --out data/human_2024-01_15m.jsonl
python -m scripts.build_q_corpus --positions 2500000 --out data/full_q_2500k.jsonl
python -m scripts.build_teacher_corpus --positions 1000000 --out data/lichess_teacher1000k.jsonl --exclude-data data/puzzles_holdout.jsonl,data/puzzles_final3000.jsonl,data/puzzles_depth3000.jsonl,data/stockfish_holdout.jsonl,data/eval_holdout.jsonl,data/endgame_holdout.jsonl
python -m scripts.fetch_expert --positions 2500000 --out data/chessbench_expert2500k.jsonl --exclude-data data/puzzles_holdout.jsonl,data/puzzles_final3000.jsonl,data/puzzles_depth3000.jsonl,data/stockfish_holdout.jsonl,data/eval_holdout.jsonl,data/endgame_holdout.jsonl
python -m scripts.fetch_fullpolicy --out data/chessbench_fullpolicy_mid50k.jsonl --min-legal-moves 8
```

The full-policy tool checks every legal action and preserves mate distance. The checked 100k variant has 3,246,189 action labels; its broader exclusions and source hashes are in `out/fullpolicy-data-protocol.json`. These rows have FEN only. `python -m scripts.gm_corpus --help` covers GM-versus-GM extraction with actual game histories.

[Lichess games, puzzles and evaluations](https://database.lichess.org/) are CC0; broadcasts are CC-BY-SA-4.0. The master-game selection is attributed in `master_openings5000.json`. [ChessBench](https://github.com/google-deepmind/searchless_chess) expert labels retain CC-BY-4.0 attribution. Source hashes and collection counts live in `out/`.

Hugging Face is the intended home for curated dataset and model releases. Pin release commits and checksums here after publication; keep raw public archives at their source. No Hub release has been published yet.
