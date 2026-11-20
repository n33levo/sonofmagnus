# Data

Large corpora stay outside Git. Download the pinned evaluation snapshots with:

```sh
python -m scripts.fetch_data
```

Training tools:

- `scripts.build_human_corpus`: rated human games; the collected corpus contains 70M positions.
- `scripts.build_q_corpus` and `scripts.build_teacher_corpus`: evaluated moves and Stockfish targets.
- `scripts.fetch_expert` and `scripts.fetch_fullpolicy`: public expert/action-value data.
- `scripts.gm_corpus`: titled-player games with full move histories.

Run each module with `--help`. Exclude evaluation boards and puzzle continuations from training. [Counts and scope](../results/data.json). The older 2.5M-action corpus mostly contains 2–5 legal moves and should not be treated as broad middlegame supervision.

[Lichess games, puzzles and evaluations](https://database.lichess.org/) are CC0; broadcasts retain CC-BY-SA-4.0. [ChessBench data](https://github.com/google-deepmind/searchless_chess) retain upstream attribution. Sources and checksums are pinned in the manifests. Trained weights are available through the [GitHub release](https://github.com/n33levo/sonofmagnus/releases/tag/v0.2.0).
