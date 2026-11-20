# Measurements

The JSON files contain aggregate results and model hashes. Account handles, game links and individual online outcomes remain private.

| Test | Result | Record |
|---|---:|---|
| 185M local engine pool | 2853; 95% interval 2819–2886; 400 games | [Local pool](local-pool.json) |
| 6M local engine pool | 2124; 95% interval 2040–2188; 400 games | [Local pool](local-pool.json) |
| Controlled BOT deployment | 2258 blitz across 172 account games; 10 audited deployment games; 492 matched neural moves | [Deployment](deployment.json) |
| 185M complete puzzle lines | 2190/3000 | [Puzzle bands](puzzles.json) |
| Compact ONNX CPU | 6.54 MB; 51.4 ms p95 over 200 history positions | [Latency](latency.json) |
| Weight-only INT8 comparison | 1000 games; FP32 advantage +1.7 Elo, interval −14.3 to +17.7 | [Quantization](quantization.json) |

Local pools use 50 colour-swapped opening pairs against four fixed Stash/Weiss versions, at 10+0.1 on Apple Silicon. Their CCRL reference ratings use different hardware and clock conditions; the intervals exclude that systematic uncertainty. These estimates cannot be averaged with or converted to Lichess/FIDE ratings.

The 185M is an LC0 pretrained policy; the compact CNN uses separate distillation/export tooling. The controlled deployment used another policy model, and the account history mixes models. Its 10 audited completed games and one aborted start establish deployment, not a standalone model rating. Warm latency excludes startup. [Released-weight serving check](serving-check.json).

The data pipeline collected 70M human positions and retained 41,765,559 after tier sampling. This is separate from Stockfish Q supervision and the pretrained 185M policy. [Data counts](data.json).

Local checks replayed 1,800 offline games and 10 completed deployment games, rematching 492 neural decisions. [Audit summary](audit.json). Original game evidence stays local. Public verification checks aggregates and file integrity:

```sh
python -m scripts.verify_results
python -m pip install -r requirements-plots.txt
python -m scripts.plot_results
```

[Release models](https://github.com/n33levo/sonofmagnus/releases/tag/v0.2.0) use [pinned checksums and source attribution](models.json). [SHA256SUMS](SHA256SUMS) covers the public snapshots. Code is MIT; pretrained weights retain upstream terms.
