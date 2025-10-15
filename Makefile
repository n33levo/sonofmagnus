.PHONY: test play_sanity export quantize install clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

test_legality:
	python -m pytest tests/test_legality.py -v

test_endings:
	python -m pytest tests/test_endings.py -v

test_nosearch:
	python -m pytest tests/test_nosearch.py -v

play_sanity:
	python -m play.runner --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --moves 10

export:
	python -m deploy.export_onnx --ckpt ckpts/latest.pt --out out/model.onnx --fp16

quantize:
	python -m deploy.quantize_int8 --in out/model.onnx --out out/model.int8.onnx

size_check:
	python -m deploy.size_latency_check --model out/model.int8.onnx

clean:
	rm -rf __pycache__ **/__pycache__ *.pyc **/*.pyc .pytest_cache
	rm -rf out/*.onnx ckpts/*.pt

download_data:
	bash scripts/download_lichess.sh

train_bc:
	python -m train.train --config configs/main.yaml --phase bc --epochs 2 --amp bf16

train_distill:
	python -m train.train --config configs/main.yaml --phase distill --epochs 1 --amp bf16

eval_puzzles:
	python -m eval.puzzles --puzzles data/puzzles.jsonl --ckpt ckpts/latest.pt

eval_arena:
	python -m eval.arena --a ckpts/v1.pt --b ckpts/v0.pt --games 100

promote:
	bash scripts/promote_snapshot.sh ckpts/latest.pt
