PYTHON ?= uv run python

.PHONY: sync lint format typecheck test scan train-baseline backtest paper lock

sync:
	uv sync

lock:
	uv lock

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

test:
	uv run pytest

scan:
	uv run pmtmax scan-markets

train-baseline:
	uv run pmtmax train-baseline

backtest:
	uv run pmtmax backtest

paper:
	uv run pmtmax paper-trader

