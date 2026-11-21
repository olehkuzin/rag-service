.PHONY: install corpus lint format typecheck test eval up down ingest ask clean

install:
	uv venv --python 3.12 .venv
	uv pip install --python .venv -e ".[dev]"

corpus:
	./scripts/fetch_corpus.sh

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .

typecheck:
	.venv/bin/mypy app eval scripts

test:
	.venv/bin/pytest -q

eval:
	.venv/bin/python -m eval.run_eval

ingest:
	.venv/bin/python -m scripts.ingest_cli ingest

ask:
	@read -p "question: " q; .venv/bin/python -m scripts.ingest_cli ask "$$q"

up:
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf data/index data/eval_index .pytest_cache .ruff_cache .mypy_cache
