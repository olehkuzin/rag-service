# CLAUDE.md

Guidance for Claude Code when working in this repository. Architecture and design
rationale: [ARCHITECTURE.md](ARCHITECTURE.md). Usage and results: [README.md](README.md).

## Commands

```bash
make install                 # uv venv (.venv, Python 3.12) + editable install with dev extras
make corpus                  # fetch the dbt docs into data/corpus (needed before the first ingest)
make test                    # offline pytest suite (~2s): no network, no keys
.venv/bin/pytest             # same suite, but prints the "N passed" summary line
.venv/bin/pytest tests/test_retrieval.py::test_concurrent_reads_during_writes_never_fail  # one test
make lint && make typecheck  # ruff + mypy --strict (app, eval, scripts)
.venv/bin/ruff format --check .   # what CI runs; `make format` applies the fixes
make eval                    # offline eval (hash embedder + fake LLM) -> eval/reports/latest.{json,md}
make ingest                  # incremental ingest with the providers configured in .env
make ask                     # interactive question against data/index
```

CI runs ruff check, ruff format --check, mypy, and pytest with coverage. Keep all of
them green.

Real-provider commands call Mistral and **cost money**. Only run them when asked:
`make ingest` with `RAG_EMBEDDING_PROVIDER=mistral` re-embeds changed docs (a full
corpus is ~9.2k chunks, ~14 min), and so do `make eval-mistral` and `make ask`.

## Secrets and local data

- `.env` holds `MISTRAL_API_KEY`. Never print it, commit it, or copy it into the
  Docker build context (`.dockerignore` excludes it).
- `data/` (corpus, index, eval indexes, traces) and generated eval reports are
  gitignored. Don't delete `data/index` casually: rebuilding it with Mistral
  embeddings takes ~14 minutes and costs money.

## Invariants: keep these true

- **Tests stay offline.** Mock HTTP for provider tests (see `tests/test_providers.py`).
  The offline defaults are `hash` / `fake` / lexical.
- **Store and BM25 reads are lock-free and snapshot-based.** Never mutate the
  published snapshot's arrays or lists in place; build new ones and swap them with a
  single assignment. `test_concurrent_reads_during_writes_never_fail` guards this.
- **Ingestion is all-or-nothing.** Embed everything first, then one
  `store.replace_docs(...)`, then the manifest. Save the store before the manifest.
- **An incompatible index is quarantined, never fatal at load time** (otherwise the
  repairing `ingest` can't run).
- **JSON parsing is literal-first** (`_extract_json_object`). Answers can contain code
  fences, so never strip fences before trying the reply as JSON.
- **Strict mode never serves an uncited answer:** no resolvable citation means
  `found: false`.
- **Eval settings don't depend on `.env`** (`eval_settings`). The hash embedder is
  512-dim.

## When you change things

- Changing chunking, retrieval or generation changes the eval. Run `make eval` and
  update the README results table and the text that interprets it. Don't leave
  claims the table contradicts.
- Keep the README test count in sync with `.venv/bin/pytest` output. `make test`
  passes `-q` on top of `addopts = "-q"`, which hides the summary line.
- Mistral API quirks are handled in `MistralLLM`: `top_p=1` with temperature 0,
  chunked message content, and the reasoning-token budget. Keep them when touching
  the client.
