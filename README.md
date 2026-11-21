# Production RAG Service with Evaluation

A retrieval-augmented generation service that answers questions over the **dbt
documentation** and returns **cited answers with abstention**. The point is not
"chat with docs" — it is measured quality, production hygiene, and a data layer
built for incremental change.

## Corpus

The [`dbt-labs/docs.getdbt.com`](https://github.com/dbt-labs/docs.getdbt.com)
documentation: **1,208 markdown/MDX files, ~1.1M words, 12 MB** (fetched with a
sparse checkout of `website/docs`). Each file has YAML frontmatter and heavy MDX
(JSX components, imports, admonitions), so parsing and cleaning are real work
rather than a toy `read_text()`.

Refresh the corpus:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/dbt-labs/docs.getdbt.com.git /tmp/dbt-docs
cd /tmp/dbt-docs && git sparse-checkout set website/docs
cp -r website/docs/* /path/to/rag-service/data/corpus/
```

## Architecture

```
Ingestion                                        Serving
─────────                                        ───────
docs ──► parse (frontmatter, MDX clean)          POST /ask
     ──► structure-aware chunk (by heading)          │
     ──► quality checks (empty/short/dup/meta)       ├─► vector search (cosine)
     ──► embed (provider interface)                  ├─► BM25 keyword search
     ──► incremental upsert by content hash          ├─► reciprocal rank fusion
                                                     ├─► reranker (cross-encoder slot)
                                                     └─► LLM with cite-or-abstain prompt
                                                          ▼
                                              answer + citations + found + usage
```

Every external dependency (embedder, LLM, reranker, vector store) sits behind a
`Protocol`, so tests, eval variants and model comparisons swap implementations
without touching call sites. Design decisions (incremental all-or-nothing
ingestion, lock-free consistent reads, cite-or-abstain parsing) are explained in
[ARCHITECTURE.md](ARCHITECTURE.md).

| Layer | Implementation |
|---|---|
| Parsing | `python-frontmatter` + custom MDX/JSX cleaner |
| Chunking | heading-path-aware sections, paragraph-boundary windows with overlap; `fixed` baseline for comparison |
| Quality | drops empty / too-short / boilerplate / duplicate chunks; validates required metadata; reports *reasons* |
| Incremental | per-document content hash in a persisted manifest; stale chunks deleted before re-index; deleted docs pruned |
| Embeddings | `hash` (offline, deterministic default) and `mistral` (`mistral-embed`, 1024-dim) |
| Vector store | `local` (numpy + JSON, persisted) and `qdrant` adapters |
| Keyword | in-memory Okapi BM25 |
| Fusion | reciprocal rank fusion (k=60) |
| Rerank | `LexicalReranker` (offline IDF-overlap) — no hosted reranker wired |
| LLM | `fake` (offline, extractive, never fabricates) and `mistral` (chat completions, JSON mode) |
| Observability | structured trace per request: chunks, scores, stage counts, latency, tokens, cost → `data/index/traces.jsonl` + `GET /traces` |

## Quick start

Runs on one machine — no cloud, no managed services.

```bash
make install            # uv venv + editable install
make corpus             # fetch the dbt docs into data/corpus (gitignored)
cp .env.example .env    # then set MISTRAL_API_KEY
rm -rf data/index && make ingest   # ~9.3k chunks, a few seconds
make ask
```

Serve it:

```bash
.venv/bin/uvicorn app.main:app --reload     # http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What are the five built-in materializations in dbt?", "top_k": 5}'
```

Or with Docker Compose (no Qdrant needed — the local store is the default):

```bash
docker compose up --build
```

The compose index volume starts empty, so `/ask` returns 503 until the first
ingestion. Run `make corpus` beforehand (compose mounts `data/corpus`), set
`MISTRAL_API_KEY` in `.env`, then index once:

```bash
curl -X POST localhost:8000/ingest
```

```bash
make install          # uv venv + editable install
make corpus           # fetch the dbt docs into data/corpus (gitignored)
make ingest           # parse → chunk → QC → embed → upsert
make ask              # interactive question
make test             # 113 tests
make eval             # run all evaluation variants
.venv/bin/uvicorn app.main:app --reload
```

Defaults run **fully offline**: `hash` embeddings + `fake` LLM, no API keys, so
tests and the eval harness work anywhere.

### Using Mistral (local, real models)

Copy `.env.example` to `.env` and set one key:

```bash
MISTRAL_API_KEY=...
RAG_EMBEDDING_PROVIDER=mistral   RAG_EMBEDDING_MODEL=mistral-embed   RAG_EMBEDDING_DIM=1024
RAG_LLM_PROVIDER=mistral         RAG_LLM_MODEL=mistral-large-latest
```

`mistral-large-latest` currently resolves to `mistral-large-4`, a reasoning
model: it emits thinking before the answer, and those tokens count against
`RAG_LLM_MAX_TOKENS` (default 4096). With too small a budget it can run out
before answering; that is reported as a provider error, not as an abstention.
Requests pin `temperature=0, top_p=1` (Mistral rejects greedy decoding with any
other `top_p`).

Then rebuild the index — **the dimension changes from 512 to 1024**, so the old
index must be discarded:

```bash
rm -rf data/index && make ingest && make ask
make eval-mistral      # real-provider evaluation
```

Switching provider/model without re-ingesting is caught rather than silently
producing garbage: the store records `provider:model` and the vector dimension
in `data/index/index_meta.json`. On a mismatch it **quarantines** the old index
(moves it to `data/index.stale-<timestamp>`) and starts empty, printing a
warning, so `make ingest` can rebuild. It deliberately does not raise: raising at
load time would make the recommended remedy impossible, because ingest loads the
store too.

Provider keys are read from `os.environ`, not via `Settings`, so `.env` is
exported into the environment at startup; a key that exists only in `.env` is
otherwise invisible. Real environment variables take precedence over `.env`.

Provider failures are reported as a single actionable message rather than a
traceback, e.g.:

```
401 from https://api.mistral.ai/v1/embeddings
  detail: Your API key expired on 2026-11-21.
  MISTRAL_API_KEY is missing, invalid, or expired.
  Check the key at https://console.mistral.ai/ ...
```

Mistral calls retry on 429/5xx and transport errors with exponential backoff, and
the chat client uses `response_format=json_object` so strict-mode parsing is
reliable rather than best-effort.

## API

```http
POST /ask
{ "question": "string", "top_k": 5,
  "hybrid": true, "rerank": true, "prompt_variant": "strict" }   # last three optional
```
```json
{ "answer": "string",
  "found": true,
  "confidence": 0.0,
  "citations": [{ "doc": "...", "section": "...", "chunk_id": "...", "score": 0.0 }],
  "retrieved": [{ "chunk_id": "...", "doc": "...", "section": "...", "score": 0.0, "rank": 1 }],
  "latency_ms": 0, "retrieval_latency_ms": 0,
  "usage": { "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0 },
  "trace_id": "...", "prompt_variant": "strict" }
```

`POST /ingest` (incremental by default, `{"force": true}` re-indexes everything),
`GET /health`, `GET /traces`.

Incremental ingestion converges: the first run over the full corpus indexes 9,344
chunks from 1,198 documents in ~2.9s, and every subsequent run processes 0
documents. A run is all-or-nothing: changed documents are embedded first and
applied to the store in one swap, so a provider failure mid-run (an expired key,
exhausted retries) leaves the index exactly as it was, and queries served during
ingestion always see a consistent index. A persisted manifest records each document version independently of
its chunks, so documents that yield zero valid chunks are not reprocessed on
every run, and documents deleted from the corpus are pruned from the index.

Abstention is a first-class outcome: strict mode requires every claim to cite a
retrieved chunk, and an answer with no resolvable citation is downgraded to
`found: false` rather than served.

## Evaluation

`eval/dataset.jsonl` holds 19 labelled questions: fact lookup, multi-chunk
synthesis, paraphrases, and 5 unanswerable questions (26%) that must trigger
abstention. Retrieval is scored against gold document paths.

Metrics: recall@k, hit@k, MRR, nDCG@k (retrieval); token F1 and keyword coverage
vs reference answers (generation); citation accuracy (cited chunks that come from
gold docs); abstention rate on unanswerable questions; p50/p95 latency and cost
per query.

Reproduce with `make eval` (writes `eval/reports/latest.{json,md}`).

### Results (k=5, 1,208-doc corpus)

| variant      |   recall@k |      hit@k |        MRR |     nDCG@k |         F1 |     kw cov |   cite acc | answer rate |    abstain |     p95 ms |     cost/q |
|--------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| baseline     |      0.286 |      0.357 |      0.226 |      0.309 |      0.076 |      0.208 |      0.143 |      1.000 |      0.000 |        0.3 |        n/a |
| +chunking    |      0.357 |      0.429 |      0.234 |      0.338 |      0.134 |      0.196 |      0.143 |      1.000 |      0.000 |        0.6 |        n/a |
| +hybrid      |      0.536 |      0.571 |      0.404 |      0.513 |      0.128 |      0.268 |      0.357 |      1.000 |      0.000 |        8.8 |        n/a |
| +rerank      |      0.536 |      0.571 |      0.348 |      0.498 |      0.107 |      0.325 |      0.214 |      1.000 |      0.000 |       10.3 |        n/a |
| strict       |      0.536 |      0.571 |      0.348 |      0.498 |      0.107 |      0.325 |      0.214 |      1.000 |      0.000 |        9.8 |        n/a |

Cost shows `n/a` under the offline stub: its token counts are synthetic, so
pricing them produced a fabricated figure. Real costs appear with `--llm mistral`.

Each row adds one stage to the row above. Headline: **recall@5 0.286 → 0.536
(+87%)** and **hit@5 0.357 → 0.571** from structure-aware chunking + hybrid
retrieval. Reranking adds no recall on top of that (see below).

### Honest reading of these numbers

These are **retrieval-focused** results from the offline defaults (`hash`
embedder + `fake` LLM), and the generation columns are weak by construction.
Cost is `n/a` under the offline stub by design: its token counts are synthetic,
so reporting a dollar figure would be invented.

- The default `fake` LLM is an **extractive stub** — it echoes the top chunk, so
  F1/keyword coverage measure *retrieval* quality, not model quality. Real F1
  requires `RAG_LLM_PROVIDER=mistral` and an API key (`make eval-mistral`).
- The default `hash` embedder is a token-hash bag-of-words, **not a semantic
  model**. Recall numbers are therefore a conservative lower bound; a hosted
  embedder should lift every retrieval row.
- Abstention on unanswerable questions is 0.0: the stub answers whenever any
  context is retrieved, and retrieval always returns something. (An earlier 0.2
  here was a parser bug, not abstention: a JSON answer containing a ```` ```sql ````
  block was mis-parsed and counted as abstaining.) The harness verifies the
  abstention plumbing; measuring abstention itself needs a real model.
- The dataset is 19 questions — enough to exercise every metric and catch
  regressions, too small to claim statistical significance (one question moves
  recall@5 by ~7 points). Scaling to 100–150 is the first next step, and the 5
  unanswerable questions are deliberately easy (absurd premises); *near-miss*
  abstention is the case a real model actually fails.
- Reranking leaves recall@5 unchanged (0.536) and lowers MRR (0.404 → 0.348),
  nDCG@5 (0.513 → 0.498) and citation accuracy (0.357 → 0.214) versus +hybrid.
  The offline lexical reranker is a net negative here, yet it is on by default
  (`RAG_ENABLE_RERANK=true`); a real cross-encoder has to show a gain before that
  default is justified.

The pipeline, metrics, and experiment structure are real; the *model* is a
placeholder so everything runs without keys.

## Tests

113 tests, no network, no API keys, no Docker (`pytest`, ~2s):

- **ingestion** — MDX cleaning, frontmatter, heading-aware chunking, chunk-id
  stability, oversized splitting, QC rules (short/duplicate/missing metadata),
  incremental skip, changed-doc replacement without leftovers, zero-chunk
  convergence, pruning of deleted documents, all-or-nothing runs (a provider
  failure mid-ingest leaves store and manifest untouched), and fence awareness
  (code blocks never split; `#` inside a fence is not a heading; `~~~` fences
  supported)
- **retrieval** — BM25 ranking, RRF agreement, store upsert/persist/delete,
  reranker ordering, stage toggles, doc filtering, index-integrity guards
  (dimension mismatch, provider switch, torn write), atomic `replace_docs`, and
  a reader/writer stress test that searches while the index is rewritten
- **generation** — context numbering, strict/basic prompts, fenced-JSON parsing,
  citation resolution, unknown-citation filtering, confidence clamping,
  abstention on bad JSON and on citation-less answers, and JSON answers that
  themselves contain code fences
- **providers** — Mistral request shaping, response index ordering, L2
  normalisation, dimension validation, batching, key handling, retry-on-429 vs
  fail-fast-on-400, usage plumbing (all via mocked HTTP)
- **api** — health, 503 before ingest, ingest→ask, validation (422), per-request
  feature overrides without leaking into the singleton, trace recording,
  incremental ingest, and a concurrency test proving a slow provider does not
  serialize requests
- **eval** — recall@k / hit@k / MRR / nDCG / token-F1 / keyword coverage /
  aggregation / abstention wiring / pricing, eval settings pinned independently
  of `.env`, plus dataset integrity checks
  (unique ids, gold docs exist on disk, unanswerable ratio in band)

CI (`.github/workflows/ci.yml`) runs ruff, format check, `mypy --strict`, the test
suite with coverage, and builds the Docker image on every push.

## Repository layout

```
app/
  api/            routes + Pydantic schemas
  retrieval/      store (local/qdrant), bm25+rrf, pipeline
  generation/     prompts, parsing, citations, abstention
  ingestion/      parser, chunker, quality, incremental indexer
  observability/  tracing, cost estimation
  providers.py    embedder / llm / reranker interfaces + implementations
  service.py      composition root
eval/             dataset.jsonl, run_eval.py, reports/
tests/            113 tests
scripts/          ingest_cli.py, fetch_corpus.sh
Dockerfile · docker-compose.yml · .github/workflows/ci.yml · Makefile
ARCHITECTURE.md   design and rationale · CLAUDE.md   guidance for coding agents
```

## Limitations and next steps

1. **Real models** — Mistral is wired; run `make eval-mistral` to replace the
   placeholder generation numbers with genuine ones.
2. **Bigger eval set** — 100–150 questions, with inter-annotator checks on the
   gold documents; add an LLM-as-judge faithfulness metric.
3. **Real reranker** — `bge-reranker` or a Mistral-compatible rerank API to
   confirm the MRR regression seen with the lexical stand-in disappears.
4. **Tracing** — Langfuse/OTel adapter behind the existing `Tracer` interface;
   the record shape is already defined.
5. **Stretch** — query decomposition for multi-hop questions, semantic caching,
   scheduled re-ingestion, and an MCP `search_docs` server over this corpus.

Known gaps: no `pre-commit` config (CI runs the same checks); no faithfulness /
LLM-as-judge metric yet — `citation_accuracy` measures whether a cited chunk
came from a gold document, **not** whether the answer is supported by the text
it cites; `langfuse_enabled`, `api_host` and `api_port` are dead config (the
tracer is local and uvicorn is started manually).

Scope: **local / single-machine deployment only.** There is no cloud deployment,
autoscaling or managed secrets story here, by choice. Docker Compose is the
supported way to run it; the local numpy vector store is the default.
