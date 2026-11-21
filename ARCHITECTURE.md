# Architecture

This document explains how the RAG service is put together and why. The
[README](README.md) covers what it does, results, and how to run it.

## Components and data flow

```mermaid
flowchart LR
    subgraph Ingestion - scripts/ingest_cli.py, POST /ingest
        D[data/corpus<br/>md / mdx] --> P[parse<br/>frontmatter, MDX clean]
        P --> C[chunk<br/>heading-aware, fence-safe]
        C --> Q[quality checks]
        Q --> E[embed<br/>Embedder]
        E --> S[stage all changes]
        S -->|one replace_docs| V[(Vector store<br/>local numpy / Qdrant)]
        S --> M[(manifest.json<br/>doc versions)]
    end

    subgraph Serving - FastAPI
        U[POST /ask] --> R1[vector search]
        U --> R2[BM25]
        R1 & R2 --> F[RRF k=60]
        F --> RR[reranker]
        RR --> G[LLM: cite-or-abstain prompt]
        G --> PR[parse JSON, resolve citations]
        PR --> U
    end

    V --> R1
    V -->|rebuilt after ingest| R2
    U -.-> T[(traces.jsonl + GET /traces)]
```

| Module | Responsibility |
|---|---|
| `app/main.py` | FastAPI app factory; warms the service at startup |
| `app/api/routes.py`, `schemas.py` | `/ask`, `/ingest`, `/health`, `/traces`; Pydantic request/response models; blocking work runs in a threadpool |
| `app/service.py` | Composition root and process singleton: builds providers, store, retriever, generator, ingestor, tracer; serialises writers with a lock |
| `app/config.py` | `Settings` (env prefix `RAG_`, plus `.env`); `load_env_file` exports `.env` so provider keys read from `os.environ` are visible |
| `app/providers.py` | `Embedder`, `LLMClient` and `Reranker` protocols, with offline (`hash`, `fake`, lexical) and Mistral implementations; HTTP retries and actionable errors |
| `app/ingestion/` | `parser` (frontmatter, MDX/JSX cleanup), `chunker` (structure and fixed strategies), `quality` (drop rules with reasons), `manifest` (per-doc versions), `indexer` (the incremental, all-or-nothing run) |
| `app/retrieval/` | `store` (local numpy store and Qdrant adapter), `hybrid` (BM25, RRF), `pipeline` (stage toggles per request) |
| `app/generation/generator.py` | Prompts (`basic`, `strict`), context numbering `c1..cN`, JSON extraction, citation resolution, abstention rules |
| `app/observability/tracing.py` | One `TraceRecord` per request (in-memory ring buffer + JSONL); cost estimate |
| `eval/run_eval.py` | Variant runner (baseline → +chunking → +hybrid → +rerank → strict) and metrics |

## Key design decisions

### Every external dependency behind a Protocol

Embedder, LLM, reranker and vector store are protocols. The offline defaults are a
deterministic hash embedder, an extractive fake LLM and a lexical reranker, so tests
and the eval harness run without keys or network. The Mistral implementations are
swapped in by config alone.

### Incremental, all-or-nothing ingestion

- **Change detection** is per document: a content hash of the *cleaned* text,
  stored in a manifest kept separately from the vectors. Documents whose chunks are
  all rejected are still recorded, so they converge instead of being reprocessed
  forever, and documents deleted from the corpus are pruned.
- **All-or-nothing commit.** A run parses, chunks, checks and embeds every changed
  document *first*, then applies the result to the store in one `replace_docs` call.
  A provider failure mid-run (expired key, exhausted retries) leaves the store and
  manifest exactly as they were. The trade-off is that embeddings computed before the
  failure are discarded.
- **Persistence order:** vectors are written before the manifest, each file via
  temp file + atomic replace. A crash in between causes a harmless re-process rather
  than documents recorded as indexed that the store never received.
- **Index compatibility.** `index_meta.json` records `provider:model` and the
  dimension. On a mismatch, or on a torn write, the store **quarantines** the old
  index (`data/index.stale-<ts>`) and starts empty instead of raising, because raising
  at load time would also block the `ingest` that repairs it.

### Reads stay consistent while ingestion writes

`/ask` runs concurrently with `/ingest`. `LocalVectorStore` keeps vectors and
payloads in one immutable snapshot. Writers build a new snapshot (copy-on-write) and
publish it with a single assignment, and every read uses the snapshot it grabbed
first. `BM25Index` does the same with its term statistics. Readers therefore never
take a lock and never see a half-written index; writers are serialised by the
service lock. A thread stress test covers this, and it fails 3/3 against the
previous in-place mutation code.

### Retrieval

Vector search (cosine on L2-normalised vectors) and BM25 each return 20 candidates,
fused with reciprocal rank fusion (k=60). The top 20 are then reranked and the top
`k` returned. Each stage can be toggled per request (`hybrid`, `rerank`), which is how
the eval measures every stage. The lexical reranker is a stand-in for a
cross-encoder. In the offline eval it lowers MRR, nDCG and citation accuracy, so its
default-on setting is unjustified until a real reranker shows a gain.

### Generation: cite or abstain

Retrieved chunks are numbered `c1..cN` in the prompt. The `strict` prompt requires
JSON `{found, answer, citations, confidence}`, and the model must cite chunk ids or
abstain. Parsing is deliberately literal-first: the reply is tried as JSON, then as
one fenced block wrapping the whole reply, and only then is the first `{` to the last
`}` extracted. This order matters because answers often contain ```` ```sql ````
blocks. An empty answer, or a strict-mode answer with no resolvable citation, is
downgraded to `found: false`. Unparseable strict output is an abstention, never a
crash.

Mistral specifics handled in `MistralLLM`:

- `response_format=json_object`.
- `temperature=0` with `top_p=1`, since the API rejects greedy decoding otherwise.
- Message content can be a list of typed chunks, and only `text` chunks are used.
- `mistral-large-latest` is a reasoning model whose thinking counts against
  `max_tokens`, so an empty truncated reply raises a clear error.

### Observability and cost

Every `/ask` records the question, retrieved chunks with per-stage scores, stage
counts, citations, latency split (retrieval vs generation), tokens and estimated cost
to `data/index/traces.jsonl`, plus an in-memory ring served by `GET /traces`. Cost is
`None` for the offline stub, because its token counts are synthetic.

### Evaluation

`eval/run_eval.py` runs each variant over `eval/dataset.jsonl` and reports:

- retrieval: recall@k, hit@k, MRR and nDCG@k against gold documents;
- generation: token F1 and keyword coverage;
- citations: accuracy;
- abstention on unanswerable questions;
- latency (p50/p95) and cost.

Eval indexes are cached under `data/eval_index/`, keyed by chunk strategy, provider,
dimension and a hash of the chunk settings, and synced incrementally. Settings are
pinned independently of `.env` (hash embedder = 512 dims), and the settings used are
recorded in each report.

## Deployment shape

The service is local / single-machine by choice. Docker Compose runs one container
with the local store on a named volume. The image creates `/app/data/*` owned by the
non-root user, so the volume is writable. The corpus is bind-mounted read-only, and
the index starts empty until `POST /ingest`. Qdrant is an optional adapter
(`RAG_VECTOR_STORE=qdrant`, requires `qdrant-client`); there, `replace_docs` is
delete-then-upsert and not atomic.

## Extending

- **New provider:** implement the protocol in `app/providers.py` and add a case to
  `build_embedder`, `build_llm` or `build_reranker`. Changing the embedder changes
  `embedding_id`, and the old index is quarantined automatically.
- **New retrieval stage:** add it to `Retriever.retrieve` behind a `RetrievalConfig`
  flag, and add an eval `Variant` that toggles it.
- **New store:** implement `VectorStore`, including `replace_docs`. Readers must
  never observe a partially applied write.
