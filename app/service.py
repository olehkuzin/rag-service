"""Service container: builds and holds the wired pipeline for the API and eval."""

from __future__ import annotations

import threading
import time
import uuid

from app.config import Settings, get_settings
from app.generation.generator import GenerationResult, Generator
from app.ingestion.indexer import Ingestor, IngestStats
from app.observability.tracing import Tracer, TraceRecord, estimate_cost
from app.providers import (
    Embedder,
    LexicalReranker,
    LLMClient,
    Reranker,
    build_embedder,
    build_llm,
)
from app.retrieval.hybrid import BM25Index
from app.retrieval.pipeline import RetrievalConfig, Retriever
from app.retrieval.store import LocalVectorStore, QdrantVectorStore, VectorStore


class Service:
    """Composition root. One instance per process, thread-safe for reads."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder: Embedder = build_embedder(
            self.settings.embedding_provider,
            self.settings.embedding_model,
            self.settings.embedding_dim,
        )
        self.llm: LLMClient = build_llm(self.settings.llm_provider, self.settings.llm_model)
        self.reranker: Reranker = LexicalReranker()
        self.store: VectorStore = self._build_store()
        self.bm25 = BM25Index()
        self.retriever = Retriever(
            store=self.store,
            embedder=self.embedder,
            settings=self.settings,
            bm25=self.bm25,
            reranker=self.reranker,
        )
        self.generator = Generator(self.llm, self.settings)
        self.ingestor = Ingestor(self.store, self.embedder, self.settings)
        self.tracer = Tracer(log_path=self.settings.index_dir / "traces.jsonl")
        self._write_lock = threading.Lock()
        self._load()

    def _build_store(self) -> VectorStore:
        if self.settings.vector_store == "qdrant":
            return QdrantVectorStore(
                url=self.settings.qdrant_url,
                collection=self.settings.qdrant_collection,
                dim=self.settings.embedding_dim,
            )
        store = LocalVectorStore(self.settings.index_dir, self.settings.embedding_dim)
        store.load()
        return store

    def _load(self) -> None:
        if self.store.count():
            self.retriever.rebuild_bm25()

    # -- operations --------------------------------------------------------
    def ingest(self, force: bool = False, limit: int | None = None) -> IngestStats:
        with self._write_lock:
            stats = self.ingestor.run(force=force, limit=limit)
            self.retriever.rebuild_bm25()
            return stats

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        config: RetrievalConfig | None = None,
        trace: bool = True,
        prompt_variant: str | None = None,
    ) -> tuple[GenerationResult, TraceRecord]:
        trace_id = uuid.uuid4().hex[:16]
        started = time.perf_counter()
        config = config or RetrievalConfig.from_settings(self.settings, top_k)

        retrieval = self.retriever.retrieve(question, config)
        gen_started = time.perf_counter()
        result = self.generator.generate(question, retrieval.chunks, variant=prompt_variant)
        generation_ms = (time.perf_counter() - gen_started) * 1000
        total_ms = (time.perf_counter() - started) * 1000

        record = TraceRecord(
            trace_id=trace_id,
            question=question,
            answer=result.answer,
            found=result.found,
            prompt_variant=result.prompt_variant,
            retrieved=[
                {
                    "chunk_id": c.chunk.chunk_id,
                    "doc": c.chunk.title,
                    "section": c.chunk.section,
                    "score": round(c.score, 4),
                    "vector_score": None if c.vector_score is None else round(c.vector_score, 4),
                    "bm25_score": None if c.bm25_score is None else round(c.bm25_score, 4),
                    "rerank_score": None if c.rerank_score is None else round(c.rerank_score, 4),
                }
                for c in retrieval.chunks
            ],
            stage_counts=retrieval.stages,
            citations=[c.chunk_id for c in result.citations],
            latency_ms=total_ms,
            retrieval_latency_ms=retrieval.latency_ms,
            generation_latency_ms=generation_ms,
            input_tokens=result.usage_input_tokens,
            output_tokens=result.usage_output_tokens,
            cost_usd=estimate_cost(
                self.settings.llm_model,
                result.usage_input_tokens,
                result.usage_output_tokens,
            ),
            model=self.settings.llm_model,
            error=result.parse_error,
        )
        if trace:
            self.tracer.record(record)
        return result, record

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "chunks": self.store.count(),
            "embedding_provider": self.settings.embedding_provider,
            "llm_provider": self.settings.llm_provider,
            "vector_store": self.settings.vector_store,
            "hybrid": self.settings.enable_hybrid,
            "rerank": self.settings.enable_rerank,
            "prompt_variant": self.settings.prompt_variant,
        }


_service: Service | None = None
_service_lock = threading.Lock()


def get_service() -> Service:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = Service()
    return _service


def reset_service() -> None:
    """Test hook: drop the cached singleton."""
    global _service
    with _service_lock:
        _service = None
