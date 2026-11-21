"""Retrieval pipeline: vector -> (bm25 -> RRF) -> rerank.

Stage toggles exist purely so the eval harness can measure each variant
independently (baseline / +hybrid / +rerank).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as dc_replace

from app.config import Settings
from app.models import ScoredChunk
from app.providers import Embedder, Reranker
from app.retrieval.hybrid import BM25Index, reciprocal_rank_fusion
from app.retrieval.store import VectorStore


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    stages: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(slots=True)
class RetrievalConfig:
    """Per-call overrides of the global feature flags (used by eval variants)."""

    enable_hybrid: bool = True
    enable_rerank: bool = True
    top_k: int = 5
    vector_candidates: int = 20
    bm25_candidates: int = 20
    rerank_candidates: int = 20

    @classmethod
    def from_settings(cls, settings: Settings, top_k: int | None = None) -> RetrievalConfig:
        return cls(
            enable_hybrid=settings.enable_hybrid,
            enable_rerank=settings.enable_rerank,
            top_k=top_k or settings.default_top_k,
            vector_candidates=settings.vector_candidates,
            bm25_candidates=settings.bm25_candidates,
            rerank_candidates=settings.rerank_candidates,
        )


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.bm25 = bm25 if bm25 is not None else BM25Index()
        self.reranker = reranker

    def rebuild_bm25(self) -> int:
        chunks = self.store.all_chunks()
        self.bm25.build(chunks)
        return len(self.bm25)

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
        doc_ids: list[str] | None = None,
    ) -> RetrievalResult:
        import time

        cfg = config or RetrievalConfig.from_settings(self.settings)
        started = time.perf_counter()

        query_vec = self.embedder.embed_query(query)
        vector_hits = self.store.search(query_vec, cfg.vector_candidates, doc_ids=doc_ids)
        stages = {"vector": len(vector_hits)}

        candidates = vector_hits
        if cfg.enable_hybrid:
            bm25_hits = self.bm25.search(query, cfg.bm25_candidates)
            if doc_ids:
                allowed = set(doc_ids)
                bm25_hits = [h for h in bm25_hits if h.chunk.doc_id in allowed]
            stages["bm25"] = len(bm25_hits)
            candidates = reciprocal_rank_fusion([vector_hits, bm25_hits], k=self.settings.rrf_k)
        stages["after_fusion"] = len(candidates)

        if cfg.enable_rerank and self.reranker is not None and candidates:
            pool = candidates[: cfg.rerank_candidates]
            scores = self.reranker.rerank(query, [c.chunk.text for c in pool])
            pool = [
                dc_replace(item, rerank_score=score, score=score)
                for item, score in zip(pool, scores, strict=True)
            ]
            pool.sort(key=lambda c: c.score, reverse=True)
            candidates = pool
            stages["reranked"] = len(pool)

        final = candidates[: cfg.top_k]
        for rank, item in enumerate(final, start=1):
            item.rank = rank

        return RetrievalResult(
            chunks=final,
            stages=stages,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
