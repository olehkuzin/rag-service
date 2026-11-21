"""Retrieval: BM25, RRF, reranking, store behaviour, stage toggles."""

from __future__ import annotations

import numpy as np
from app.models import Chunk, ScoredChunk
from app.providers import HashEmbedder, LexicalReranker
from app.retrieval.hybrid import BM25Index, reciprocal_rank_fusion, tokenize
from app.retrieval.pipeline import RetrievalConfig
from app.retrieval.store import LocalVectorStore
from app.service import Service


def _chunk(cid: str, text: str, doc: str = "d1") -> Chunk:
    return Chunk(
        chunk_id=cid,
        text=text,
        doc_id=doc,
        doc_path=f"{doc}.md",
        title="Title",
        section="Section",
        content_hash=cid,
    )


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("What is a dbt model?") == ["dbt", "model"]


def test_hash_embedder_is_deterministic_and_normalised() -> None:
    emb = HashEmbedder(dim=32)
    a = emb.embed_query("incremental model")
    b = emb.embed_query("incremental model")
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5


def test_hash_embedder_batch_shape() -> None:
    emb = HashEmbedder(dim=16)
    vectors = emb.embed_documents(["one", "two", "three"])
    assert vectors.shape == (3, 16)
    assert emb.embed_documents([]).shape == (0, 16)


def test_bm25_ranks_matching_chunk_first() -> None:
    idx = BM25Index()
    idx.build(
        [
            _chunk("a", "incremental models transform only new rows"),
            _chunk("b", "generic tests assert uniqueness of a column"),
            _chunk("c", "python models use dataframes"),
        ]
    )
    hits = idx.search("incremental rows", top_k=3)
    assert hits[0].chunk_id == "a"
    assert hits[0].bm25_score is not None


def test_bm25_empty_index_returns_nothing() -> None:
    assert BM25Index().search("anything", top_k=5) == []


def test_rrf_rewards_agreement_across_runs() -> None:
    c1, c2, c3 = _chunk("c1", "x"), _chunk("c2", "y"), _chunk("c3", "z")
    run_a = [ScoredChunk(c1, 0.9, vector_score=0.9), ScoredChunk(c2, 0.8, vector_score=0.8)]
    run_b = [ScoredChunk(c2, 3.0, bm25_score=3.0), ScoredChunk(c3, 1.0, bm25_score=1.0)]
    fused = reciprocal_rank_fusion([run_a, run_b], k=60)
    # c2 appears in both runs -> should win
    assert fused[0].chunk_id == "c2"
    assert fused[0].bm25_score == 3.0
    assert fused[0].vector_score == 0.8


def test_rrf_respects_top_k() -> None:
    runs = [[ScoredChunk(_chunk(f"c{i}", "t"), float(i)) for i in range(5)]]
    assert len(reciprocal_rank_fusion(runs, top_k=2)) == 2


def test_store_upsert_updates_in_place(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "s", dim=4)
    vec = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
    store.upsert([_chunk("a", "old")], vec)
    store.upsert([_chunk("a", "new")], vec)
    assert store.count() == 1
    assert store.all_chunks()[0].text == "new"


def test_store_persists_and_reloads(tmp_path) -> None:
    path = tmp_path / "s"
    store = LocalVectorStore(path, dim=4)
    store.upsert([_chunk("a", "hello")], np.array([[1.0, 0, 0, 0]], np.float32))
    store.save()

    reloaded = LocalVectorStore(path, dim=4)
    reloaded.load()
    assert reloaded.count() == 1
    hits = reloaded.search(np.array([1.0, 0, 0, 0], np.float32), top_k=1)
    assert hits[0].chunk_id == "a"


def test_store_delete_docs(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "s", dim=2)
    store.upsert(
        [_chunk("a", "one", doc="d1"), _chunk("b", "two", doc="d2")],
        np.array([[1.0, 0], [0, 1.0]], np.float32),
    )
    assert store.delete_docs(["d1"]) == 1
    assert store.count() == 1
    assert store.all_chunks()[0].doc_id == "d2"


def test_lexical_reranker_prefers_exact_overlap() -> None:
    reranker = LexicalReranker()
    scores = reranker.rerank(
        "incremental model",
        ["incremental models only add new rows", "generic tests check uniqueness"],
    )
    assert scores[0] > scores[1]


def test_retriever_stage_toggles_change_pipeline(service: Service) -> None:
    query = "incremental models"
    base = service.retriever.retrieve(
        query, RetrievalConfig(enable_hybrid=False, enable_rerank=False, top_k=3)
    )
    assert "bm25" not in base.stages
    assert "reranked" not in base.stages

    hybrid = service.retriever.retrieve(
        query, RetrievalConfig(enable_hybrid=True, enable_rerank=True, top_k=3)
    )
    assert "bm25" in hybrid.stages
    assert "reranked" in hybrid.stages
    assert len(hybrid.chunks) <= 3
    assert all(0 < c.rank <= 3 for c in hybrid.chunks)


def test_retriever_doc_filter(service: Service) -> None:
    doc_ids = [c.doc_id for c in service.store.all_chunks()][:1]
    result = service.retriever.retrieve(
        "models",
        RetrievalConfig(enable_hybrid=False, enable_rerank=False, top_k=5),
        doc_ids=doc_ids,
    )
    assert all(c.chunk.doc_id in doc_ids for c in result.chunks)
