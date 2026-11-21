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


def test_incompatible_index_is_quarantined_not_fatal(tmp_path) -> None:
    """A stale index must never brick `ingest --force`.

    Regression test: load() used to raise on a dimension mismatch, which made
    the recommended remedy (re-ingest) impossible, since ingest also loads.
    """
    path = tmp_path / "index"
    old = LocalVectorStore(path, dim=8, embedding_id="hash:local")
    old.upsert([_chunk("a", "text")], np.zeros((1, 8), np.float32))
    old.save()

    new = LocalVectorStore(path, dim=4, embedding_id="mistral:mistral-embed")
    new.load()  # must not raise

    assert new.count() == 0, "stale index should not be served"
    assert new.quarantined_reason is not None
    assert new.quarantined_to is not None and new.quarantined_to.exists()
    assert not path.exists(), "stale dir is moved aside, leaving a clean path"

    # and the fresh store is fully usable
    new.upsert([_chunk("b", "text")], np.zeros((1, 4), np.float32))
    new.save()
    reloaded = LocalVectorStore(path, dim=4, embedding_id="mistral:mistral-embed")
    reloaded.load()
    assert reloaded.count() == 1
    assert reloaded.quarantined_reason is None


def test_provider_switch_is_quarantined(tmp_path) -> None:
    path = tmp_path / "index"
    old = LocalVectorStore(path, dim=4, embedding_id="hash:local")
    old.upsert([_chunk("a", "t")], np.zeros((1, 4), np.float32))
    old.save()

    new = LocalVectorStore(path, dim=4, embedding_id="mistral:mistral-embed")
    new.load()
    assert new.count() == 0
    assert "mistral:mistral-embed" in (new.quarantined_reason or "")


def test_corrupt_index_is_quarantined(tmp_path) -> None:
    import json

    path = tmp_path / "index"
    store = LocalVectorStore(path, dim=4)
    store.upsert([_chunk("a", "t")], np.zeros((1, 4), np.float32))
    store.save()

    payloads = json.loads((path / "payloads.json").read_text())
    payloads.append(dict(payloads[0], chunk_id="ghost"))
    (path / "payloads.json").write_text(json.dumps(payloads))

    broken = LocalVectorStore(path, dim=4)
    broken.load()
    assert broken.count() == 0
    assert "inconsistent" in (broken.quarantined_reason or "")


def test_replace_docs_swaps_one_doc_and_leaves_others(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "s", dim=2)
    store.upsert(
        [
            _chunk("a1", "old a", doc="a"),
            _chunk("a2", "old a2", doc="a"),
            _chunk("b1", "b", doc="b"),
        ],
        np.array([[1.0, 0], [1.0, 0], [0, 1.0]], np.float32),
    )
    new_vec = np.array([[0.6, 0.8]], np.float32)
    written = store.replace_docs(["a"], [_chunk("a3", "new a", doc="a")], new_vec)
    assert written == 1
    assert sorted(c.chunk_id for c in store.all_chunks()) == ["a3", "b1"]
    hit = store.search(np.array([0.6, 0.8], np.float32), top_k=1)[0]
    assert hit.chunk_id == "a3" and hit.chunk.text == "new a"


def test_upsert_with_duplicate_ids_in_one_batch_keeps_the_last(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "s", dim=2)
    store.upsert(
        [_chunk("x", "first"), _chunk("x", "second")], np.array([[1.0, 0], [0, 1.0]], np.float32)
    )
    assert store.count() == 1
    hit = store.search(np.array([0, 1.0], np.float32), top_k=1)[0]
    assert hit.chunk.text == "second" and abs(hit.score - 1.0) < 1e-6


def test_concurrent_reads_during_writes_never_fail(tmp_path) -> None:
    """Regression: /ask during /ingest could IndexError or pair a vector with the wrong payload."""
    import threading

    dim = 8
    store = LocalVectorStore(tmp_path / "s", dim=dim)
    bm25 = BM25Index()
    rng = np.random.default_rng(0)

    def batch(prefix: str, n: int) -> tuple[list[Chunk], np.ndarray]:
        chunks = [
            _chunk(f"{prefix}-{i}", f"incremental model {prefix} {i}", doc=prefix) for i in range(n)
        ]
        vecs = rng.normal(size=(n, dim)).astype(np.float32)
        return chunks, vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    store.upsert(*batch("seed", 20))
    bm25.build(store.all_chunks())
    stop = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        q = np.ones(dim, np.float32) / np.sqrt(dim)
        while not stop.is_set():
            try:
                for hit in store.search(q, top_k=5):
                    # the payload must belong to the row that was scored
                    assert hit.chunk.chunk_id.split("-")[0] == hit.chunk.doc_id
                bm25.search("incremental model", 5)
                store.count()
            except BaseException as exc:  # pragma: no cover - only on regression
                errors.append(exc)
                return

    import sys

    readers = [threading.Thread(target=reader) for _ in range(3)]
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads constantly to provoke interleavings
    for t in readers:
        t.start()
    try:
        for i in range(50):
            doc = f"d{i % 5}"
            if i % 3 == 0:
                store.upsert(*batch(doc, 1 + i % 7))
            elif i % 3 == 1:
                store.replace_docs([doc], *batch(doc, 1 + i % 4))
            else:
                store.delete_docs([doc])
            bm25.build(store.all_chunks())
    finally:
        stop.set()
        for t in readers:
            t.join()
        sys.setswitchinterval(old_interval)
    assert not errors, errors[0]
