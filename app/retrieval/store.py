"""Vector store abstraction.

`LocalVectorStore` persists embeddings + payloads to disk with numpy, so the
service runs with no Docker or server. `QdrantVectorStore` is the production
adapter; both satisfy `VectorStore`, so retrieval code is unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.models import Chunk, ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int: ...

    def search(
        self, vector: np.ndarray, top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]: ...

    def delete_docs(self, doc_ids: Sequence[str]) -> int: ...

    def existing_hashes(self) -> dict[str, str]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def count(self) -> int: ...


class LocalVectorStore:
    """Numpy-backed cosine store with JSON metadata, persisted to a directory."""

    def __init__(self, path: Path, dim: int) -> None:
        self.path = Path(path)
        self.dim = dim
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._payloads: list[dict[str, Any]] = []

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        vec_file = self.path / "vectors.npy"
        meta_file = self.path / "payloads.json"
        if vec_file.exists() and meta_file.exists():
            self._vectors = np.load(vec_file)
            self._payloads = json.loads(meta_file.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "vectors.npy", self._vectors)
        (self.path / "payloads.json").write_text(
            json.dumps(self._payloads, ensure_ascii=False), encoding="utf-8"
        )

    # -- writes ------------------------------------------------------------
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if len(chunks) == 0:
            return 0

        by_id = {p["chunk_id"]: i for i, p in enumerate(self._payloads)}
        new_vecs: list[np.ndarray] = []
        new_payloads: list[dict[str, Any]] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            payload = chunk.to_payload()
            if chunk.chunk_id in by_id:
                self._payloads[by_id[chunk.chunk_id]] = payload
                self._vectors[by_id[chunk.chunk_id]] = vec
            else:
                new_payloads.append(payload)
                new_vecs.append(vec)

        if new_vecs:
            self._vectors = np.vstack([self._vectors, np.vstack(new_vecs).astype(np.float32)])
            self._payloads.extend(new_payloads)
        return len(chunks)

    def delete_docs(self, doc_ids: Sequence[str]) -> int:
        targets = set(doc_ids)
        keep = [i for i, p in enumerate(self._payloads) if p["doc_id"] not in targets]
        removed = len(self._payloads) - len(keep)
        if removed:
            self._vectors = self._vectors[keep] if keep else np.zeros((0, self.dim), np.float32)
            self._payloads = [self._payloads[i] for i in keep]
        return removed

    # -- reads -------------------------------------------------------------
    def search(
        self, vector: np.ndarray, top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        if len(self._payloads) == 0:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        scores = self._vectors @ q  # vectors are L2-normalised at embed time

        if doc_ids:
            allowed = set(doc_ids)
            mask = np.array([p["doc_id"] in allowed for p in self._payloads])
            scores = np.where(mask, scores, -np.inf)

        top_k = min(top_k, len(scores))
        idxs = np.argpartition(-scores, top_k - 1)[:top_k]
        idxs = idxs[np.argsort(-scores[idxs])]
        results = []
        for rank, idx in enumerate(idxs):
            score = float(scores[idx])
            if not np.isfinite(score):
                continue
            results.append(
                ScoredChunk(
                    chunk=Chunk.from_payload(self._payloads[idx]),
                    score=score,
                    vector_score=score,
                    rank=rank + 1,
                )
            )
        return results

    def existing_hashes(self) -> dict[str, str]:
        """Map doc_id -> doc-level content hash (not the per-chunk hash)."""
        out: dict[str, str] = {}
        for p in self._payloads:
            out[p["doc_id"]] = p.get("doc_content_hash", "")
        return out

    def all_chunks(self) -> list[Chunk]:
        return [Chunk.from_payload(p) for p in self._payloads]

    def count(self) -> int:
        return len(self._payloads)

    def chunks_by_doc(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in self._payloads:
            out.setdefault(p["doc_id"], []).append(p["chunk_id"])
        return out


class QdrantVectorStore:
    """Qdrant adapter. Selected when RAG_VECTOR_STORE=qdrant."""

    def __init__(self, url: str, collection: str, dim: int) -> None:
        self.url = url
        self.collection = collection
        self.dim = dim

    def _client(self) -> Any:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=self.url)
        if not client.collection_exists(self.collection):
            from qdrant_client.models import Distance, VectorParams

            client.create_collection(
                self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
        return client

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        from qdrant_client.models import PointStruct

        client = self._client()
        points = [
            PointStruct(
                id=_stable_point_id(chunk.chunk_id),
                vector=vec.tolist(),
                payload=chunk.to_payload(),
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        client.upsert(self.collection, points=points)
        return len(points)

    def search(
        self, vector: np.ndarray, top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        client = self._client()
        query_filter = (
            Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=list(doc_ids)))])
            if doc_ids
            else None
        )
        hits = client.search(
            self.collection,
            query_vector=np.asarray(vector, dtype=np.float32).tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk=Chunk.from_payload(hit.payload or {}),
                score=float(hit.score),
                vector_score=float(hit.score),
                rank=i + 1,
            )
            for i, hit in enumerate(hits)
        ]

    def delete_docs(self, doc_ids: Sequence[str]) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        client = self._client()
        client.delete(
            self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=list(doc_ids)))]
            ),
        )
        return len(doc_ids)

    def existing_hashes(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for chunk in self.all_chunks():
            out[chunk.doc_id] = str(chunk.metadata.get("doc_content_hash", ""))
        return out

    def all_chunks(self) -> list[Chunk]:
        client = self._client()
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = client.scroll(
                self.collection, limit=512, offset=offset, with_payload=True, with_vectors=False
            )
            chunks.extend(Chunk.from_payload(p.payload or {}) for p in points)
            if offset is None:
                break
        return chunks

    def count(self) -> int:
        total: int = self._client().count(self.collection).count
        return total


def _stable_point_id(chunk_id: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(chunk_id.encode(), digest_size=8).digest(), "big") % (
        2**63
    )
