"""Vector store abstraction.

`LocalVectorStore` persists embeddings + payloads to disk with numpy, so the
service runs with no Docker or server — this is the default and the only store
used in local deployments. `QdrantVectorStore` is an optional adapter (requires
`qdrant-client`); both satisfy `VectorStore`, so retrieval code is unchanged.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
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

    def replace_docs(
        self, doc_ids: Sequence[str], chunks: Sequence[Chunk], vectors: np.ndarray
    ) -> int: ...

    def existing_hashes(self) -> dict[str, str]: ...

    def all_chunks(self) -> list[Chunk]: ...

    def count(self) -> int: ...


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """Vectors and their payloads, row-aligned. Never mutated once published."""

    vectors: np.ndarray
    payloads: tuple[dict[str, Any], ...]


class LocalVectorStore:
    """Numpy-backed cosine store with JSON metadata, persisted to a directory.

    Safe for concurrent readers during a write: all state lives in one immutable
    `_Snapshot`; writers build a new one (copy-on-write) and publish it with a
    single assignment, and every read works on the snapshot it grabbed first.
    Writers themselves must be serialised by the caller (the service holds a lock).
    """

    def __init__(self, path: Path, dim: int, embedding_id: str = "") -> None:
        self.path = Path(path)
        self.dim = dim
        # e.g. "mistral:mistral-embed" — lets load() detect a provider switch
        self.embedding_id = embedding_id
        self._snap = _Snapshot(np.zeros((0, dim), dtype=np.float32), ())
        # set by load() when an incompatible index was set aside
        self.quarantined_reason: str | None = None
        self.quarantined_to: Path | None = None

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        """Load a persisted index, quarantining it if it cannot be used.

        An incompatible or corrupt index must not brick the process: every
        command would fail at load time, *including* `ingest --force` that is
        supposed to repair it. So instead of raising, we move the bad index
        aside (so `--force` can rebuild cleanly) and start empty.
        """
        vec_file = self.path / "vectors.npy"
        meta_file = self.path / "payloads.json"
        if not (vec_file.exists() and meta_file.exists()):
            return

        try:
            vectors = np.load(vec_file)
            self._check_compatible(vectors)
            payloads = json.loads(meta_file.read_text(encoding="utf-8"))
            if len(payloads) != len(vectors):
                raise ValueError(
                    f"index is inconsistent: {len(vectors)} vectors vs {len(payloads)} payloads"
                )
        except ValueError as exc:
            self._quarantine(reason=str(exc))
            return

        self._snap = _Snapshot(vectors.astype(np.float32, copy=False), tuple(payloads))

    def _check_compatible(self, vectors: np.ndarray) -> None:
        actual = vectors.shape[1] if vectors.ndim == 2 else None
        if actual != self.dim:
            raise ValueError(
                f"persisted index dimension {actual or 'unknown'} != configured {self.dim}"
            )
        meta_file = self.path / "index_meta.json"
        if meta_file.exists() and self.embedding_id:
            stored = json.loads(meta_file.read_text(encoding="utf-8")).get("embedding_id")
            if stored and stored != self.embedding_id:
                raise ValueError(f"index built with {stored!r} != configured {self.embedding_id!r}")

    def _quarantine(self, reason: str) -> None:
        """Move an unusable index aside so it is not silently reused."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = self.path.with_name(f"{self.path.name}.stale-{stamp}")
        try:
            self.path.rename(dest)
        except OSError:  # pragma: no cover - filesystem edge case
            shutil.rmtree(self.path, ignore_errors=True)
            dest = self.path
        self.quarantined_reason = reason
        self.quarantined_to = dest

    def save(self) -> None:
        """Persist the current snapshot; each file is replaced atomically."""
        snap = self._snap
        self.path.mkdir(parents=True, exist_ok=True)

        def write(name: str, writer: Any) -> None:
            tmp = self.path / f".{name}.tmp"
            with tmp.open("wb") as fh:
                writer(fh)
            tmp.replace(self.path / name)

        write("vectors.npy", lambda fh: np.save(fh, snap.vectors))
        payloads_json = json.dumps(list(snap.payloads), ensure_ascii=False).encode("utf-8")
        write("payloads.json", lambda fh: fh.write(payloads_json))
        meta = {"embedding_id": self.embedding_id, "dim": self.dim, "count": len(snap.payloads)}
        write("index_meta.json", lambda fh: fh.write(json.dumps(meta, indent=1).encode("utf-8")))

    # -- writes (copy-on-write; callers serialise writers) -----------------
    def _replace(
        self, remove_doc_ids: Sequence[str], chunks: Sequence[Chunk], vectors: np.ndarray
    ) -> int:
        """Drop `remove_doc_ids`, then upsert `chunks`, publishing one new snapshot."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        snap = self._snap
        targets = set(remove_doc_ids)
        keep = [i for i, p in enumerate(snap.payloads) if p["doc_id"] not in targets]
        removed = len(snap.payloads) - len(keep)

        rows = np.asarray(vectors, dtype=np.float32).reshape(len(chunks), self.dim)
        merged_vecs = snap.vectors[keep]  # fancy indexing copies
        merged_payloads = [snap.payloads[i] for i in keep]
        position = {p["chunk_id"]: i for i, p in enumerate(merged_payloads)}
        base = len(merged_payloads)  # rows < base live in merged_vecs, the rest in new_vecs
        new_vecs: list[np.ndarray] = []
        for chunk, vec in zip(chunks, rows, strict=True):
            payload = chunk.to_payload()
            if chunk.chunk_id in position:
                row = position[chunk.chunk_id]
                merged_payloads[row] = payload
                if row < base:
                    merged_vecs[row] = vec
                else:
                    new_vecs[row - base] = vec
            else:
                position[chunk.chunk_id] = len(merged_payloads)
                merged_payloads.append(payload)
                new_vecs.append(vec)
        if new_vecs:
            merged_vecs = np.vstack([merged_vecs, np.vstack(new_vecs)])

        if removed or len(chunks):
            self._snap = _Snapshot(merged_vecs, tuple(merged_payloads))
        return removed

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        self._replace((), chunks, vectors)
        return len(chunks)

    def delete_docs(self, doc_ids: Sequence[str]) -> int:
        return self._replace(doc_ids, (), np.zeros((0, self.dim), np.float32))

    def replace_docs(
        self, doc_ids: Sequence[str], chunks: Sequence[Chunk], vectors: np.ndarray
    ) -> int:
        """Atomically drop every chunk of `doc_ids` and insert `chunks`. Returns chunks written."""
        self._replace(doc_ids, chunks, vectors)
        return len(chunks)

    # -- reads -------------------------------------------------------------
    def search(
        self, vector: np.ndarray, top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        snap = self._snap  # one consistent view for the whole search
        if len(snap.payloads) == 0:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        scores = snap.vectors @ q  # vectors are L2-normalised at embed time

        if doc_ids:
            allowed = set(doc_ids)
            mask = np.array([p["doc_id"] in allowed for p in snap.payloads])
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
                    chunk=Chunk.from_payload(snap.payloads[idx]),
                    score=score,
                    vector_score=score,
                    rank=rank + 1,
                )
            )
        return results

    def existing_hashes(self) -> dict[str, str]:
        """Map doc_id -> doc-level content hash (not the per-chunk hash)."""
        out: dict[str, str] = {}
        for p in self._snap.payloads:
            out[p["doc_id"]] = p.get("doc_content_hash", "")
        return out

    def all_chunks(self) -> list[Chunk]:
        return [Chunk.from_payload(p) for p in self._snap.payloads]

    def count(self) -> int:
        return len(self._snap.payloads)

    def chunks_by_doc(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in self._snap.payloads:
            out.setdefault(p["doc_id"], []).append(p["chunk_id"])
        return out


class QdrantVectorStore:
    """Qdrant adapter. Selected when RAG_VECTOR_STORE=qdrant."""

    def __init__(self, url: str, collection: str, dim: int) -> None:
        self.url = url
        self.collection = collection
        self.dim = dim

    def _client(self) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # optional dependency, not installed by default
            raise RuntimeError(
                "RAG_VECTOR_STORE=qdrant requires the optional dependency: "
                "`uv pip install qdrant-client`, and a running Qdrant instance "
                f"at {self.url}. The default local numpy store needs neither."
            ) from exc

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

    def replace_docs(
        self, doc_ids: Sequence[str], chunks: Sequence[Chunk], vectors: np.ndarray
    ) -> int:
        # Not atomic on Qdrant: readers can briefly see the docs missing between the calls.
        if doc_ids:
            self.delete_docs(doc_ids)
        return self.upsert(chunks, vectors) if len(chunks) else 0

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
