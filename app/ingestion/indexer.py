"""Incremental ingestion pipeline: discover -> parse -> chunk -> QC -> embed -> upsert.

Incremental by content hash: a document is re-processed only when its cleaned
text changes. Changed docs have their old chunks replaced so deleted sections do
not linger in the index.

A run is all-or-nothing: every changed document is parsed, chunked, checked and
embedded first, and only then are the changes applied to the store in one
`replace_docs` call. If a provider fails mid-run (401, exhausted retries), the
store and manifest are exactly as before and the next run simply retries; the
trade-off is that embeddings computed before the failure are not kept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.ingestion.chunker import chunk_document
from app.ingestion.manifest import Manifest
from app.ingestion.parser import discover_docs, parse_file
from app.ingestion.quality import QualityReport, check_chunks
from app.models import Chunk
from app.providers import Embedder
from app.retrieval.store import VectorStore

EMBED_BATCH = 64


@dataclass(slots=True)
class IngestStats:
    docs_seen: int = 0
    docs_changed: int = 0
    docs_skipped: int = 0
    docs_empty: int = 0
    docs_failed: int = 0
    docs_removed: int = 0
    chunks_created: int = 0
    chunks_dropped: int = 0
    chunks_written: int = 0
    vectors_embedded: int = 0
    latency_ms: float = 0.0
    quality: QualityReport | None = None
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "docs_seen": self.docs_seen,
            "docs_changed": self.docs_changed,
            "docs_skipped": self.docs_skipped,
            "docs_empty": self.docs_empty,
            "docs_failed": self.docs_failed,
            "docs_removed": self.docs_removed,
            "chunks_created": self.chunks_created,
            "chunks_dropped": self.chunks_dropped,
            "chunks_written": self.chunks_written,
            "vectors_embedded": self.vectors_embedded,
            "latency_ms": round(self.latency_ms, 1),
            "quality": self.quality.summary() if self.quality else None,
            "errors": self.errors[:20],
        }


class Ingestor:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings,
        corpus_dir: Path | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.corpus_dir = corpus_dir or settings.corpus_dir
        self.manifest = Manifest(settings.index_dir / "manifest.json")

    def run(self, force: bool = False, limit: int | None = None) -> IngestStats:
        started = time.perf_counter()
        stats = IngestStats()
        report = QualityReport()
        self.manifest.load()  # re-read from disk so concurrent edits converge

        files = discover_docs(self.corpus_dir)
        if limit is not None:
            files = files[:limit]
        stats.docs_seen = len(files)

        # Incremental state comes from the manifest, not the store: a doc whose
        # chunks were all rejected must still be recorded as processed.
        existing = {} if force else self.manifest.hashes()

        seen_doc_ids: set[str] = set()
        replace_ids: list[str] = []
        staged_chunks: list[Chunk] = []
        staged_vectors: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        for path in files:
            try:
                doc = parse_file(path, self.corpus_dir)
            except OSError as exc:
                stats.docs_failed += 1
                stats.errors.append(f"unreadable {path.relative_to(self.corpus_dir)}: {exc}")
                continue
            if doc is None:
                # valid file with no indexable body: expected, not a failure
                stats.docs_empty += 1
                continue

            seen_doc_ids.add(doc.doc_id)

            if not force and existing.get(doc.doc_id) == doc.content_hash:
                stats.docs_skipped += 1
                continue

            stats.docs_changed += 1
            replace_ids.append(doc.doc_id)  # the old version's chunks are replaced on commit

            raw_chunks: list[Chunk] = chunk_document(doc, self.settings)
            stats.chunks_created += len(raw_chunks)
            kept, partial = check_chunks(raw_chunks, self.settings)
            _merge_report(report, partial)
            stats.chunks_dropped += len(raw_chunks) - len(kept)

            if kept:
                staged_vectors.append(self._embed(kept))
                staged_chunks.extend(kept)

            # record even when chunk_count == 0, so the doc converges
            records.append(
                {
                    "doc_id": doc.doc_id,
                    "rel_path": doc.rel_path,
                    "content_hash": doc.content_hash,
                    "chunk_count": len(kept),
                    "words": int(doc.metadata.get("word_count", 0)),
                }
            )

        # prune documents whose files disappeared from the corpus
        removed = sorted(set(self.manifest.docs) - seen_doc_ids) if limit is None else []
        stats.docs_removed = len(removed)

        # --- commit: only reached when every embedding call succeeded ---
        if replace_ids or removed:
            vectors = (
                np.vstack(staged_vectors)
                if staged_vectors
                else np.zeros((0, self.embedder.dim), dtype=np.float32)
            )
            stats.chunks_written = self.store.replace_docs(
                [*replace_ids, *removed], staged_chunks, vectors
            )
            stats.vectors_embedded = len(staged_chunks)
        for record in records:
            self.manifest.record(**record)
        for doc_id in removed:
            self.manifest.forget(doc_id)

        # store before manifest: a crash in between re-processes docs (harmless)
        # instead of recording docs as indexed that the store never received
        if hasattr(self.store, "save"):
            self.store.save()
        self.manifest.save()
        stats.quality = report
        stats.latency_ms = (time.perf_counter() - started) * 1000
        return stats

    def _embed(self, chunks: list[Chunk]) -> np.ndarray:
        batches = [
            self.embedder.embed_documents([c.text for c in chunks[start : start + EMBED_BATCH]])
            for start in range(0, len(chunks), EMBED_BATCH)
        ]
        return np.vstack(batches)


def _merge_report(agg: QualityReport, part: QualityReport) -> None:
    agg.total_in += part.total_in
    agg.total_out += part.total_out
    agg.dropped.extend(part.dropped)
    agg.warnings.extend(part.warnings)
    agg.dupe_groups += part.dupe_groups
