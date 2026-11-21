"""Incremental ingestion pipeline: discover -> parse -> chunk -> QC -> embed -> upsert.

Incremental by content hash: a document is re-processed only when its cleaned
text changes. Changed docs have their old chunks removed first so deleted
sections do not linger in the index.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

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
            # remove the old version's chunks before writing the new ones
            self.store.delete_docs([doc.doc_id])

            raw_chunks: list[Chunk] = chunk_document(doc, self.settings)
            stats.chunks_created += len(raw_chunks)
            kept, partial = check_chunks(raw_chunks, self.settings)
            _merge_report(report, partial)
            stats.chunks_dropped += len(raw_chunks) - len(kept)

            written = self._embed_and_store(kept) if kept else 0
            stats.chunks_written += written
            stats.vectors_embedded += written

            # record even when chunk_count == 0, so the doc converges
            self.manifest.record(
                doc_id=doc.doc_id,
                rel_path=doc.rel_path,
                content_hash=doc.content_hash,
                chunk_count=written,
                words=int(doc.metadata.get("word_count", 0)),
            )

        # prune manifest entries whose files disappeared from the corpus
        if limit is None:
            for doc_id in set(self.manifest.docs) - seen_doc_ids:
                self.manifest.forget(doc_id)
                self.store.delete_docs([doc_id])
                stats.docs_removed += 1

        self.manifest.save()
        if hasattr(self.store, "save"):
            self.store.save()
        stats.quality = report
        stats.latency_ms = (time.perf_counter() - started) * 1000
        return stats

    def _embed_and_store(self, chunks: list[Chunk]) -> int:
        written = 0
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            vectors = self.embedder.embed_documents([c.text for c in batch])
            written += self.store.upsert(batch, vectors)
        return written


def _merge_report(agg: QualityReport, part: QualityReport) -> None:
    agg.total_in += part.total_in
    agg.total_out += part.total_out
    agg.dropped.extend(part.dropped)
    agg.warnings.extend(part.warnings)
    agg.dupe_groups += part.dupe_groups
