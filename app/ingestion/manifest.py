"""Ingestion manifest: tracks every processed document, independent of chunks.

Storing incremental state *only* in the vector index is subtly wrong: a document
whose chunks are all rejected by quality checks stores nothing, so its hash is
never recorded and it is reprocessed on every run. The manifest decouples "have
I seen this document version?" from "did this document produce any chunks?".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocRecord:
    doc_id: str
    rel_path: str
    content_hash: str
    chunk_count: int
    words: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "rel_path": self.rel_path,
            "content_hash": self.content_hash,
            "chunk_count": self.chunk_count,
            "words": self.words,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocRecord:
        return cls(
            doc_id=data["doc_id"],
            rel_path=data.get("rel_path", ""),
            content_hash=data["content_hash"],
            chunk_count=int(data.get("chunk_count", 0)),
            words=int(data.get("words", 0)),
        )


@dataclass
class Manifest:
    """Persisted map of doc_id -> last indexed version."""

    path: Path
    docs: dict[str, DocRecord] = field(default_factory=dict)

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.docs = {item["doc_id"]: DocRecord.from_dict(item) for item in raw.get("docs", [])}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "docs": [r.to_dict() for r in sorted(self.docs.values(), key=lambda r: r.doc_id)]
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self.path)  # atomic swap: no torn manifest on crash

    def hashes(self) -> dict[str, str]:
        return {doc_id: rec.content_hash for doc_id, rec in self.docs.items()}

    def record(
        self, doc_id: str, rel_path: str, content_hash: str, chunk_count: int, words: int
    ) -> None:
        self.docs[doc_id] = DocRecord(
            doc_id=doc_id,
            rel_path=rel_path,
            content_hash=content_hash,
            chunk_count=chunk_count,
            words=words,
        )

    def forget(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)

    def stats(self) -> dict[str, int]:
        return {
            "docs": len(self.docs),
            "docs_without_chunks": sum(1 for r in self.docs.values() if r.chunk_count == 0),
        }
