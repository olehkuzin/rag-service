"""Shared domain models used across ingestion, retrieval, generation and eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text with full provenance metadata."""

    chunk_id: str
    text: str
    doc_id: str
    doc_path: str
    title: str
    section: str
    heading_path: tuple[str, ...] = ()
    section_index: int = 0
    char_start: int = 0
    char_end: int = 0
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Flat metadata dict stored alongside the vector."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "doc_path": self.doc_path,
            "title": self.title,
            "section": self.section,
            "heading_path": list(self.heading_path),
            "section_index": self.section_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "content_hash": self.content_hash,
            **self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Chunk:
        known = {
            "chunk_id",
            "text",
            "doc_id",
            "doc_path",
            "title",
            "section",
            "heading_path",
            "section_index",
            "char_start",
            "char_end",
            "content_hash",
        }
        return cls(
            chunk_id=payload["chunk_id"],
            text=payload["text"],
            doc_id=payload["doc_id"],
            doc_path=payload["doc_path"],
            title=payload["title"],
            section=payload["section"],
            heading_path=tuple(payload.get("heading_path", ())),
            section_index=payload.get("section_index", 0),
            char_start=payload.get("char_start", 0),
            char_end=payload.get("char_end", 0),
            content_hash=payload.get("content_hash", ""),
            metadata={k: v for k, v in payload.items() if k not in known},
        )


@dataclass(slots=True)
class ScoredChunk:
    """A chunk plus the scores that produced its rank (for tracing/debugging)."""

    chunk: Chunk
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    rank: int = 0

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass(slots=True)
class ParsedDocument:
    """Output of the parse step: cleaned markdown with frontmatter metadata."""

    doc_id: str
    path: str
    rel_path: str
    title: str
    text: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)
