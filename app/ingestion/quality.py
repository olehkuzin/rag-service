"""Data quality checks for ingested chunks.

Runs between chunking and embedding. Bad chunks are dropped, and the reason is
recorded so ingestion telemetry can show *why* documents lose content rather
than silently shrinking the corpus.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.config import Settings
from app.models import Chunk

_NAV_LINE_RE = re.compile(
    r"^(?:[-*]\s*)?\[(?:next|previous|prev|back|home)\][^\n]*$", re.IGNORECASE
)
_BOILERPLATE_PATTERNS = (
    re.compile(r"^was this (?:page|article) helpful\?", re.IGNORECASE),
    re.compile(r"^sign up for (?:the )?(?:dbt )?newsletter", re.IGNORECASE),
    re.compile(r"^edit this page", re.IGNORECASE),
    re.compile(r"^table of contents$", re.IGNORECASE),
)


@dataclass(slots=True)
class QualityIssue:
    chunk_id: str
    doc_path: str
    rule: str
    detail: str


@dataclass(slots=True)
class QualityReport:
    """Aggregate result of a quality pass."""

    total_in: int = 0
    total_out: int = 0
    dropped: list[QualityIssue] = field(default_factory=list)
    warnings: list[QualityIssue] = field(default_factory=list)
    dupe_groups: int = 0

    @property
    def drop_rate(self) -> float:
        return len(self.dropped) / self.total_in if self.total_in else 0.0

    def reasons(self) -> dict[str, int]:
        return dict(Counter(i.rule for i in self.dropped))

    def summary(self) -> dict[str, object]:
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "dropped": len(self.dropped),
            "warnings": len(self.warnings),
            "dupe_groups": self.dupe_groups,
            "drop_rate": round(self.drop_rate, 4),
            "drop_reasons": self.reasons(),
        }


def _is_boilerplate(text: str) -> str | None:
    stripped = text.strip()
    for line in stripped.splitlines():
        candidate = line.strip()
        for pattern in _BOILERPLATE_PATTERNS:
            if pattern.search(candidate):
                return candidate[:80]
    if _NAV_LINE_RE.match(stripped):
        return stripped[:80]
    return None


def _required_metadata_missing(chunk: Chunk) -> list[str]:
    missing = []
    for name in ("doc_id", "doc_path", "title", "section"):
        if not str(getattr(chunk, name)).strip():
            missing.append(name)
    return missing


def check_chunks(chunks: list[Chunk], settings: Settings) -> tuple[list[Chunk], QualityReport]:
    """Apply structural quality rules. Returns (kept_chunks, report)."""
    report = QualityReport(total_in=len(chunks))
    kept: list[Chunk] = []
    seen_hashes: dict[str, str] = {}
    dupes: dict[str, list[str]] = defaultdict(list)

    for chunk in chunks:
        text = chunk.text.strip()

        if not text:
            report.dropped.append(
                QualityIssue(chunk.chunk_id, chunk.doc_path, "empty", "chunk text is blank")
            )
            continue

        if len(text) < settings.chunk_min_chars:
            report.dropped.append(
                QualityIssue(
                    chunk.chunk_id,
                    chunk.doc_path,
                    "too_short",
                    f"{len(text)} < {settings.chunk_min_chars} chars",
                )
            )
            continue

        missing = _required_metadata_missing(chunk)
        if missing:
            report.dropped.append(
                QualityIssue(
                    chunk.chunk_id,
                    chunk.doc_path,
                    "missing_metadata",
                    f"missing: {', '.join(missing)}",
                )
            )
            continue

        boiler = _is_boilerplate(text)
        if boiler and len(text) < settings.chunk_min_chars * 2:
            report.dropped.append(
                QualityIssue(chunk.chunk_id, chunk.doc_path, "boilerplate", boiler)
            )
            continue

        if chunk.content_hash in seen_hashes:
            dupes[chunk.content_hash].append(chunk.chunk_id)
            report.dropped.append(
                QualityIssue(
                    chunk.chunk_id,
                    chunk.doc_path,
                    "duplicate",
                    f"identical to {seen_hashes[chunk.content_hash]}",
                )
            )
            continue
        seen_hashes[chunk.content_hash] = chunk.chunk_id

        # Non-fatal: oversized chunk means chunker config may need tuning.
        if len(text) > settings.chunk_max_chars * 1.2:
            report.warnings.append(
                QualityIssue(
                    chunk.chunk_id,
                    chunk.doc_path,
                    "oversized",
                    f"{len(text)} > {settings.chunk_max_chars}",
                )
            )

        kept.append(chunk)

    report.total_out = len(kept)
    report.dupe_groups = sum(1 for ids in dupes.values() if ids)
    return kept, report
