"""Structure-aware chunking: split on markdown headings, then by size.

Rationale: dbt docs are organised by heading, and headings are usually
self-contained topics. Splitting on headings yields semantically coherent
chunks; oversized sections are then split with overlap so answers spanning a
long section are still retrievable.
"""

from __future__ import annotations

import hashlib
import re

from app.config import Settings
from app.models import Chunk, ParsedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_sections(text: str) -> list[tuple[tuple[str, ...], int, str, int]]:
    """Return (heading_path, char_start, body, section_index) per heading anchor."""
    lines = text.splitlines()
    sections: list[tuple[tuple[str, ...], int, str, int]] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_path: tuple[str, ...] = ()
    start = 0
    offset = 0
    index = 0

    def flush() -> None:
        nonlocal index
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_path, start, body, index))
            index += 1

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = [(lv, tt) for lv, tt in heading_stack if lv < level]
            heading_stack.append((level, title))
            current_path = tuple(t for _, t in heading_stack)
            # the heading text lives in `section`, so keep it out of the body; the
            # body starts on the line after the heading
            current_lines = []
            start = offset + len(line) + 1
        else:
            current_lines.append(line)
        offset += len(line) + 1

    flush()
    return sections


def _window(body: str, max_chars: int, overlap: int) -> list[tuple[int, str]]:
    """Split an oversized body into overlapping windows on paragraph boundaries."""
    if len(body) <= max_chars:
        return [(0, body)]

    paragraphs = body.split("\n\n")
    windows: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_len = 0
    char_cursor = 0
    window_start = 0

    for para in paragraphs:
        plen = len(para) + 2
        if buf and buf_len + plen > max_chars:
            text = "\n\n".join(buf).strip()
            windows.append((window_start, text))
            # carry overlap from the tail of the previous window
            overlap_buf: list[str] = []
            overlap_len = 0
            for prev in reversed(buf):
                if overlap_len + len(prev) + 2 > overlap:
                    break
                overlap_buf.insert(0, prev)
                overlap_len += len(prev) + 2
            buf = overlap_buf
            buf_len = overlap_len
            window_start = char_cursor - overlap_len
        buf.append(para)
        buf_len += plen
        char_cursor += plen

    if buf:
        windows.append((max(window_start, 0), "\n\n".join(buf).strip()))
    return windows


def _fixed_sections(text: str, max_chars: int) -> list[tuple[tuple[str, ...], int, str, int]]:
    """Baseline strategy: ignore headings, slice into fixed-size windows."""
    sections: list[tuple[tuple[str, ...], int, str, int]] = []
    for i, start in enumerate(range(0, len(text), max_chars)):
        body = text[start : start + max_chars].strip()
        if body:
            sections.append(((), start, body, i))
    return sections


def chunk_document(doc: ParsedDocument, settings: Settings) -> list[Chunk]:
    """Chunk a parsed doc into structure-aware, size-bounded pieces."""
    chunks: list[Chunk] = []
    if settings.chunk_strategy == "fixed":
        sections = _fixed_sections(doc.text, settings.chunk_max_chars)
    else:
        sections = _split_sections(doc.text)
    # Preamble before the first heading still belongs to the doc.
    if not sections:
        sections = [((), 0, doc.text, 0)]

    for heading_path, sec_start, body, sec_index in sections:
        section_title = heading_path[-1] if heading_path else doc.title
        # fixed strategy already sized its windows; structure strategy may need to split
        windows = (
            [(0, body)]
            if settings.chunk_strategy == "fixed"
            else _window(body, settings.chunk_max_chars, settings.chunk_overlap_chars)
        )
        for win_start, window_text in windows:
            if not window_text.strip():
                continue
            char_start = sec_start + win_start
            char_end = char_start + len(window_text)
            chunk_id = hashlib.sha256(
                f"{doc.doc_id}:{sec_index}:{char_start}:{_hash(window_text)}".encode()
            ).hexdigest()[:24]
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=window_text,
                    doc_id=doc.doc_id,
                    doc_path=doc.rel_path,
                    title=doc.title,
                    section=section_title,
                    heading_path=heading_path,
                    section_index=sec_index,
                    char_start=char_start,
                    char_end=char_end,
                    content_hash=_hash(window_text),
                    metadata={
                        "word_count": len(window_text.split()),
                        "has_code": "```" in window_text,
                        "doc_description": doc.metadata.get("description", ""),
                        "doc_content_hash": doc.content_hash,
                    },
                )
            )
    return chunks
