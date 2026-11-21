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


_FENCE_OPEN_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def _fence_state_flags(lines: list[str]) -> list[bool]:
    """For each line, True if that line is *inside* a fenced code block (inclusive).

    Single source of truth for fence awareness: heading detection and paragraph
    splitting both rely on it, so they cannot drift apart.
    """
    inside: list[bool] = []
    fence: str | None = None
    for line in lines:
        match = _FENCE_OPEN_RE.match(line)
        if fence is None:
            if match:
                fence = match.group(1)[0] * 3
                inside.append(True)  # the opening fence itself
            else:
                inside.append(False)
        else:
            inside.append(True)
            if match and match.group(1).startswith(fence):
                fence = None
    return inside


def _split_code_safe(body: str) -> list[str]:
    """Split on blank lines, but never inside a fenced code block.

    A blank line inside a SQL/Jinja fence must not become a split point,
    otherwise a chunk starts or ends mid-statement.
    """
    lines = body.splitlines()
    inside = _fence_state_flags(lines)
    blocks: list[str] = []
    current: list[str] = []
    prev_blank = False

    for line, in_fence in zip(lines, inside, strict=True):
        blank = not line.strip()
        # a blank line separates blocks only when outside any fence
        if blank and not in_fence and prev_blank:
            if current:
                blocks.append("\n".join(current))
                current = []
            prev_blank = True
            continue
        if blank and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
            prev_blank = True
            continue
        current.append(line)
        prev_blank = False

    if current:
        blocks.append("\n".join(current))
    return [b for b in blocks if b.strip()]


def _split_sections(text: str) -> list[tuple[tuple[str, ...], int, str, int]]:
    """Return (heading_path, char_start, body, section_index) per heading anchor."""
    lines = text.splitlines()
    in_fence = _fence_state_flags(lines)
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

    for line, fenced in zip(lines, in_fence, strict=True):
        match = None if fenced else _HEADING_RE.match(line)
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
    """Split an oversized body into overlapping windows on paragraph boundaries.

    Paragraph boundaries come from a fence-aware splitter, so code blocks are
    never cut in half. Overlap also snaps to block boundaries for that reason.
    """
    if len(body) <= max_chars:
        return [(0, body)]

    blocks = _split_code_safe(body)
    if not blocks:
        return [(0, body)]

    windows: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_len = 0
    cursor = 0
    window_start = 0

    for block in blocks:
        plen = len(block) + 2
        # A single block larger than the budget cannot be split without cutting
        # a fence or a statement; emit it whole rather than truncate it.
        if plen > max_chars and not buf:
            windows.append((cursor, block.strip()))
            cursor += plen
            window_start = cursor
            continue

        if buf and buf_len + plen > max_chars:
            windows.append((window_start, "\n\n".join(buf).strip()))
            overlap_buf: list[str] = []
            overlap_len = 0
            for prev in reversed(buf):
                if overlap_len + len(prev) + 2 > overlap:
                    break
                overlap_buf.insert(0, prev)
                overlap_len += len(prev) + 2
            buf = overlap_buf
            buf_len = overlap_len
            window_start = max(cursor - overlap_len, 0)

        buf.append(block)
        buf_len += plen
        cursor += plen

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
