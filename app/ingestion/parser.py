"""Parse dbt docs markdown/MDX into cleaned text with frontmatter metadata.

The corpus is Docusaurus MDX: YAML frontmatter, JSX components (`<Constant ... />`,
`<Tabs>`, `<div>`) and imports. Those must be stripped before chunking, otherwise
the embedder sees markup noise instead of prose.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import frontmatter

from app.models import ParsedDocument

# --- cleaning rules -------------------------------------------------------

_FRONTMATTER_KEYS = (
    "title",
    "description",
    "id",
    "sidebar_label",
    "pagination_next",
    "pagination_prev",
)

# `import X from '...'` / `import {a, b} from '...'`
_IMPORT_RE = re.compile(r"^\s*import\s+.+?from\s+['\"].+?['\"];?\s*$", re.MULTILINE)
# JSX component tags: <Constant name="core" /> or <Tabs>...</Tabs>
_JSX_TAG_RE = re.compile(r"</?[A-Z][A-Za-z0-9]*(?:\s[^<>]*?)?/?>")
# HTML-ish tags that survive in the docs
_HTML_TAG_RE = re.compile(r"</?(?:div|br|span|img|a|p|table|tr|td|th|thead|tbody)\b[^<>]*/?>")
# Docusaurus admonition markers and JSX comment blocks
_JSX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
# Collapse 3+ newlines
_MULTI_NL_RE = re.compile(r"\n{3,}")
# `:::` admonition fences
_ADMONITION_RE = re.compile(r"^\s*:::{1,2}\s*\w*\s*$", re.MULTILINE)
# HTML entities that show up in the corpus
_ENTITIES = {
    "&mdash;": "—",
    "&ndash;": "-",
    "&nbsp;": " ",
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
}

# Fenced code blocks are preserved but flagged: they matter for dbt (SQL/Jinja).
_CODE_FENCE_RE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)


def _decode_entities(text: str) -> str:
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    return text


def clean_markdown(raw: str) -> str:
    """Strip MDX/JSX scaffolding and normalise whitespace, preserving prose + code."""
    text = _JSX_COMMENT_RE.sub("", raw)
    text = _IMPORT_RE.sub("", text)
    text = _ADMONITION_RE.sub("", text)
    text = _JSX_TAG_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _decode_entities(text)
    # trim trailing spaces, collapse blank runs
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def _slug_from_path(rel_path: str) -> str:
    slug = rel_path.removesuffix(".md").removesuffix(".mdx")
    slug = slug.replace("\\", "/").replace("/", "__")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", slug)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_file(path: Path, root: Path) -> ParsedDocument | None:
    """Parse a single doc file.

    Returns None when the document has no indexable body (e.g. whitespace-only).
    Raises OSError when the file cannot be read, so callers can report a real
    failure rather than silently treating it as empty.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")

    try:
        post = frontmatter.loads(raw)
        fm: dict[str, object] = dict(post.metadata)
        body = post.content
    except Exception:  # malformed frontmatter -> treat whole file as body
        fm = {}
        body = raw

    cleaned = clean_markdown(body)
    if not cleaned.strip():
        return None

    rel_path = path.relative_to(root).as_posix()
    title = str(fm.get("title") or _first_heading(cleaned) or path.stem).strip()
    description = str(fm.get("description") or "").strip()

    metadata = {
        "description": description,
        "frontmatter_id": str(fm.get("id") or ""),
        "sidebar_label": str(fm.get("sidebar_label") or ""),
        "source_format": path.suffix.lstrip("."),
        "word_count": len(cleaned.split()),
        "has_code": bool(_CODE_FENCE_RE.search(cleaned)),
    }
    for key in _FRONTMATTER_KEYS:
        if key in fm and key not in {"title", "description", "id", "sidebar_label"}:
            metadata[f"fm_{key}"] = str(fm[key])

    return ParsedDocument(
        doc_id=_slug_from_path(rel_path),
        path=str(path),
        rel_path=rel_path,
        title=title,
        text=cleaned,
        content_hash=_hash(cleaned),
        metadata=metadata,
    )


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def discover_docs(corpus_dir: Path) -> list[Path]:
    """All markdown/MDX files under the corpus dir, deterministic order."""
    files = [
        p
        for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".mdx"} and "node_modules" not in p.parts
    ]
    return sorted(files)
