"""Prompts and structured generation with citations + abstention.

Two prompt variants are provided so the eval can compare them:
  * `basic`  — answer from context, no strict citation discipline.
  * `strict` — must cite chunk ids and must abstain when context is insufficient.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.models import ScoredChunk
from app.providers import LLMClient

BASIC_SYSTEM = """You are a helpful assistant answering questions about dbt.
Answer using the provided context. Be concise."""

STRICT_SYSTEM = """You are a documentation assistant for dbt (data build tool).

Rules:
1. Answer ONLY using the numbered context chunks provided.
2. Every factual claim must cite the chunk id(s) it came from, like [c3].
3. If the context does not contain enough information, set "found" to false, leave
   "answer" empty, and do not guess. Abstaining is correct when evidence is missing.
4. Prefer precise, specific answers over general advice.

Respond with JSON only, matching this schema:
{"found": bool, "answer": string, "citations": [string], "confidence": number}
where citations are chunk ids such as "c1"."""


@dataclass(slots=True)
class Citation:
    doc: str
    section: str
    chunk_id: str
    score: float
    doc_path: str = ""


@dataclass(slots=True)
class GenerationResult:
    answer: str
    found: bool
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    raw: str = ""
    prompt_variant: str = "strict"
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0
    parse_error: str | None = None


def build_context(chunks: list[ScoredChunk]) -> tuple[str, dict[str, ScoredChunk]]:
    """Number chunks c1..cN and return the context block plus id lookup."""
    lines: list[str] = []
    lookup: dict[str, ScoredChunk] = {}
    for i, scored in enumerate(chunks, start=1):
        key = f"c{i}"
        lookup[key] = scored
        lines.append(
            f"[{key}] doc: {scored.chunk.title} | section: {scored.chunk.section}\n"
            f"{scored.chunk.text}"
        )
    return "\n\n".join(lines), lookup


def build_prompt(
    question: str, chunks: list[ScoredChunk], variant: str
) -> tuple[str, str, dict[str, ScoredChunk]]:
    system = STRICT_SYSTEM if variant == "strict" else BASIC_SYSTEM
    context, lookup = build_context(chunks)
    if not context:
        context = "(no context retrieved)"
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    if variant != "strict":
        user += "\n\nAnswer the question in plain text."
    return system, user, lookup


def parse_response(
    raw: str, lookup: dict[str, ScoredChunk], variant: str
) -> tuple[GenerationResult, str | None]:
    """Parse model output into structured form. Never raises on bad JSON."""
    text = raw.strip()
    # tolerate fenced code blocks around JSON
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    data: dict[str, Any] | None = None
    if text.startswith("{") or fence:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None

    if data is None:
        if variant == "strict":
            # unparseable strict output is treated as an abstention, not a crash
            return (
                GenerationResult(
                    answer="",
                    found=False,
                    raw=raw,
                    prompt_variant=variant,
                    parse_error="unparseable_json",
                ),
                "unparseable_json",
            )
        return (
            GenerationResult(answer=raw.strip(), found=True, raw=raw, prompt_variant=variant),
            None,
        )

    found = bool(data.get("found", True))
    answer = str(data.get("answer") or "").strip()
    confidence = _clamp(data.get("confidence", 0.0))
    cited_ids = [str(c) for c in data.get("citations") or []]

    citations: list[Citation] = []
    for key in cited_ids:
        scored = lookup.get(key)
        if scored is None:
            continue
        citations.append(
            Citation(
                doc=scored.chunk.title,
                section=scored.chunk.section,
                chunk_id=scored.chunk.chunk_id,
                score=round(scored.score, 4),
                doc_path=scored.chunk.doc_path,
            )
        )

    # strict mode: an answer with no resolvable citation is not trustworthy
    if variant == "strict" and found and not citations:
        return (
            GenerationResult(
                answer=answer,
                found=False,
                citations=[],
                confidence=0.0,
                raw=raw,
                prompt_variant=variant,
                parse_error="citation_missing",
            ),
            "citation_missing",
        )

    if not answer and found:
        found = False

    return (
        GenerationResult(
            answer=answer,
            found=found,
            citations=citations,
            confidence=confidence,
            raw=raw,
            prompt_variant=variant,
        ),
        None,
    )


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


class Generator:
    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self.llm = llm
        self.settings = settings

    def generate(
        self,
        question: str,
        chunks: list[ScoredChunk],
        variant: str | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        variant = variant or self.settings.prompt_variant
        system, user, lookup = build_prompt(question, chunks, variant)
        raw, in_tok, out_tok = self.llm.complete(
            system=system, user=user, max_tokens=max_tokens or self.settings.llm_max_tokens
        )
        result, _ = parse_response(raw, lookup, variant)
        result.usage_input_tokens = in_tok
        result.usage_output_tokens = out_tok
        return result
