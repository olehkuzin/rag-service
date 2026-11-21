"""Lexical retrieval (BM25) and rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.models import Chunk, ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True, slots=True)
class _BM25State:
    """Everything a search needs, built once and never mutated after publishing."""

    chunks: tuple[Chunk, ...] = ()
    tfs: tuple[Counter[str], ...] = ()
    lengths: tuple[int, ...] = ()
    df: Counter[str] = field(default_factory=Counter)
    avgdl: float = 0.0


@dataclass
class BM25Index:
    """In-memory Okapi BM25 over the chunk corpus. Rebuilt after ingestion.

    `build` computes a new `_BM25State` off to the side and publishes it with one
    assignment, so a concurrent `search` always sees a complete, consistent index.
    """

    k1: float = 1.5
    b: float = 0.75
    _state: _BM25State = field(default_factory=_BM25State)

    def build(self, chunks: Sequence[Chunk]) -> None:
        tfs: list[Counter[str]] = []
        lengths: list[int] = []
        df: Counter[str] = Counter()
        for chunk in chunks:
            tokens = tokenize(f"{chunk.title} {chunk.section} {chunk.text}")
            tf = Counter(tokens)
            tfs.append(tf)
            lengths.append(len(tokens))
            df.update(tf.keys())
        self._state = _BM25State(
            chunks=tuple(chunks),
            tfs=tuple(tfs),
            lengths=tuple(lengths),
            df=df,
            avgdl=sum(lengths) / len(lengths) if lengths else 0.0,
        )

    @staticmethod
    def _idf(state: _BM25State, term: str) -> float:
        n = len(state.chunks)
        df = state.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        state = self._state  # one consistent view for the whole search
        if not state.chunks:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []

        idf = {term: self._idf(state, term) for term in set(q_terms)}
        scores = []
        for tf, length in zip(state.tfs, state.lengths, strict=True):
            denom_norm = self.k1 * (1 - self.b + self.b * length / (state.avgdl or 1.0))
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f:
                    score += idf[term] * (f * (self.k1 + 1)) / (f + denom_norm)
            scores.append(score)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            ScoredChunk(
                chunk=state.chunks[i],
                score=scores[i],
                bm25_score=scores[i],
                rank=rank + 1,
            )
            for rank, i in enumerate(ranked)
            if scores[i] > 0
        ]

    def __len__(self) -> int:
        return len(self._state.chunks)


def reciprocal_rank_fusion(
    runs: Sequence[Sequence[ScoredChunk]], k: int = 60, top_k: int | None = None
) -> list[ScoredChunk]:
    """Merge ranked lists with RRF: score = sum(1 / (k + rank)) per document."""
    fused: dict[str, ScoredChunk] = {}
    totals: dict[str, float] = {}

    for run in runs:
        for rank, scored in enumerate(run, start=1):
            cid = scored.chunk_id
            totals[cid] = totals.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in fused:
                fused[cid] = ScoredChunk(
                    chunk=scored.chunk,
                    score=0.0,
                    vector_score=scored.vector_score,
                    bm25_score=scored.bm25_score,
                )
            else:
                # preserve whichever signal each run contributed
                if scored.vector_score is not None:
                    fused[cid].vector_score = scored.vector_score
                if scored.bm25_score is not None:
                    fused[cid].bm25_score = scored.bm25_score

    ordered = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]
    results = []
    for rank, (cid, score) in enumerate(ordered, start=1):
        item = fused[cid]
        item.score = score
        item.rank = rank
        results.append(item)
    return results
