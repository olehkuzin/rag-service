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


@dataclass
class BM25Index:
    """In-memory Okapi BM25 over the chunk corpus. Rebuilt after ingestion."""

    k1: float = 1.5
    b: float = 0.75
    _chunks: list[Chunk] = field(default_factory=list)
    _tfs: list[Counter[str]] = field(default_factory=list)
    _lengths: list[int] = field(default_factory=list)
    _df: Counter[str] = field(default_factory=Counter)
    _avgdl: float = 0.0

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tfs = []
        self._lengths = []
        self._df = Counter()
        for chunk in self._chunks:
            tokens = tokenize(f"{chunk.title} {chunk.section} {chunk.text}")
            tf = Counter(tokens)
            self._tfs.append(tf)
            self._lengths.append(len(tokens))
            for term in tf:
                self._df[term] += 1
        self._avgdl = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        q_terms = tokenize(query)
        if not q_terms:
            return []

        scores = []
        for i, tf in enumerate(self._tfs):
            length = self._lengths[i]
            denom_norm = self.k1 * (1 - self.b + self.b * length / (self._avgdl or 1.0))
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f:
                    score += self._idf(term) * (f * (self.k1 + 1)) / (f + denom_norm)
            scores.append(score)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            ScoredChunk(
                chunk=self._chunks[i],
                score=scores[i],
                bm25_score=scores[i],
                rank=rank + 1,
            )
            for rank, i in enumerate(ranked)
            if scores[i] > 0
        ]

    def __len__(self) -> int:
        return len(self._chunks)


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
