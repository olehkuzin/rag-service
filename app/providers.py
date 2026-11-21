"""Provider abstractions: embeddings, LLM generation, reranking.

Every external dependency sits behind a Protocol so tests, eval variants and
model comparisons can swap implementations without touching call sites. Each
provider ships a deterministic local fallback that requires no API key.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic bag-of-token-hashes embedder. No API key, fully reproducible.

    Not semantically strong, but it makes the whole pipeline (and the eval
    harness) runnable and testable offline. Swap for a hosted model to get real
    retrieval quality.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class VoyageEmbedder:
    """Voyage hosted embeddings (requires VOYAGE_API_KEY)."""

    def __init__(self, model: str = "voyage-3-lite", dim: int = 512) -> None:
        self.model = model
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._call(texts, input_type="document")

    def embed_query(self, text: str) -> np.ndarray:
        vector: np.ndarray = self._call([text], input_type="query")[0]
        return vector

    def _call(self, texts: Sequence[str], input_type: str) -> np.ndarray:
        import httpx

        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {_require_env('VOYAGE_API_KEY')}"},
            json={"input": list(texts), "model": self.model, "input_type": input_type},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vectors: np.ndarray = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        return vectors


class OpenAIEmbedder:
    """OpenAI hosted embeddings (requires OPENAI_API_KEY)."""

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536) -> None:
        self.model = model
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._call(texts)

    def embed_query(self, text: str) -> np.ndarray:
        vector: np.ndarray = self._call([text])[0]
        return vector

    def _call(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {_require_env('OPENAI_API_KEY')}"},
            json={"input": list(texts), "model": self.model},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vectors: np.ndarray = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        return vectors


def _require_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for this provider")
    return value


def build_embedder(provider: str, model: str, dim: int) -> Embedder:
    match provider:
        case "hash":
            return HashEmbedder(dim=dim)
        case "voyage":
            return VoyageEmbedder(model=model, dim=dim)
        case "openai":
            return OpenAIEmbedder(model=model, dim=dim)
        case _:
            raise ValueError(f"unknown embedding provider: {provider}")


# --------------------------------------------------------------------------
# Reranker
# --------------------------------------------------------------------------


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, passages: Sequence[str]) -> list[float]: ...


class LexicalReranker:
    """Offline cross-encoder stand-in: IDF-weighted query-term overlap.

    Cheap, deterministic, and measurably better than raw vector order — enough
    to demonstrate the reranker stage and let the eval compare with/without it.
    """

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        q_terms = set(_TOKEN_RE.findall(query.lower()))
        if not q_terms:
            return [0.0] * len(passages)

        tokenised = [_TOKEN_RE.findall(p.lower()) for p in passages]
        n = len(tokenised)
        df = {t: sum(1 for doc in tokenised if t in set(doc)) for t in q_terms}
        scores: list[float] = []
        for doc in tokenised:
            doc_set = set(doc)
            matched = q_terms & doc_set
            idf = sum(math.log((n + 1) / (df[t] + 0.5)) for t in matched)
            coverage = len(matched) / len(q_terms)
            length_penalty = 1.0 / (1.0 + len(doc) / 400.0)
            scores.append(idf * coverage * length_penalty)
        return scores


class CohereReranker:
    """Cohere hosted rerank (requires COHERE_API_KEY)."""

    def __init__(self, model: str = "rerank-v3.5") -> None:
        self.model = model

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        import httpx

        resp = httpx.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {_require_env('COHERE_API_KEY')}"},
            json={
                "model": self.model,
                "query": query,
                "documents": list(passages),
                "top_n": len(passages),
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        results = sorted(resp.json()["results"], key=lambda r: r["index"])
        return [float(r["relevance_score"]) for r in results]


def build_reranker(provider: str) -> Reranker:
    match provider:
        case "lexical" | "local":
            return LexicalReranker()
        case "cohere":
            return CohereReranker()
        case _:
            raise ValueError(f"unknown reranker provider: {provider}")


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Returns raw text; structured parsing happens in the generation layer."""

    def complete(self, *, system: str, user: str, max_tokens: int) -> tuple[str, int, int]: ...


class FakeLLM:
    """Deterministic offline LLM used by tests and for key-free smoke runs.

    It does not fabricate answers: given context it echoes a short extractive
    summary and reports `found` based on whether context was supplied. This
    keeps eval plumbing honest while never pretending to be a real model.
    """

    def __init__(self, answer: str | None = None) -> None:
        self._answer = answer

    def complete(self, *, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        del max_tokens  # the offline stub ignores the token budget
        if self._answer is not None:
            return self._answer, len(system + user) // 4, len(self._answer) // 4

        context_marker = "CONTEXT:"
        context = user.split(context_marker, 1)[1] if context_marker in user else ""
        # collect the c1..cN keys the generator put in the prompt
        keys = re.findall(r"^\[(c\d+)\]", context, re.MULTILINE)
        if not keys:
            answer = '{"found": false, "answer": "", "citations": [], "confidence": 0.0}'
            return answer, len(system + user) // 4, len(answer) // 4

        # extractive stand-in: first prose sentence of the top chunk, citing it
        first = context.split("[c1]", 1)[1]
        lines = [ln for ln in first.splitlines() if ln.strip()]
        # drop the "doc: ... | section: ..." header line the prompt builder adds
        if lines and lines[0].strip().startswith("doc: "):
            lines = lines[1:]
        body = " ".join(" ".join(lines).split())[:300].replace('"', "'")
        if not body:
            answer = '{"found": false, "answer": "", "citations": [], "confidence": 0.0}'
            return answer, len(system + user) // 4, len(answer) // 4
        payload = {
            "found": True,
            "answer": body,
            "citations": [keys[0]],
            "confidence": 0.5,
        }
        answer = json.dumps(payload)
        return answer, len(system + user) // 4, len(answer) // 4


class AnthropicLLM:
    """Anthropic Claude via the official SDK."""

    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, *, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        import anthropic

        client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text")
        usage = msg.usage
        return text, usage.input_tokens, usage.output_tokens


def build_llm(provider: str, model: str) -> LLMClient:
    match provider:
        case "fake":
            return FakeLLM()
        case "anthropic":
            return AnthropicLLM(model=model)
        case _:
            raise ValueError(f"unknown llm provider: {provider}")
