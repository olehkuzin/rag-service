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
from typing import Any, Protocol, runtime_checkable

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


class MistralEmbedder:
    """Mistral hosted embeddings (requires MISTRAL_API_KEY).

    `mistral-embed` returns 1024-dimensional vectors. The returned dimension is
    validated against the configured one, because a silent mismatch would
    corrupt the persisted vector index.
    """

    API_URL = "https://api.mistral.ai/v1/embeddings"
    MAX_BATCH = 64

    def __init__(self, model: str = "mistral-embed", dim: int = 1024) -> None:
        self.model = model
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.MAX_BATCH):
            batches.append(self._call(list(texts[start : start + self.MAX_BATCH])))
        return np.vstack(batches)

    def embed_query(self, text: str) -> np.ndarray:
        vector: np.ndarray = self._call([text])[0]
        return vector

    def _call(self, texts: Sequence[str]) -> np.ndarray:
        resp = _post_with_retry(
            self.API_URL,
            headers={"Authorization": f"Bearer {_require_env('MISTRAL_API_KEY')}"},
            json={"input": list(texts), "model": self.model, "encoding_format": "float"},
        )
        rows = sorted(resp.json()["data"], key=lambda d: d["index"])
        vectors = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Mistral returned {len(vectors)} embeddings for {len(texts)} inputs"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"embedding dim mismatch: {self.model} returned {vectors.shape[1]}, "
                f"but RAG_EMBEDDING_DIM is {self.dim}. "
                f"Set RAG_EMBEDDING_DIM={vectors.shape[1]} and re-ingest."
            )
        # L2-normalise so cosine similarity == dot product in the local store
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


def _require_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for this provider")
    return value


def _post_with_retry(url: str, headers: dict[str, str], json: dict[str, object]) -> Any:
    """POST with retries on 429/5xx and transport errors (exponential backoff)."""
    import httpx
    from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

    def _retryable(exc: BaseException) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {429, 500, 502, 503, 504}
        return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def _do() -> Any:
        resp = httpx.post(url, headers=headers, json=json, timeout=60.0)
        if resp.status_code >= 400:
            # Retryable statuses must keep bubbling as HTTPStatusError so the
            # retry policy can act on them; only terminal ones become ProviderError.
            if resp.status_code in {429, 500, 502, 503, 504}:
                resp.raise_for_status()
            raise ProviderError(_describe_http_error(resp, url))
        return resp

    return _do()


class ProviderError(RuntimeError):
    """Provider call failed with an actionable, user-facing message."""


def _describe_http_error(resp: Any, url: str) -> str:
    """Turn a provider HTTP error into something a human can act on."""
    try:
        detail = resp.json().get("detail") or resp.json().get("message")
    except Exception:
        detail = resp.text[:300]

    status = resp.status_code
    if status in {401, 403}:
        hint = (
            "MISTRAL_API_KEY is missing, invalid, or expired.\n"
            "  Check the key at https://console.mistral.ai/ and put the current "
            "value in .env as MISTRAL_API_KEY=..."
        )
    elif status == 402:
        hint = "Mistral account has no remaining credit; check plan/billing."
    elif status == 404:
        hint = "Model or endpoint not found; check RAG_EMBEDDING_MODEL / RAG_LLM_MODEL."
    elif status == 422:
        hint = "Mistral rejected the request payload (often a bad model name or input)."
    else:
        hint = "Unexpected provider error."

    return f"{status} from {url}\n  detail: {detail}\n  {hint}"


def build_embedder(provider: str, model: str, dim: int) -> Embedder:
    match provider:
        case "hash":
            return HashEmbedder(dim=dim)
        case "mistral":
            return MistralEmbedder(model=model, dim=dim)
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


class MistralLLM:
    """Mistral chat completions via REST (requires MISTRAL_API_KEY).

    Uses `response_format=json_object` so strict-mode JSON parsing is reliable
    rather than a best-effort retry.
    """

    API_URL = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    def complete(self, *, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        resp = _post_with_retry(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {_require_env('MISTRAL_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                # Mistral rejects greedy decoding (temperature 0) unless top_p is exactly 1,
                # and the model default is not 1
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        data = resp.json()
        choice = data["choices"][0]
        content = _message_text(choice["message"]["content"]).strip()
        usage = data.get("usage") or {}
        if not content and choice.get("finish_reason") == "length":
            # reasoning models (mistral-large-latest is one) spend max_tokens on
            # thinking first; an empty, truncated reply is a config error, not an abstention
            raise ProviderError(
                f"{self.model} hit max_tokens={max_tokens} before producing an answer "
                f"({usage.get('completion_tokens', '?')} completion tokens, mostly reasoning).\n"
                "  Raise RAG_LLM_MAX_TOKENS (4096 is the default)."
            )
        return content, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def _message_text(content: Any) -> str:
    """Chat message content as text.

    Mistral returns either a plain string or a list of typed chunks
    (`{"type": "text", "text": ...}`, plus e.g. thinking/reference chunks). Only
    text chunks are the answer; anything else would corrupt the JSON payload.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        str(chunk.get("text", ""))
        for chunk in content
        if isinstance(chunk, dict) and chunk.get("type") == "text"
    )


def build_llm(provider: str, model: str) -> LLMClient:
    match provider:
        case "fake":
            return FakeLLM()
        case "mistral":
            return MistralLLM(model=model)
        case _:
            raise ValueError(f"unknown llm provider: {provider}")
