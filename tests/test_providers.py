"""Mistral provider adapters, exercised without network access via httpx mocks.

These verify request shaping, dimension validation, normalisation, usage
plumbing and retry semantics — the parts most likely to break against a real API.
"""

from __future__ import annotations

import numpy as np
import pytest
from app.providers import (
    MistralEmbedder,
    MistralLLM,
    ProviderError,
    _post_with_retry,
    build_embedder,
    build_llm,
)


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        import httpx

        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.mistral.ai")
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )


def _embedding_payload(vectors: list[list[float]], reverse: bool = False) -> dict:
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    if reverse:
        data = data[::-1]  # the API may return out of order; adapters must sort
    return {"data": data, "model": "mistral-embed", "usage": {"total_tokens": 5}}


def test_embedder_requests_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    captured: dict = {}

    def fake_post(url: str, headers: dict, json: dict, **kwargs) -> _FakeResponse:
        captured.update(url=url, headers=headers, json=json)
        return _FakeResponse(_embedding_payload([[1.0, 0.0]]))

    monkeypatch.setattr("httpx.post", fake_post)
    emb = MistralEmbedder(dim=2)
    emb.embed_documents(["hello"])

    assert captured["url"].endswith("/v1/embeddings")
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "mistral-embed"
    assert captured["json"]["input"] == ["hello"]


def test_embedder_sorts_by_index_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    # two unnormalised vectors, returned in reverse order
    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **k: _FakeResponse(_embedding_payload([[3.0, 0.0], [0.0, 4.0]], reverse=True)),
    )
    emb = MistralEmbedder(dim=2)
    out = emb.embed_documents(["first", "second"])

    # sorted back into input order, then L2-normalised
    assert np.allclose(out[0], [1.0, 0.0])
    assert np.allclose(out[1], [0.0, 1.0])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_embedder_rejects_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider/dim mismatch must fail loudly, not corrupt the index."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(
        "httpx.post", lambda *a, **k: _FakeResponse(_embedding_payload([[1.0] * 1024]))
    )
    emb = MistralEmbedder(dim=512)  # configured 512, model returns 1024
    with pytest.raises(ValueError, match="dim mismatch"):
        emb.embed_documents(["x"])


def test_embedder_empty_input_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):
        raise AssertionError("should not call the API for empty input")

    monkeypatch.setattr("httpx.post", boom)
    out = MistralEmbedder(dim=8).embed_documents([])
    assert out.shape == (0, 8)


def test_embedder_batches_large_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    calls: list[int] = []

    def fake_post(url: str, headers: dict, json: dict, **kwargs) -> _FakeResponse:
        calls.append(len(json["input"]))
        return _FakeResponse(_embedding_payload([[1.0, 0.0] for _ in json["input"]]))

    monkeypatch.setattr("httpx.post", fake_post)
    emb = MistralEmbedder(dim=2)
    out = emb.embed_documents([f"doc {i}" for i in range(MistralEmbedder.MAX_BATCH + 5)])

    assert calls == [MistralEmbedder.MAX_BATCH, 5]
    assert out.shape == (MistralEmbedder.MAX_BATCH + 5, 2)


def test_embedder_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        MistralEmbedder(dim=2).embed_query("x")


def test_llm_shapes_request_and_reads_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    captured: dict = {}

    def fake_post(url: str, headers: dict, json: dict, **kwargs) -> _FakeResponse:
        captured.update(url=url, headers=headers, json=json)
        return _FakeResponse(
            {
                "choices": [{"message": {"content": '{"found": false, "answer": ""}'}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
        )

    monkeypatch.setattr("httpx.post", fake_post)
    text, in_tok, out_tok = MistralLLM(model="mistral-large-latest").complete(
        system="sys", user="usr", max_tokens=128
    )

    assert captured["json"]["model"] == "mistral-large-latest"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["json"]["messages"][1] == {"role": "user", "content": "usr"}
    assert text.startswith("{") and "found" in text
    assert (in_tok, out_tok) == (11, 7)


def test_llm_tolerates_null_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **k: _FakeResponse({"choices": [{"message": {"content": None}}], "usage": {}}),
    )
    text, in_tok, out_tok = MistralLLM(model="m").complete(system="s", user="u", max_tokens=8)
    assert text == ""
    assert (in_tok, out_tok) == (0, 0)


def test_post_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """429 is transient: retry, then succeed. Backoff is real, so keep attempts low."""
    attempts: list[int] = []

    def fake_post(url: str, headers: dict, json: dict, **kwargs) -> _FakeResponse:
        attempts.append(1)
        if len(attempts) == 1:
            return _FakeResponse({}, status=429)
        return _FakeResponse({"ok": True})

    monkeypatch.setattr("httpx.post", fake_post)
    resp = _post_with_retry("https://api.mistral.ai/v1/x", {}, {})
    assert resp.json() == {"ok": True}
    assert len(attempts) == 2


def test_post_does_not_retry_on_400(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []

    def fake_post(url: str, headers: dict, json: dict, **kwargs) -> _FakeResponse:
        attempts.append(1)
        return _FakeResponse({}, status=400)

    monkeypatch.setattr("httpx.post", fake_post)

    with pytest.raises(ProviderError):
        _post_with_retry("https://api.mistral.ai/v1/x", {}, {})
    assert len(attempts) == 1  # client error: fail fast, no retries


def test_builders_wire_mistral() -> None:
    assert isinstance(build_embedder("mistral", "mistral-embed", 1024), MistralEmbedder)
    assert isinstance(build_llm("mistral", "mistral-large-latest"), MistralLLM)


# --- actionable errors ----------------------------------------------------


def test_expired_key_produces_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 401 must explain itself, not surface as a raw httpx traceback."""
    monkeypatch.setenv("MISTRAL_API_KEY", "expired-key")

    class Unauthorized(_FakeResponse):
        def __init__(self) -> None:
            super().__init__({"detail": "Your API key expired on 2026-11-21."}, status=401)

    monkeypatch.setattr("httpx.post", lambda *a, **k: Unauthorized())
    with pytest.raises(ProviderError) as info:
        MistralEmbedder(dim=2).embed_query("x")

    msg = str(info.value)
    assert "401" in msg
    assert "expired" in msg
    assert "MISTRAL_API_KEY" in msg  # names the thing to fix
    assert "console.mistral.ai" in msg  # says where


def test_402_mentions_billing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setattr("httpx.post", lambda *a, **k: _FakeResponse({}, status=402))
    with pytest.raises(ProviderError, match="credit"):
        MistralEmbedder(dim=2).embed_query("x")


def test_404_points_at_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    monkeypatch.setattr("httpx.post", lambda *a, **k: _FakeResponse({}, status=404))
    with pytest.raises(ProviderError, match=r"RAG_(LLM|EMBEDDING)_MODEL"):
        MistralLLM(model="nope").complete(system="s", user="u", max_tokens=8)


def test_provider_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """401 must fail immediately; retrying a bad key just wastes time."""
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    calls: list[int] = []

    def fake_post(*a, **k):
        calls.append(1)
        return _FakeResponse({}, status=401)

    monkeypatch.setattr("httpx.post", fake_post)
    with pytest.raises(ProviderError):
        MistralEmbedder(dim=2).embed_query("x")
    assert len(calls) == 1
