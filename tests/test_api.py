"""API contract tests (httpx + FastAPI app, fake providers, no network)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from app import service as service_module
from app.config import Settings
from app.main import create_app
from app.service import reset_service
from fastapi.testclient import TestClient


@pytest.fixture()
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    reset_service()
    monkeypatch.setattr(service_module, "Settings", lambda: settings)
    monkeypatch.setattr("app.service.get_settings", lambda: settings)
    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_service()


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["embedding_provider"] == "hash"


def test_ask_before_ingest_returns_503(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "what is a model"})
    assert resp.status_code == 503


def test_ingest_then_ask(client: TestClient) -> None:
    ingest = client.post("/ingest", json={"force": True})
    assert ingest.status_code == 200
    stats = ingest.json()
    assert stats["chunks_written"] > 0
    assert stats["docs_failed"] == 0

    ask = client.post("/ask", json={"question": "how do models run", "top_k": 3})
    assert ask.status_code == 200
    body = ask.json()
    assert body["found"] is True
    assert body["citations"]
    assert body["trace_id"]
    assert body["latency_ms"] >= 0
    assert body["usage"]["input_tokens"] >= 0
    assert 1 <= len(body["retrieved"]) <= 3


def test_ask_validates_top_k(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "x", "top_k": 99})
    assert resp.status_code == 422


def test_ask_rejects_empty_question(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_per_request_variant_override(client: TestClient) -> None:
    client.post("/ingest", json={"force": True})
    resp = client.post(
        "/ask",
        json={
            "question": "incremental models",
            "hybrid": False,
            "rerank": False,
            "prompt_variant": "basic",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["prompt_variant"] == "basic"


def test_traces_endpoint_records_requests(client: TestClient) -> None:
    client.post("/ingest", json={"force": True})
    client.post("/ask", json={"question": "models"})
    resp = client.get("/traces?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stats"]["count"] >= 1
    assert body["recent"][0]["question"] == "models"
    assert "retrieved" in body["recent"][0]


def test_incremental_ingest_via_api(client: TestClient) -> None:
    first = client.post("/ingest", json={"force": False}).json()
    second = client.post("/ingest", json={"force": False}).json()
    assert first["docs_changed"] == first["docs_seen"] - first["docs_empty"]
    assert second["docs_skipped"] == second["docs_seen"] - second["docs_empty"]


def test_prompt_variant_override_does_not_leak_between_requests(client: TestClient) -> None:
    """Per-request overrides must not mutate the shared service singleton."""
    client.post("/ingest", json={"force": True})
    first = client.post("/ask", json={"question": "models", "prompt_variant": "basic"})
    assert first.json()["prompt_variant"] == "basic"

    # a request without the override must fall back to the configured default
    second = client.post("/ask", json={"question": "models"})
    assert second.json()["prompt_variant"] == "strict"
