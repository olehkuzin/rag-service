"""Pydantic request/response schemas for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    # optional per-request feature overrides (useful for demos + eval)
    hybrid: bool | None = None
    rerank: bool | None = None
    prompt_variant: str | None = Field(default=None, pattern="^(basic|strict)$")


class CitationModel(BaseModel):
    doc: str
    section: str
    chunk_id: str
    score: float
    doc_path: str = ""


class UsageModel(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    # None when the provider has no known pricing (e.g. the offline stub)
    cost_usd: float | None = None


class RetrievedChunkModel(BaseModel):
    chunk_id: str
    doc: str
    section: str
    score: float
    rank: int


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationModel] = []
    found: bool
    confidence: float = 0.0
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    usage: UsageModel = UsageModel()
    trace_id: str = ""
    prompt_variant: str = "strict"
    retrieved: list[RetrievedChunkModel] = []


class IngestRequest(BaseModel):
    force: bool = False
    limit: int | None = Field(default=None, ge=1)


class IngestResponse(BaseModel):
    docs_seen: int
    docs_changed: int
    docs_skipped: int
    docs_empty: int = 0
    docs_failed: int
    docs_removed: int = 0
    chunks_created: int
    chunks_dropped: int
    chunks_written: int
    latency_ms: float
    quality: dict[str, object] | None = None
    errors: list[str] = []


class HealthResponse(BaseModel):
    status: str
    warning: str | None = None
    chunks: int
    embedding_provider: str
    llm_provider: str
    vector_store: str
    hybrid: bool
    rerank: bool
    prompt_variant: str
