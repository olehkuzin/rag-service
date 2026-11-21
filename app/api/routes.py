"""FastAPI routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import (
    AskRequest,
    AskResponse,
    CitationModel,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetrievedChunkModel,
    UsageModel,
)
from app.service import Service, get_service

router = APIRouter()


def _service() -> Service:
    return get_service()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse.model_validate(_service().health())


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    service = _service()
    if service.store.count() == 0:
        raise HTTPException(status_code=503, detail="index is empty; run POST /ingest first")

    config = None
    if request.hybrid is not None or request.rerank is not None:
        from app.retrieval.pipeline import RetrievalConfig

        config = RetrievalConfig.from_settings(service.settings, request.top_k)
        if request.hybrid is not None:
            config.enable_hybrid = request.hybrid
        if request.rerank is not None:
            config.enable_rerank = request.rerank

    try:
        # Retrieval and generation are synchronous (httpx, numpy) and would block
        # the event loop for every other request. Offload to a worker thread.
        result, trace = await run_in_threadpool(
            service.ask,
            request.question,
            top_k=request.top_k,
            config=config,
            # pass the variant per call; never mutate shared singleton state
            prompt_variant=request.prompt_variant,
        )
    except Exception as exc:  # surface provider failures as 502, not 500 traces
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc

    retrieved: list[RetrievedChunkModel] = []
    for item in trace.retrieved:
        retrieved.append(
            RetrievedChunkModel(
                chunk_id=item["chunk_id"],
                doc=item["doc"],
                section=item["section"],
                score=item["score"],
                rank=len(retrieved) + 1,
            )
        )

    return AskResponse(
        answer=result.answer,
        found=result.found,
        citations=[
            CitationModel(
                doc=c.doc,
                section=c.section,
                chunk_id=c.chunk_id,
                score=c.score,
                doc_path=c.doc_path,
            )
            for c in result.citations
        ],
        confidence=result.confidence,
        latency_ms=round(trace.latency_ms, 1),
        retrieval_latency_ms=round(trace.retrieval_latency_ms, 1),
        usage=UsageModel(
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            cost_usd=None if trace.cost_usd is None else round(trace.cost_usd, 6),
        ),
        trace_id=trace.trace_id,
        prompt_variant=trace.prompt_variant,
        retrieved=retrieved,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    try:
        stats = await run_in_threadpool(_service().ingest, force=request.force, limit=request.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}") from exc
    payload = stats.summary()
    return IngestResponse(**payload)  # type: ignore[arg-type]


@router.get("/traces")
async def traces(limit: int = 20) -> dict[str, object]:
    service = _service()
    return {
        "stats": service.tracer.stats(),
        "recent": [t.to_dict() for t in service.tracer.recent(limit)],
    }
