"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.service import get_service


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    get_service()  # warm the pipeline at startup
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="dbt docs RAG service",
        version="0.1.0",
        description="Cited answers over the dbt documentation corpus.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "rag-service", "docs": "/docs", "health": "/health"}

    return app


app = create_app()
