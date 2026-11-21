"""Application settings, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every provider is swappable via env vars."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="RAG_"
    )

    # --- corpus / storage ---
    corpus_dir: Path = Path("data/corpus")
    index_dir: Path = Path("data/index")
    vector_store: str = "local"  # local | qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "dbt_docs"

    # --- embeddings ---
    embedding_provider: str = "hash"  # hash | voyage | openai
    embedding_model: str = "voyage-3-lite"
    embedding_dim: int = 512

    # --- LLM ---
    llm_provider: str = "fake"  # fake | anthropic
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 1024

    # --- retrieval ---
    chunk_max_chars: int = 1800
    chunk_min_chars: int = 120
    chunk_overlap_chars: int = 200
    chunk_strategy: str = "structure"  # structure | fixed
    default_top_k: int = 5
    vector_candidates: int = 20
    bm25_candidates: int = 20
    rerank_candidates: int = 20
    rrf_k: int = 60

    # --- features (used by eval variants) ---
    enable_hybrid: bool = True
    enable_rerank: bool = True
    prompt_variant: str = "strict"  # basic | strict

    # --- observability ---
    langfuse_enabled: bool = False

    # --- api ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
