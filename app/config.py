"""Application settings, loaded from environment / .env."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(".env")


def load_env_file(path: Path = _ENV_FILE) -> None:
    """Export `.env` into os.environ.

    Provider keys (MISTRAL_API_KEY) are read via os.environ, not through
    Settings, so a key placed only in `.env` would otherwise be invisible and
    every API call would fail claiming the key is missing. Real environment
    variables win over `.env`, matching pydantic-settings precedence.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split(" #", 1)[0].strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


class Settings(BaseSettings):
    """Runtime configuration. Every provider is swappable via env vars."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="RAG_"
    )

    # --- corpus / storage ---
    corpus_dir: Path = Path("data/corpus")
    index_dir: Path = Path("data/index")
    vector_store: str = "local"  # local (default) | qdrant (optional dep)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "dbt_docs"

    # --- embeddings ---
    embedding_provider: str = "hash"  # hash | mistral
    embedding_model: str = "mistral-embed"
    embedding_dim: int = 512

    # --- LLM ---
    llm_provider: str = "fake"  # fake | mistral
    llm_model: str = "mistral-large-latest"
    # reasoning models count thinking tokens against this budget, so keep it generous
    llm_max_tokens: int = 4096

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
    load_env_file()
    return Settings()
