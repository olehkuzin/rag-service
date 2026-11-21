"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.config import Settings
from app.service import Service


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Small synthetic corpus with dbt-like structure and MDX noise."""
    root = tmp_path / "corpus"
    (root / "build").mkdir(parents=True)
    (root / "build" / "models.md").write_text(
        """---
title: "About dbt models"
description: "Models transform data."
id: "models"
---

Models are where developers spend most of their time. A model is a single file
containing a final `select` statement. <Constant name="core" /> supports them.

## How models run

When you execute `dbt run`, dbt builds the model in your warehouse.

### Incremental models

Incremental models only transform new rows since the last run.
""",
        encoding="utf-8",
    )
    (root / "build" / "tests.md").write_text(
        """---
title: "About data tests"
---

Data tests assert that a column is unique or not null.

## Generic tests

Add `unique` and `not_null` to a column in the YAML file.
""",
        encoding="utf-8",
    )
    (root / "empty.md").write_text("---\ntitle: Empty\n---\n\n   \n", encoding="utf-8")
    return root


@pytest.fixture()
def settings(tmp_path: Path, corpus: Path) -> Settings:
    return Settings(
        corpus_dir=corpus,
        index_dir=tmp_path / "index",
        embedding_provider="hash",
        embedding_dim=64,
        llm_provider="fake",
        chunk_max_chars=400,
        chunk_min_chars=20,
        chunk_overlap_chars=50,
    )


@pytest.fixture()
def service(settings: Settings) -> Service:
    svc = Service(settings)
    svc.ingest(force=True)
    return svc
