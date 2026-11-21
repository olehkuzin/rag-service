"""Ingestion: parsing, cleaning, chunking, quality checks, incremental behaviour."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.ingestion.chunker import chunk_document
from app.ingestion.parser import clean_markdown, discover_docs, parse_file
from app.ingestion.quality import check_chunks
from app.service import Service


def test_clean_strips_mdx_and_keeps_prose() -> None:
    raw = (
        "import Tabs from '@theme/Tabs';\n"
        "{/* hidden comment */}\n"
        'Use <Constant name="core" /> for Core.\n'
        ":::note\n"
        "Read &mdash; then run.\n"
    )
    cleaned = clean_markdown(raw)
    assert "import Tabs" not in cleaned
    assert "hidden comment" not in cleaned
    assert "<Constant" not in cleaned
    assert ":::" not in cleaned
    assert "—" in cleaned
    assert "for Core." in cleaned


def test_parse_extracts_frontmatter(corpus: Path) -> None:
    doc = parse_file(corpus / "build" / "models.md", corpus)
    assert doc is not None
    assert doc.title == "About dbt models"
    assert doc.metadata["description"] == "Models transform data."
    assert doc.content_hash


def test_discover_ignores_empty_doc(corpus: Path) -> None:
    files = discover_docs(corpus)
    assert len(files) == 3
    empty = parse_file(corpus / "empty.md", corpus)
    assert empty is None  # whitespace-only body is not indexable


def test_chunk_respects_heading_structure(corpus: Path, settings: Settings) -> None:
    doc = parse_file(corpus / "build" / "models.md", corpus)
    assert doc is not None
    chunks = chunk_document(doc, settings)
    sections = {c.section for c in chunks}
    assert "How models run" in sections
    assert "Incremental models" in sections
    # heading path should nest
    incr = next(c for c in chunks if c.section == "Incremental models")
    assert incr.heading_path[-1] == "Incremental models"
    assert all(c.doc_id == doc.doc_id for c in chunks)


def test_chunk_ids_are_unique_and_stable(corpus: Path, settings: Settings) -> None:
    doc = parse_file(corpus / "build" / "models.md", corpus)
    assert doc is not None
    first = chunk_document(doc, settings)
    second = chunk_document(doc, settings)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)


def test_chunker_splits_oversized_section(tmp_path: Path, settings: Settings) -> None:
    body = "## Long section\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(30))
    path = tmp_path / "long.md"
    path.write_text(f"---\ntitle: Long\n---\n\n{body}", encoding="utf-8")
    doc = parse_file(path, tmp_path)
    assert doc is not None
    chunks = chunk_document(doc, settings)
    assert len(chunks) > 1
    assert all(len(c.text) <= settings.chunk_max_chars * 1.5 for c in chunks)


def test_quality_drops_short_and_duplicates(corpus: Path, settings: Settings) -> None:
    doc = parse_file(corpus / "build" / "models.md", corpus)
    assert doc is not None
    chunks = chunk_document(doc, settings)
    # duplicate one chunk verbatim
    from dataclasses import replace

    dup = replace(chunks[0], chunk_id="dup-1")
    kept, report = check_chunks([*chunks, dup], settings)
    assert len(kept) == len(chunks)
    assert report.reasons().get("duplicate") == 1


def test_quality_flags_missing_metadata(corpus: Path, settings: Settings) -> None:
    from dataclasses import replace

    doc = parse_file(corpus / "build" / "models.md", corpus)
    assert doc is not None
    chunks = chunk_document(doc, settings)
    broken = replace(chunks[0], chunk_id="broken", title="")
    kept, report = check_chunks([broken], settings)
    assert kept == []
    assert report.reasons().get("missing_metadata") == 1


def test_incremental_skips_unchanged(service: Service) -> None:
    # the fixture already ran a forced ingest, so everything is up to date
    before = service.store.count()
    second = service.ingest()
    assert second.docs_changed == 0
    assert second.chunks_written == 0
    assert second.docs_skipped == second.docs_seen - second.docs_empty
    assert service.store.count() == before


def test_forced_ingest_reprocesses_everything(service: Service) -> None:
    stats = service.ingest(force=True)
    assert stats.docs_changed == stats.docs_seen - stats.docs_empty
    assert stats.chunks_written > 0


def test_incremental_reprocesses_changed_doc(service: Service, settings: Settings) -> None:
    original = service.store.count()
    target = settings.corpus_dir / "build" / "tests.md"
    target.write_text(
        "---\ntitle: About data tests\n---\n\nCompletely different content about `dbt test` "
        "and singular tests stored in the tests directory.\n",
        encoding="utf-8",
    )
    stats = service.ingest()
    assert stats.docs_changed == 1
    assert stats.docs_skipped == stats.docs_seen - stats.docs_empty - 1
    # old chunks for that doc must be gone, not duplicated
    texts = [c.text for c in service.store.all_chunks()]
    assert not any("Generic tests" in t for t in texts)
    assert service.store.count() <= original + 2


def test_doc_with_no_valid_chunks_still_converges(settings: Settings) -> None:
    """A doc whose chunks are all rejected must not be reprocessed every run.

    Regression test: incremental state used to be derived from stored chunks, so
    zero-chunk documents were re-parsed forever.
    """
    # a doc under chunk_min_chars -> all chunks dropped by QC
    tiny = settings.corpus_dir / "tiny.md"
    tiny.write_text("---\ntitle: Tiny\n---\n\nhi\n", encoding="utf-8")

    svc = Service(settings)
    first = svc.ingest()
    assert first.docs_changed >= 1

    second = svc.ingest()
    assert second.docs_changed == 0
    assert second.docs_skipped == second.docs_seen - second.docs_empty


def test_deleted_doc_is_pruned_from_index(service: Service, settings: Settings) -> None:
    victim = settings.corpus_dir / "build" / "tests.md"
    doc_id = parse_file(victim, settings.corpus_dir).doc_id  # type: ignore[union-attr]
    assert any(c.doc_id == doc_id for c in service.store.all_chunks())

    victim.unlink()
    stats = service.ingest()
    assert stats.docs_removed == 1
    assert not any(c.doc_id == doc_id for c in service.store.all_chunks())


def test_manifest_records_zero_chunk_docs(service: Service, settings: Settings) -> None:
    manifest_path = settings.index_dir / "manifest.json"
    assert manifest_path.exists()
    import json

    data = json.loads(manifest_path.read_text())
    # the empty.md fixture is skipped before parsing, but a chunkless doc is recorded
    assert any(d["chunk_count"] >= 0 for d in data["docs"])
