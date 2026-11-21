"""Tests for the evaluation metrics.

These functions produce every headline number in the README, so silent errors
here would misreport results. All are pure, so they are cheap to cover.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.observability.tracing import estimate_cost
from eval.run_eval import (
    EvalQuestion,
    QuestionResult,
    Variant,
    _aggregate,
    _mean_cost,
    hit_at_k,
    keyword_coverage,
    load_dataset,
    ndcg_at_k,
    normalize_tokens,
    recall_at_k,
    reciprocal_rank,
    results_table,
    token_f1,
)

GOLD = ["docs/build/a.md", "docs/build/b.md"]


# --- retrieval metrics ----------------------------------------------------


def test_recall_at_k_counts_covered_gold_docs() -> None:
    assert recall_at_k(GOLD, GOLD, k=5) == 1.0
    assert recall_at_k(["docs/build/a.md"], GOLD, k=5) == 0.5
    assert recall_at_k([], GOLD, k=5) == 0.0


def test_recall_at_k_only_inspects_top_k() -> None:
    retrieved = ["x.md", "y.md", "docs/build/a.md"]
    assert recall_at_k(retrieved, GOLD, k=2) == 0.0
    assert recall_at_k(retrieved, GOLD, k=3) == 0.5


def test_recall_without_gold_docs_is_zero_not_error() -> None:
    assert recall_at_k(["a.md"], [], k=5) == 0.0


def test_hit_at_k() -> None:
    assert hit_at_k(["x.md", "docs/build/b.md"], GOLD, k=2) == 1.0
    assert hit_at_k(["x.md"], GOLD, k=2) == 0.0


def test_reciprocal_rank_uses_first_gold_hit() -> None:
    assert reciprocal_rank(["x.md", "docs/build/a.md"], GOLD) == 0.5
    assert reciprocal_rank(["docs/build/a.md"], GOLD) == 1.0
    assert reciprocal_rank(["x.md"], GOLD) == 0.0


def test_ndcg_is_one_for_perfect_ranking() -> None:
    assert ndcg_at_k(GOLD, GOLD, k=2) == pytest.approx(1.0)


def test_ndcg_penalises_late_hits() -> None:
    early = ndcg_at_k(["docs/build/a.md", "x.md"], GOLD, k=2)
    late = ndcg_at_k(["x.md", "docs/build/a.md"], GOLD, k=2)
    assert early > late > 0


def test_ndcg_bounded() -> None:
    value = ndcg_at_k(["x.md", "y.md"], GOLD, k=2)
    assert 0.0 <= value <= 1.0


# --- generation metrics ---------------------------------------------------


def test_normalize_tokens_strips_stopwords_and_punctuation() -> None:
    assert normalize_tokens("What is a dbt model?") == ["dbt", "model"]


def test_token_f1_perfect_match() -> None:
    assert token_f1("incremental models add rows", "incremental models add rows") == 1.0


def test_token_f1_disjoint_is_zero() -> None:
    assert token_f1("cats", "dogs") == 0.0


def test_token_f1_is_set_based_and_symmetric_bounded() -> None:
    # partial overlap, order-insensitive
    a = token_f1("incremental models", "incremental models add rows")
    b = token_f1("incremental models add rows", "incremental models")
    assert a == b
    assert 0.0 < a < 1.0


def test_token_f1_empty_side_is_zero() -> None:
    assert token_f1("", "something") == 0.0
    assert token_f1("something", "") == 0.0


def test_keyword_coverage_is_fraction_found() -> None:
    assert keyword_coverage("table view incremental", ["table", "view"]) == 1.0
    assert keyword_coverage("table only", ["table", "view"]) == 0.5
    assert keyword_coverage("nothing", ["table"]) == 0.0


def test_keyword_coverage_no_keywords_is_zero() -> None:
    assert keyword_coverage("text", []) == 0.0


def test_keyword_coverage_is_case_insensitive() -> None:
    assert keyword_coverage("Incremental MODELS", ["incremental"]) == 1.0


# --- aggregation ----------------------------------------------------------


def _result(**kwargs: object) -> QuestionResult:
    base: dict[str, object] = {
        "question_id": "q",
        "type": "lookup",
        "answerable": True,
        "found": True,
        "recall": 1.0,
        "hit": 1.0,
        "rr": 1.0,
        "ndcg": 1.0,
        "f1": 1.0,
        "keywords": 1.0,
        "citation_acc": 1.0,
        "latency_ms": 10.0,
        "cost_usd": None,
    }
    base.update(kwargs)
    return QuestionResult(**base)  # type: ignore[arg-type]


def test_aggregate_separates_answerable_from_unanswerable() -> None:
    variant = Variant("v", "d", retrieval=None, prompt_variant="strict")  # type: ignore[arg-type]
    results = [
        _result(question_id="a1"),
        _result(question_id="a2", recall=0.0, found=False),
        _result(question_id="u1", type="unanswerable", answerable=False, found=False),
        _result(question_id="u2", type="unanswerable", answerable=False, found=True),
    ]
    report = _aggregate(variant, results, k=5)

    # retrieval averages only over answerable questions
    assert report.metrics["recall@5"] == pytest.approx(0.5)
    assert report.metrics["answer_rate"] == pytest.approx(0.5)
    # abstention only over unanswerable ones
    assert report.metrics["abstention_rate_unanswerable"] == pytest.approx(0.5)
    assert report.per_type["unanswerable"]["n"] == 2
    assert report.per_type["unanswerable"]["abstention_rate"] == pytest.approx(0.5)


def test_aggregate_handles_empty_result_set() -> None:
    variant = Variant("v", "d", retrieval=None, prompt_variant="strict")  # type: ignore[arg-type]
    report = _aggregate(variant, [], k=5)
    assert report.n_questions == 0
    assert report.metrics["recall@5"] == 0.0
    assert report.metrics["cost_per_query_usd"] is None


def test_mean_cost_is_none_when_all_unknown() -> None:
    assert _mean_cost([_result(), _result()]) is None


def test_mean_cost_ignores_unknown_entries() -> None:
    assert _mean_cost([_result(cost_usd=0.2), _result(cost_usd=None)]) == pytest.approx(0.2)


# --- pricing --------------------------------------------------------------


def test_estimate_cost_known_model() -> None:
    cost = estimate_cost("mistral-large-latest", 1_000_000, 1_000_000)
    assert cost == pytest.approx(2.0 + 6.0)


def test_estimate_cost_unknown_model_is_none() -> None:
    # distinguishes "unknown/offline" from "free"
    assert estimate_cost("some-unknown-model", 100, 100) is None


# --- rendering / dataset --------------------------------------------------


def test_results_table_renders_na_for_unknown_cost() -> None:
    variant = Variant("v", "d", retrieval=None, prompt_variant="strict")  # type: ignore[arg-type]
    report = _aggregate(variant, [_result()], k=5)
    table = results_table([report])
    assert "n/a" in table
    assert "recall@k" in table


def test_dataset_loads_and_is_well_formed() -> None:
    questions = load_dataset()
    assert len(questions) >= 15
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"

    for q in questions:
        assert q.question.strip()
        if q.answerable:
            assert q.gold_docs, f"{q.id} is answerable but has no gold docs"
            assert q.reference_answer.strip()
        else:
            assert not q.gold_docs, f"{q.id} is unanswerable but lists gold docs"


def test_dataset_has_unanswerable_coverage() -> None:
    questions = load_dataset()
    unanswerable = [q for q in questions if not q.answerable]
    # NOTE: with 19 questions one question moves this by ~0.05, so the band is
    # necessarily loose. Tighten once the set is expanded to ~120.
    ratio = len(unanswerable) / len(questions)
    assert 0.15 <= ratio <= 0.35, f"unanswerable ratio {ratio:.2f} outside target band"


def test_dataset_gold_docs_exist_in_corpus() -> None:
    """Gold paths must point at real files, or retrieval metrics are meaningless."""
    corpus = Path("data/corpus")
    if not corpus.exists():
        pytest.skip("corpus not fetched; run scripts/fetch_corpus.sh")
    for q in load_dataset():
        for doc in q.gold_docs:
            assert (corpus / doc).exists(), f"{q.id}: missing gold doc {doc}"


def test_dataset_jsonl_is_parseable_line_by_line() -> None:
    path = Path("eval/dataset.jsonl")
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - failure path
            pytest.fail(f"dataset line {lineno} is not valid JSON: {exc}")


def test_eval_question_defaults() -> None:
    q = EvalQuestion(
        id="x",
        type="lookup",
        question="q",
        reference_answer="a",
        gold_docs=["d.md"],
        gold_keywords=["k"],
        answerable=True,
    )
    assert q.answerable is True


def test_eval_settings_ignore_env_embedding_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the hash embedder inherited RAG_EMBEDDING_DIM=1024 from a Mistral .env."""
    from eval.run_eval import eval_settings

    monkeypatch.setenv("RAG_EMBEDDING_DIM", "1024")
    corpus = Path("data/corpus")
    assert eval_settings(corpus, "fake", "hash").embedding_dim == 512
    assert eval_settings(corpus, "fake", "mistral").embedding_dim == 1024
    assert eval_settings(corpus, "fake", "hash", embedding_dim=256).embedding_dim == 256


def test_eval_index_dir_changes_with_vector_affecting_settings(tmp_path: Path) -> None:
    from eval.run_eval import eval_settings, index_dir_for

    base = eval_settings(Path("c"), "fake", "hash")
    variants = [
        base,
        base.model_copy(update={"embedding_dim": 256}),
        base.model_copy(update={"chunk_max_chars": 900}),
        base.model_copy(update={"embedding_provider": "mistral"}),
    ]
    dirs = {index_dir_for(s, "structure", tmp_path) for s in variants}
    assert len(dirs) == len(variants)
    assert index_dir_for(base, "structure", tmp_path) == index_dir_for(base, "structure", tmp_path)
    assert index_dir_for(base, "fixed", tmp_path) != index_dir_for(base, "structure", tmp_path)
