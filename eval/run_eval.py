"""Evaluation harness.

Runs a labelled question set against pipeline variants and reports retrieval
metrics (recall@k, MRR, nDCG), generation metrics (answer-token F1 against the
reference, keyword coverage), citation accuracy, abstention behaviour on
unanswerable questions, and operational metrics (p50/p95 latency, cost).

Usage:
    python -m eval.run_eval                          # all variants
    python -m eval.run_eval --variants baseline strict # subset
    python -m eval.run_eval --limit 20               # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import ScoredChunk
from app.retrieval.pipeline import RetrievalConfig
from app.service import Service

EVAL_DIR = Path(__file__).parent
DATASET = EVAL_DIR / "dataset.jsonl"
REPORTS = EVAL_DIR / "reports"

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
    ]
)


def normalize_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


@dataclass(slots=True)
class EvalQuestion:
    id: str
    type: str
    question: str
    reference_answer: str
    gold_docs: list[str]
    gold_keywords: list[str]
    answerable: bool


@dataclass(slots=True)
class Variant:
    name: str
    description: str
    retrieval: RetrievalConfig
    prompt_variant: str
    chunk_strategy: str = "structure"


DEFAULT_VARIANTS = [
    Variant(
        "baseline",
        "fixed chunks, vector only, basic prompt",
        RetrievalConfig(enable_hybrid=False, enable_rerank=False, top_k=5),
        "basic",
        chunk_strategy="fixed",
    ),
    Variant(
        "+chunking",
        "structure-aware chunks, vector only",
        RetrievalConfig(enable_hybrid=False, enable_rerank=False, top_k=5),
        "basic",
        chunk_strategy="structure",
    ),
    Variant(
        "+hybrid",
        "structure chunks, vector+BM25 RRF",
        RetrievalConfig(enable_hybrid=True, enable_rerank=False, top_k=5),
        "basic",
        chunk_strategy="structure",
    ),
    Variant(
        "+rerank",
        "structure chunks, hybrid, reranker",
        RetrievalConfig(enable_hybrid=True, enable_rerank=True, top_k=5),
        "basic",
        chunk_strategy="structure",
    ),
    Variant(
        "strict",
        "hybrid + rerank + strict cite-or-abstain prompt",
        RetrievalConfig(enable_hybrid=True, enable_rerank=True, top_k=5),
        "strict",
        chunk_strategy="structure",
    ),
]


def load_dataset(path: Path = DATASET, limit: int | None = None) -> list[EvalQuestion]:
    rows: list[EvalQuestion] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(
                EvalQuestion(
                    id=data["id"],
                    type=data["type"],
                    question=data["question"],
                    reference_answer=data.get("reference_answer", ""),
                    gold_docs=data.get("gold_docs", []),
                    gold_keywords=data.get("gold_keywords", []),
                    answerable=data.get("answerable", True),
                )
            )
    return rows[:limit] if limit else rows


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def recall_at_k(retrieved_docs: list[str], gold_docs: list[str], k: int) -> float:
    if not gold_docs:
        return 0.0
    top = set(retrieved_docs[:k])
    return len(top & set(gold_docs)) / len(set(gold_docs))


def hit_at_k(retrieved_docs: list[str], gold_docs: list[str], k: int) -> float:
    if not gold_docs:
        return 0.0
    return 1.0 if set(retrieved_docs[:k]) & set(gold_docs) else 0.0


def reciprocal_rank(retrieved_docs: list[str], gold_docs: list[str]) -> float:
    gold = set(gold_docs)
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_docs: list[str], gold_docs: list[str], k: int) -> float:
    gold = set(gold_docs)
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved_docs[:k], start=1)
        if doc in gold
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_tokens(prediction)
    ref = normalize_tokens(reference)
    if not pred or not ref:
        return 0.0
    common = set(pred) & set(ref)
    if not common:
        return 0.0
    precision = len(common) / len(set(pred))
    recall = len(common) / len(set(ref))
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(prediction: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text = prediction.lower()
    found = sum(1 for kw in keywords if kw.lower() in text)
    return found / len(keywords)


def citation_accuracy(
    citations: list[str], gold_docs: list[str], retrieved: list[ScoredChunk]
) -> float:
    """Fraction of cited chunks that come from a gold document."""
    if not citations:
        return 0.0
    by_id = {c.chunk.chunk_id: c.chunk.doc_path for c in retrieved}
    gold = set(gold_docs)
    good = sum(1 for cid in citations if by_id.get(cid) in gold)
    return good / len(citations)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


@dataclass
class QuestionResult:
    question_id: str
    type: str
    answerable: bool
    found: bool
    recall: float = 0.0
    hit: float = 0.0
    rr: float = 0.0
    ndcg: float = 0.0
    f1: float = 0.0
    keywords: float = 0.0
    citation_acc: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    abstained: bool = False
    answer: str = ""
    error: str | None = None


@dataclass
class VariantReport:
    variant: str
    description: str
    k: int
    n_questions: int
    metrics: dict[str, float]
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)
    results: list[QuestionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "description": self.description,
            "k": self.k,
            "n_questions": self.n_questions,
            "metrics": self.metrics,
            "per_type": self.per_type,
            "results": [
                {
                    "id": r.question_id,
                    "type": r.type,
                    "answerable": r.answerable,
                    "found": r.found,
                    "recall": round(r.recall, 4),
                    "rr": round(r.rr, 4),
                    "ndcg": round(r.ndcg, 4),
                    "f1": round(r.f1, 4),
                    "keywords": round(r.keywords, 4),
                    "citation_acc": round(r.citation_acc, 4),
                    "latency_ms": round(r.latency_ms, 1),
                    "cost_usd": round(r.cost_usd, 6),
                    "answer": r.answer[:500],
                }
                for r in self.results
            ],
        }


def _build_service(settings: Settings, variant: Variant, workdir: Path) -> Service:
    """Fresh service + index per chunking strategy (cache by strategy)."""
    index_dir = workdir / f"index_{variant.chunk_strategy}_{settings.embedding_provider}"
    variant_settings = settings.model_copy(
        update={
            "index_dir": index_dir,
            "chunk_strategy": variant.chunk_strategy,
            "enable_hybrid": variant.retrieval.enable_hybrid,
            "enable_rerank": variant.retrieval.enable_rerank,
            "prompt_variant": variant.prompt_variant,
        }
    )
    svc = Service(variant_settings)
    if svc.store.count() == 0:
        svc.ingest(force=True)
    return svc


def run_variant(
    service: Service, variant: Variant, questions: list[EvalQuestion], k: int
) -> VariantReport:
    results: list[QuestionResult] = []
    chunk_map = {c.chunk_id: c.doc_path for c in service.retriever.store.all_chunks()}

    for q in questions:
        started = time.perf_counter()
        try:
            gen, trace = service.ask(
                q.question,
                top_k=k,
                config=variant.retrieval,
                trace=False,
                prompt_variant=variant.prompt_variant,
            )
            latency = (time.perf_counter() - started) * 1000

            retrieved_paths = [chunk_map.get(item["chunk_id"], "") for item in trace.retrieved]

            if q.answerable:
                recall = recall_at_k(retrieved_paths, q.gold_docs, k)
                hit = hit_at_k(retrieved_paths, q.gold_docs, k)
                rr = reciprocal_rank(retrieved_paths, q.gold_docs)
                ndcg = ndcg_at_k(retrieved_paths, q.gold_docs, k)
                f1 = token_f1(gen.answer, q.reference_answer)
                kw = keyword_coverage(gen.answer, q.gold_keywords)
                cited = {c.chunk_id for c in gen.citations}
                gold_chunk_ids = {
                    cid for cid, path in chunk_map.items() if path in set(q.gold_docs)
                }
                cit = len(cited & gold_chunk_ids) / len(cited) if cited else 0.0
            else:
                recall = hit = rr = ndcg = 1.0  # not applicable; excluded from retrieval avg
                f1 = kw = cit = 0.0

            results.append(
                QuestionResult(
                    question_id=q.id,
                    type=q.type,
                    answerable=q.answerable,
                    found=gen.found,
                    recall=recall,
                    hit=hit,
                    rr=rr,
                    ndcg=ndcg,
                    f1=f1,
                    keywords=kw,
                    citation_acc=cit,
                    latency_ms=latency,
                    cost_usd=trace.cost_usd,
                    abstained=not gen.found,
                    answer=gen.answer,
                    error=gen.parse_error,
                )
            )
        except Exception as exc:  # keep the run alive, record the failure
            results.append(
                QuestionResult(
                    question_id=q.id,
                    type=q.type,
                    answerable=q.answerable,
                    found=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return _aggregate(variant, results, k)


def _aggregate(variant: Variant, results: list[QuestionResult], k: int) -> VariantReport:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    def pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, round((p / 100) * (len(ordered) - 1)))
        return round(ordered[idx], 1)

    latencies = [r.latency_ms for r in results if r.error is None]
    metrics = {
        f"recall@{k}": mean([r.recall for r in answerable]),
        f"hit@{k}": mean([r.hit for r in answerable]),
        "mrr": mean([r.rr for r in answerable]),
        f"ndcg@{k}": mean([r.ndcg for r in answerable]),
        "answer_f1": mean([r.f1 for r in answerable]),
        "keyword_coverage": mean([r.keywords for r in answerable]),
        "citation_accuracy": mean([r.citation_acc for r in answerable if r.found]),
        "answer_rate": mean([1.0 if r.found else 0.0 for r in answerable]),
        "abstention_rate_unanswerable": mean([1.0 if r.abstained else 0.0 for r in unanswerable]),
        "p50_latency_ms": pct(latencies, 50),
        "p95_latency_ms": pct(latencies, 95),
        "cost_per_query_usd": round(statistics.fmean([r.cost_usd for r in results]), 6)
        if results
        else 0.0,
        "errors": sum(1 for r in results if r.error),
    }

    per_type: dict[str, dict[str, float]] = {}
    for qtype in sorted({r.type for r in results}):
        subset = [r for r in results if r.type == qtype]
        if qtype == "unanswerable":
            per_type[qtype] = {
                "n": len(subset),
                "abstention_rate": mean([1.0 if r.abstained else 0.0 for r in subset]),
            }
        else:
            per_type[qtype] = {
                "n": len(subset),
                "recall": mean([r.recall for r in subset]),
                "mrr": mean([r.rr for r in subset]),
                "f1": mean([r.f1 for r in subset]),
            }

    return VariantReport(
        variant=variant.name,
        description=variant.description,
        k=k,
        n_questions=len(results),
        metrics=metrics,
        per_type=per_type,
        results=results,
    )


def results_table(reports: list[VariantReport]) -> str:
    cols = [
        ("recall", "recall@k"),
        ("hit", "hit@k"),
        ("mrr", "MRR"),
        ("ndcg", "nDCG@k"),
        ("answer_f1", "F1"),
        ("keyword_coverage", "kw cov"),
        ("citation_accuracy", "cite acc"),
        ("answer_rate", "answer rate"),
        ("abstention_rate_unanswerable", "abstain"),
        ("p95_latency_ms", "p95 ms"),
        ("cost_per_query_usd", "cost/q"),
    ]
    header = f"| {'variant':<12} | " + " | ".join(f"{label:>10}" for _, label in cols) + " |"
    sep = "|" + "-" * 14 + "|" + "|".join("-" * 12 for _ in cols) + "|"
    lines = [header, sep]

    def to_key(col: str) -> str:
        if col == "recall":
            return f"recall@{reports[0].k}" if reports else "recall@5"
        if col == "hit":
            return f"hit@{reports[0].k}" if reports else "hit@5"
        if col == "ndcg":
            return f"ndcg@{reports[0].k}" if reports else "ndcg@5"
        return col

    for report in reports:
        cells = []
        for col, _ in cols:
            key = to_key(col)
            value = report.metrics.get(key, 0.0)
            if col in {"p95_latency_ms"}:
                cells.append(f"{value:>10.1f}")
            elif col == "cost_per_query_usd":
                cells.append(f"{value:>10.5f}")
            else:
                cells.append(f"{value:>10.3f}")
        lines.append(f"| {report.variant:<12} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation variants")
    parser.add_argument("--variants", nargs="*", default=None, help="variant names to run")
    parser.add_argument("--limit", type=int, default=None, help="limit number of questions")
    parser.add_argument("--k", type=int, default=5, help="retrieval cutoff")
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--workdir", default="data/eval_index")
    parser.add_argument("--out", default=None, help="report output dir")
    parser.add_argument(
        "--llm",
        default="fake",
        choices=["fake", "mistral"],
        help="LLM provider for generation metrics (mistral needs MISTRAL_API_KEY)",
    )
    parser.add_argument(
        "--embedder",
        default="hash",
        choices=["hash", "mistral"],
        help="embedding provider (mistral uses RAG_EMBEDDING_DIM, default 1024)",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=None, help="override embedding dimension"
    )
    args = parser.parse_args(argv)

    questions = load_dataset(limit=args.limit)
    variants = DEFAULT_VARIANTS
    if args.variants:
        wanted = set(args.variants)
        variants = [v for v in variants if v.name in wanted]
        if not variants:
            print(f"no matching variants; available: {[v.name for v in DEFAULT_VARIANTS]}")
            return 2

    overrides: dict[str, object] = {
        "corpus_dir": Path(args.corpus),
        "llm_provider": args.llm,
        "embedding_provider": args.embedder,
    }
    if args.embedding_dim is not None:
        overrides["embedding_dim"] = args.embedding_dim
    elif args.embedder == "mistral":
        overrides["embedding_dim"] = 1024
    base_settings = Settings(**overrides)  # type: ignore[arg-type]
    workdir = Path(args.workdir)
    reports: list[VariantReport] = []
    services: dict[str, Service] = {}

    for variant in variants:
        strategy_key = variant.chunk_strategy
        if strategy_key not in services:
            print(f"indexing corpus for chunk strategy '{strategy_key}'...", file=sys.stderr)
            services[strategy_key] = _build_service(base_settings, variant, workdir)
        # variant-specific toggles are applied per call via RetrievalConfig
        service = services[strategy_key]
        service.settings = service.settings.model_copy(
            update={"prompt_variant": variant.prompt_variant}
        )
        service.generator.settings = service.settings

        print(f"running variant {variant.name} on {len(questions)} questions...", file=sys.stderr)
        report = run_variant(service, variant, questions, args.k)
        reports.append(report)

    table = results_table(reports)
    print()
    print(table)

    out_dir = Path(args.out) if args.out else REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    payload = {
        "run_at": stamp,
        "k": args.k,
        "n_questions": len(questions),
        "corpus": args.corpus,
        "variants": [r.to_dict() for r in reports],
        "table": table,
    }
    (out_dir / f"report-{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "latest.md").write_text(
        f"# Evaluation report ({stamp})\n\n"
        f"Corpus: `{args.corpus}` · questions: {len(questions)} · k={args.k}\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir / 'latest.json'} and {out_dir / 'latest.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
