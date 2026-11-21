"""CLI for ingestion and ad-hoc queries.

python -m scripts.ingest_cli ingest --force
python -m scripts.ingest_cli ask "what is an incremental model?"
"""

from __future__ import annotations

import argparse
import json
import sys

from app.config import get_settings
from app.providers import ProviderError
from app.service import Service


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="dbt docs RAG service CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="run (incremental) ingestion")
    ingest.add_argument("--force", action="store_true", help="reprocess all documents")
    ingest.add_argument("--limit", type=int, default=None, help="limit docs processed")

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--no-hybrid", action="store_true")
    ask.add_argument("--no-rerank", action="store_true")
    ask.add_argument("--variant", choices=["basic", "strict"], default=None)

    args = parser.parse_args(argv)
    service = Service(get_settings())

    notice = service.stale_index_notice()
    if notice:
        print(f"warning: {notice}\n", file=sys.stderr)

    if args.command == "ingest":
        try:
            stats = service.ingest(force=args.force, limit=args.limit)
        except ProviderError as exc:
            print(f"\ningestion failed:\n{exc}\n", file=sys.stderr)
            return 3
        print(json.dumps(stats.summary(), indent=2))
        return 0

    from app.retrieval.pipeline import RetrievalConfig

    config = RetrievalConfig.from_settings(service.settings, args.top_k)
    if args.no_hybrid:
        config.enable_hybrid = False
    if args.no_rerank:
        config.enable_rerank = False

    if service.store.count() == 0:
        print("index is empty — run `ingest` first", file=sys.stderr)
        return 1

    try:
        result, trace = service.ask(
            args.question, top_k=args.top_k, config=config, prompt_variant=args.variant
        )
    except ProviderError as exc:
        print(f"\nquery failed:\n{exc}\n", file=sys.stderr)
        return 3
    print(result.answer or "(no answer)")
    print()
    print(
        f"found={result.found} confidence={result.confidence:.2f} latency={trace.latency_ms:.0f}ms"
    )
    for citation in result.citations:
        print(f"  - {citation.doc} > {citation.section} [{citation.chunk_id}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
