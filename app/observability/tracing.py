"""Tracing / observability.

A single `Tracer` writes one structured record per request (query, retrieved
chunks and scores, prompt variant, latency, token usage, cost). Backed by an
in-memory ring buffer and a JSONL file; Langfuse can be plugged in behind the
same interface without touching call sites.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# approximate USD per 1M tokens; override as models change
PRICING = {
    # USD per 1M tokens; approximate Mistral public list prices. Override as needed.
    "mistral-large-latest": {"input": 2.0, "output": 6.0},
    "mistral-medium-latest": {"input": 0.4, "output": 2.0},
    "mistral-small-latest": {"input": 0.1, "output": 0.3},
    "open-mistral-nemo": {"input": 0.15, "output": 0.15},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated USD cost, or None when the model has no known pricing.

    Returning None (rather than 0.0) keeps "unknown/offline" distinguishable from
    "genuinely free" in traces and eval reports.
    """
    price = PRICING.get(model)
    if not price:
        return None
    return (input_tokens / 1_000_000) * price["input"] + (output_tokens / 1_000_000) * price[
        "output"
    ]


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    question: str
    answer: str
    found: bool
    prompt_variant: str
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    stage_counts: dict[str, int] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    model: str = ""
    error: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tracer:
    def __init__(self, log_path: Path | None = None, max_memory: int = 500) -> None:
        self.log_path = Path(log_path) if log_path else None
        self._recent: deque[TraceRecord] = deque(maxlen=max_memory)
        self._lock = threading.Lock()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace: TraceRecord) -> None:
        with self._lock:
            self._recent.append(trace)
            if self.log_path:
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def recent(self, limit: int = 50) -> list[TraceRecord]:
        with self._lock:
            items = list(self._recent)
        return items[-limit:][::-1]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            items = list(self._recent)
        if not items:
            return {"count": 0}
        latencies = sorted(i.latency_ms for i in items)
        return {
            "count": len(items),
            "abstain_rate": round(sum(1 for i in items if not i.found) / len(items), 3),
            "p50_latency_ms": round(_pct(latencies, 50), 1),
            "p95_latency_ms": round(_pct(latencies, 95), 1),
            "avg_cost_usd": _avg_cost(items),
            "errors": sum(1 for i in items if i.error),
        }


def _avg_cost(items: list[TraceRecord]) -> float | None:
    known = [i.cost_usd for i in items if i.cost_usd is not None]
    return round(sum(known) / len(known), 6) if known else None


def _pct(sorted_values: list[float], pct: int) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, round((pct / 100) * (len(sorted_values) - 1)))
    return sorted_values[idx]
