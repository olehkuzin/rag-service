"""Generation: prompt building, strict parsing, citation resolution, abstention."""

from __future__ import annotations

import json

from app.generation.generator import (
    Generator,
    build_context,
    build_prompt,
    parse_response,
)
from app.models import Chunk, ScoredChunk
from app.providers import FakeLLM


def _scored(cid: str, text: str, section: str = "Section") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid,
        text=text,
        doc_id="d1",
        doc_path="d1.md",
        title="Doc title",
        section=section,
        content_hash=cid,
    )
    return ScoredChunk(chunk=chunk, score=0.5)


def test_build_context_numbers_chunks() -> None:
    context, lookup = build_context([_scored("a", "first"), _scored("b", "second")])
    assert "[c1]" in context
    assert "[c2]" in context
    assert lookup["c1"].chunk_id == "a"


def test_strict_prompt_includes_abstain_rule() -> None:
    system, user, _ = build_prompt("q?", [_scored("a", "text")], "strict")
    assert "found" in system
    assert "CONTEXT:" in user
    assert "QUESTION: q?" in user


def test_parse_resolves_citations_to_metadata() -> None:
    _, lookup = build_context([_scored("a", "text", section="Incremental")])
    raw = json.dumps({"found": True, "answer": "Yes", "citations": ["c1"], "confidence": 0.9})
    result, err = parse_response(raw, lookup, "strict")
    assert err is None
    assert result.found is True
    assert result.confidence == 0.9
    assert result.citations[0].section == "Incremental"
    assert result.citations[0].chunk_id == "a"


def test_parse_handles_fenced_json() -> None:
    _, lookup = build_context([_scored("a", "text")])
    raw = '```json\n{"found": true, "answer": "x", "citations": ["c1"]}\n```'
    result, err = parse_response(raw, lookup, "strict")
    assert err is None
    assert result.found is True


def test_strict_mode_abstains_without_citations() -> None:
    _, lookup = build_context([_scored("a", "text")])
    raw = json.dumps({"found": True, "answer": "unsupported claim", "citations": []})
    result, err = parse_response(raw, lookup, "strict")
    assert result.found is False
    assert err == "citation_missing"


def test_strict_mode_abstains_on_bad_json() -> None:
    _, lookup = build_context([_scored("a", "text")])
    result, err = parse_response("not json at all", lookup, "strict")
    assert result.found is False
    assert err == "unparseable_json"


def test_basic_mode_accepts_plain_text() -> None:
    _, lookup = build_context([_scored("a", "text")])
    result, err = parse_response("just prose", lookup, "basic")
    assert result.found is True
    assert result.answer == "just prose"
    assert err is None


def test_unknown_citation_ids_are_ignored() -> None:
    _, lookup = build_context([_scored("a", "text")])
    raw = json.dumps({"found": True, "answer": "x", "citations": ["c1", "c99"]})
    result, _ = parse_response(raw, lookup, "strict")
    assert [c.chunk_id for c in result.citations] == ["a"]


def test_confidence_is_clamped() -> None:
    _, lookup = build_context([_scored("a", "text")])
    raw = json.dumps({"found": True, "answer": "x", "citations": ["c1"], "confidence": 5})
    result, _ = parse_response(raw, lookup, "strict")
    assert result.confidence == 1.0


def test_fake_llm_cites_top_chunk() -> None:
    _, lookup = build_context([_scored("a", "Incremental models add new rows.")])
    llm = FakeLLM()
    system, user, _ = build_prompt(
        "what is incremental?", [_scored("a", "Incremental models add new rows.")], "strict"
    )
    raw, in_tok, out_tok = llm.complete(system=system, user=user, max_tokens=100)
    result, err = parse_response(raw, lookup, "strict")
    assert err is None
    assert result.found is True
    assert result.citations[0].chunk_id == "a"
    assert in_tok >= 0
    assert out_tok >= 0


def test_generator_records_usage(service) -> None:
    result = service.ask("what are models")[0]
    assert result.usage_input_tokens >= 0
    assert result.usage_output_tokens >= 0


def test_abstains_with_no_chunks() -> None:
    gen = Generator(FakeLLM(), __import__("app.config", fromlist=["Settings"]).Settings())
    result = gen.generate("question", [])
    assert result.found is False


def _answer_with_code() -> str:
    return json.dumps(
        {
            "found": True,
            "answer": (
                "Use this model:\n```sql\nselect * from {{ ref('orders') }}\n```\n"
                "Then run dbt build."
            ),
            "citations": ["c1"],
            "confidence": 0.9,
        }
    )


def test_json_answer_containing_a_code_fence_parses_in_strict_mode() -> None:
    """Regression: the fence regex used to grab the SQL inside the JSON and abstain."""
    _, lookup = build_context([_scored("a", "text")])
    result, err = parse_response(_answer_with_code(), lookup, "strict")
    assert err is None
    assert result.found is True
    assert "select * from {{ ref('orders') }}" in result.answer
    assert result.citations[0].chunk_id == "a"


def test_json_answer_containing_a_code_fence_parses_in_basic_mode() -> None:
    _, lookup = build_context([_scored("a", "text")])
    result, err = parse_response(_answer_with_code(), lookup, "basic")
    assert err is None
    assert result.found is True
    assert result.answer.startswith("Use this model:")  # not the raw JSON blob
    assert result.citations[0].chunk_id == "a"


def test_fenced_json_whose_answer_contains_a_fence_parses() -> None:
    _, lookup = build_context([_scored("a", "text")])
    raw = f"```json\n{_answer_with_code()}\n```"
    result, err = parse_response(raw, lookup, "strict")
    assert err is None
    assert result.found is True
    assert "```sql" in result.answer


def test_json_after_prose_preamble_parses() -> None:
    _, lookup = build_context([_scored("a", "text")])
    raw = 'Here is the answer:\n{"found": true, "answer": "x", "citations": ["c1"]}\nHope it helps.'
    result, err = parse_response(raw, lookup, "strict")
    assert err is None
    assert result.found is True
    assert result.answer == "x"
