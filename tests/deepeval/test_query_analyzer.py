"""
DeepEval tests for orchestrator/query_analyzer.py

Metrics:
  - GEval "Intent Accuracy"        — correct intent classification (save/search/improve/delete)
  - GEval "Search Type Relevance"  — chosen search type is appropriate for the query
  - Deterministic checks for temporal flag and valid enum values

Note: Intent classification (save/delete/improve) requires an LLM key.
      Without LLM_API_KEY, the analyzer falls back to intent="search" for all inputs.
      Intent tests are automatically skipped when no key is configured.

Run:
    pytest tests/deepeval/test_query_analyzer.py -v
"""
import asyncio
import os

import pytest
from deepeval import assert_test

_HAS_LLM = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams as LLMTestCaseParams

from tests.deepeval.dataset import QUERY_ANALYZER_CASES
from orchestrator.query_analyzer import SEARCH_TYPES


# ── Metric definitions ────────────────────────────────────────────────────────

def _intent_accuracy_metric(model: str) -> GEval:
    return GEval(
        name="Intent Accuracy",
        model=model,
        criteria=(
            "The Actual Output is routing metadata in the format "
            "'intent=<X>, search_type=<Y>, is_temporal=<Z>, confidence=<W>'. "
            "The Expected Output contains the correct intent value, e.g. 'intent=search'. "
            "Score 1.0 if the intent field in Actual Output exactly matches the intent value "
            "in Expected Output. Score 0.0 if they differ. "
            "Do NOT look for intent fields in the Input — only compare Actual Output vs Expected Output."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.8,
    )


def _search_type_relevance_metric(model: str) -> GEval:
    return GEval(
        name="Search Type Relevance",
        model=model,
        criteria=(
            "The Actual Output is ROUTING METADATA only — it contains the selected "
            "intent and search_type for the query, not the retrieved content itself. "
            "Evaluate ONLY whether the chosen search_type is the right strategy for "
            "the query in the Input. Do NOT penalise for missing retrieval content. "
            "Scoring guide: "
            "TEMPORAL → correct for queries referencing dates, quarters, years, or time periods. "
            "GRAPH_SUMMARY_COMPLETION → correct for overview/summary/explain queries. "
            "GRAPH_COMPLETION → correct for factual or relationship queries (default). "
            "CHUNKS / CHUNKS_LEXICAL → correct for keyword/similarity searches. "
            "Score 1.0 if the search_type is optimal, 0.5 if acceptable but not ideal, "
            "0.0 if clearly wrong."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.6,
    )


# ── Test cases ────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAS_LLM, reason="LLM key required for intent classification")
@pytest.mark.parametrize(
    "case",
    QUERY_ANALYZER_CASES,
    ids=[c["id"] for c in QUERY_ANALYZER_CASES],
)
def test_query_analyzer_intent(case, judge_model):
    """Each query must map to the correct intent via GEval judge."""
    from orchestrator.query_analyzer import analyze

    import json as _json
    result = asyncio.run(analyze(case["text"]))

    actual_output = _json.dumps({
        "intent": result.intent,
        "search_type": result.search_type,
        "is_temporal": result.is_temporal,
        "confidence": round(result.confidence, 2),
    })
    expected_output = _json.dumps({"intent": case["expected_intent"]})

    test_case = LLMTestCase(
        input=case["text"],
        actual_output=actual_output,
        expected_output=expected_output,
        additional_metadata={"description": case["description"]},
    )

    assert_test(test_case, [_intent_accuracy_metric(judge_model)])


@pytest.mark.skipif(not _HAS_LLM, reason="LLM key required for search type classification")
@pytest.mark.parametrize(
    "case",
    [c for c in QUERY_ANALYZER_CASES if c["expected_search_type"] is not None],
    ids=[c["id"] for c in QUERY_ANALYZER_CASES if c["expected_search_type"] is not None],
)
def test_query_analyzer_search_type_relevance(case, judge_model):
    """Search queries must produce a contextually appropriate search_type."""
    from orchestrator.query_analyzer import analyze

    result = asyncio.run(analyze(case["text"]))

    actual_output = (
        f"intent={result.intent}, search_type={result.search_type}, "
        f"is_temporal={result.is_temporal}"
    )

    test_case = LLMTestCase(
        input=case["text"],
        actual_output=actual_output,
    )

    assert_test(test_case, [_search_type_relevance_metric(judge_model)])


# ── Deterministic checks (no LLM judge) ──────────────────────────────────────

@pytest.mark.parametrize(
    "case",
    QUERY_ANALYZER_CASES,
    ids=[c["id"] for c in QUERY_ANALYZER_CASES],
)
def test_query_analyzer_temporal_flag(case):
    """Temporal detection must be correct for all labelled cases."""
    from orchestrator.query_analyzer import analyze

    result = asyncio.run(analyze(case["text"]))

    assert result.is_temporal == case["expected_temporal"], (
        f"[{case['id']}] Expected is_temporal={case['expected_temporal']}, "
        f"got {result.is_temporal} for query: '{case['text']}'"
    )


@pytest.mark.parametrize(
    "case",
    QUERY_ANALYZER_CASES,
    ids=[c["id"] for c in QUERY_ANALYZER_CASES],
)
def test_query_analyzer_valid_enum_values(case):
    """Intent and search_type must always be valid enum values."""
    from orchestrator.query_analyzer import analyze

    result = asyncio.run(analyze(case["text"]))

    assert result.intent in ("save", "search", "improve", "delete"), (
        f"[{case['id']}] Invalid intent: '{result.intent}'"
    )
    assert result.search_type in SEARCH_TYPES, (
        f"[{case['id']}] Invalid search_type: '{result.search_type}'"
    )
    assert 0.0 <= result.confidence <= 1.0, (
        f"[{case['id']}] Confidence {result.confidence} out of range"
    )


@pytest.mark.parametrize(
    "case",
    [c for c in QUERY_ANALYZER_CASES if c["expected_intent"] == "search" and c["expected_temporal"]],
    ids=[c["id"] for c in QUERY_ANALYZER_CASES if c["expected_intent"] == "search" and c["expected_temporal"]],
)
def test_query_analyzer_temporal_search_type_override(case):
    """Temporal search queries must resolve to TEMPORAL search_type."""
    from orchestrator.query_analyzer import analyze

    result = asyncio.run(analyze(case["text"]))

    assert result.search_type == "TEMPORAL", (
        f"[{case['id']}] Temporal query should use TEMPORAL search, got '{result.search_type}'"
    )
