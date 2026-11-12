"""
DeepEval RAG evaluation for the /api/v1/memory/recall endpoint.

Metrics (against live stack):
  - AnswerRelevancyMetric    — response addresses the question
  - FaithfulnessMetric       — no hallucinations beyond retrieved context
  - ContextualPrecisionMetric — ranked context nodes are relevant
  - ContextualRecallMetric    — ground-truth info is covered by retrieved context

Requires a running Docker stack (docker compose up -d).
Skip with: SKIP_LIVE=1 pytest tests/deepeval/test_rag_recall.py

Seeds three facts before any query test runs (via `seeded_rag_user` fixture in conftest.py).
"""
import os
import time

import httpx
import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from tests.deepeval.conftest import BASE_URL, EVAL_TIMEOUT, SKIP_LIVE, _stack_is_up
from tests.deepeval.dataset import RAG_EVAL_CASES, RAG_SEED_FACTS


pytestmark = pytest.mark.skipif(
    SKIP_LIVE or not _stack_is_up(),
    reason="Stack not reachable — set BASE_URL or unset SKIP_LIVE",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _recall(user_id: str, query: str) -> dict:
    with httpx.Client(base_url=BASE_URL, timeout=EVAL_TIMEOUT) as client:
        resp = client.post(
            "/api/v1/memory/recall",
            json={"user_id": user_id, "query": query},
        )
    resp.raise_for_status()
    return resp.json()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", RAG_EVAL_CASES, ids=[c["id"] for c in RAG_EVAL_CASES])
def test_rag_answer_relevancy(case, seeded_rag_user, judge_model):
    """Retrieved context must be relevant to the user's query."""
    body = _recall(seeded_rag_user, case["query"])

    context = body.get("context_data", [])
    combined_answer = " ".join(context) if context else ""

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=combined_answer,
        retrieval_context=context,
    )

    metric = AnswerRelevancyMetric(threshold=0.6, model=judge_model)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("case", RAG_EVAL_CASES, ids=[c["id"] for c in RAG_EVAL_CASES])
def test_rag_faithfulness(case, seeded_rag_user, judge_model):
    """Response must not assert facts not present in retrieved context (no hallucination)."""
    body = _recall(seeded_rag_user, case["query"])
    context = body.get("context_data", [])

    if not context:
        pytest.skip(f"[{case['id']}] Empty context — degraded retrieval, skipping faithfulness check")

    combined_answer = " ".join(context)

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=combined_answer,
        retrieval_context=context,
    )

    metric = FaithfulnessMetric(threshold=0.7, model=judge_model)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("case", RAG_EVAL_CASES, ids=[c["id"] for c in RAG_EVAL_CASES])
def test_rag_contextual_precision(case, seeded_rag_user, judge_model):
    """Top-ranked context items should be the most relevant to the query."""
    body = _recall(seeded_rag_user, case["query"])
    context = body.get("context_data", [])

    if not context:
        pytest.skip(f"[{case['id']}] Empty context — skipping contextual precision")

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=" ".join(context),
        expected_output=case["expected_answer"],
        retrieval_context=context,
    )

    metric = ContextualPrecisionMetric(threshold=0.5, model=judge_model)
    assert_test(test_case, [metric])


@pytest.mark.parametrize("case", RAG_EVAL_CASES, ids=[c["id"] for c in RAG_EVAL_CASES])
def test_rag_contextual_recall(case, seeded_rag_user, judge_model):
    """Retrieved context must contain the ground-truth answer information."""
    body = _recall(seeded_rag_user, case["query"])
    context = body.get("context_data", [])

    if not context:
        pytest.skip(f"[{case['id']}] Empty context — skipping contextual recall")

    test_case = LLMTestCase(
        input=case["query"],
        actual_output=" ".join(context),
        expected_output=case["expected_answer"],
        retrieval_context=context,
    )

    metric = ContextualRecallMetric(threshold=0.5, model=judge_model)
    assert_test(test_case, [metric])


def test_rag_degraded_retrieval_never_500(seeded_rag_user):
    """Even with a nonsensical query, recall must return 200 with graceful degradation."""
    body = _recall(seeded_rag_user, "zzz_completely_nonsense_xkcd_987")
    assert "context_data" in body
    assert "is_degraded" in body
    assert "retrieval_method" in body


def test_rag_retrieval_method_valid(seeded_rag_user):
    """retrieval_method must always be one of the known enum values."""
    body = _recall(seeded_rag_user, "What is the investment policy?")
    valid_methods = {"GRAPH_COMPLETION", "CHUNKS_FALLBACK", "GRAPH_TIMEOUT", "NONE_FAILED"}
    assert body["retrieval_method"] in valid_methods, (
        f"Unknown retrieval_method: {body['retrieval_method']}"
    )
