"""
End-to-end DeepEval tests for the Orchestrator Agent layer (server.py / orchestrator/).

Metrics:
  - GEval "Task Completion"   — did the orchestrator correctly route and respond?
  - GEval "Response Quality"  — is the search result coherent and useful?

Requires a running Orchestrator service at ORCHESTRATOR_URL (default: http://localhost:8003).
Skip with: SKIP_LIVE=1

The orchestrator exposes 4 simplified tools: save / search / improve / delete.
"""
import os

import httpx
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams as LLMTestCaseParams

from tests.deepeval.conftest import SKIP_LIVE

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8003")
EVAL_TIMEOUT = float(os.getenv("EVAL_TIMEOUT", "60"))


def _orchestrator_is_up() -> bool:
    try:
        r = httpx.get(f"{ORCHESTRATOR_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    SKIP_LIVE or not _orchestrator_is_up(),
    reason="Orchestrator not reachable — set ORCHESTRATOR_URL or unset SKIP_LIVE",
)


# ── Metric definitions ────────────────────────────────────────────────────────

def _task_completion_metric(model: str) -> GEval:
    return GEval(
        name="Orchestrator Task Completion",
        model=model,
        criteria=(
            "The orchestrator must have routed the request to the correct tool "
            "(save/search/improve/delete) and returned a meaningful response. "
            "A response of 'No results found.' for a factual query scores 0. "
            "A response acknowledging a save operation scores 1 for save intents."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.6,
    )


def _response_quality_metric(model: str) -> GEval:
    return GEval(
        name="Orchestrator Response Quality",
        model=model,
        criteria=(
            "The response must be coherent, non-empty, and directly address "
            "the user's input. Generic failure messages or empty results score 0."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_orchestrator(action: str, payload: dict) -> dict:
    with httpx.Client(base_url=ORCHESTRATOR_URL, timeout=EVAL_TIMEOUT) as client:
        resp = client.post(f"/{action}", json=payload)
    resp.raise_for_status()
    return resp.json()


# ── Test cases ────────────────────────────────────────────────────────────────

ORCHESTRATOR_CASES = [
    {
        "id": "e2e-001",
        "action": "save",
        "payload": {
            "text": "The company's annual sustainability report shows a 15% reduction in carbon emissions.",
            "session_id": "eval_session_001",
        },
        "expected_output": "saved successfully",
        "description": "Save action — document stored in the knowledge graph",
    },
    {
        "id": "e2e-002",
        "action": "search",
        "payload": {
            "query": "What is the carbon emission trend in the sustainability report?",
            "session_id": "eval_session_001",
        },
        "expected_output": "carbon emissions reduction",
        "description": "Search action — retrieves previously saved sustainability fact",
    },
    {
        "id": "e2e-003",
        "action": "search",
        "payload": {
            "query": "What happened in Q3 last year regarding procurement?",
            "session_id": "eval_session_001",
        },
        "expected_output": "procurement results",
        "description": "Temporal search — orchestrator routes to TEMPORAL search type",
    },
]


@pytest.mark.parametrize("case", ORCHESTRATOR_CASES, ids=[c["id"] for c in ORCHESTRATOR_CASES])
def test_orchestrator_task_completion(case, judge_model):
    """Orchestrator must route correctly and return a meaningful result."""
    body = _call_orchestrator(case["action"], case["payload"])
    actual_output = body.get("result", "") or body.get("message", "") or str(body)

    test_case = LLMTestCase(
        input=str(case["payload"]),
        actual_output=actual_output,
        expected_output=case["expected_output"],
        additional_metadata={"description": case["description"]},
    )

    assert_test(test_case, [_task_completion_metric(judge_model)])


@pytest.mark.parametrize(
    "case",
    [c for c in ORCHESTRATOR_CASES if c["action"] == "search"],
    ids=[c["id"] for c in ORCHESTRATOR_CASES if c["action"] == "search"],
)
def test_orchestrator_search_relevance(case, judge_model):
    """Search responses must be relevant to the query."""
    body = _call_orchestrator(case["action"], case["payload"])
    actual_output = body.get("result", "") or str(body)

    if not actual_output or actual_output.strip().lower() == "no results found.":
        pytest.skip(f"[{case['id']}] No results — knowledge graph may not be seeded")

    test_case = LLMTestCase(
        input=case["payload"]["query"],
        actual_output=actual_output,
    )

    metric = AnswerRelevancyMetric(threshold=0.5, model=judge_model)
    assert_test(test_case, [metric])


def test_orchestrator_health():
    """Orchestrator health endpoint must return ok."""
    r = httpx.get(f"{ORCHESTRATOR_URL}/health", timeout=10)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
