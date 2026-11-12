"""
DeepEval tests for orchestrator/document_classifier.py

Metrics:
  - GEval "Classification Correctness" — rubric-based label accuracy
  - GEval "Confidence Calibration"     — confidence reflects actual correctness

Run:
    pytest tests/deepeval/test_document_classifier.py -v
or with DeepEval's verbose output:
    deepeval test run tests/deepeval/test_document_classifier.py
"""
import asyncio

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams as LLMTestCaseParams

from tests.deepeval.dataset import CLASSIFIER_CASES


# ── Metric definitions ────────────────────────────────────────────────────────

def _correctness_metric(model: str) -> GEval:
    return GEval(
        name="Classification Correctness",
        model=model,
        criteria=(
            "The Actual Output is a metadata string in the format "
            "'label=<X>, dataset=<Y>, confidence=<Z>'. "
            "The Expected Output is 'label=<X>, dataset=<Y>'. "
            "Score 1.0 if the label AND dataset in the Actual Output exactly match "
            "the Expected Output. Score 0.0 if either is wrong. "
            "Ignore the confidence value — it is not part of the correctness check."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
    )


def _confidence_calibration_metric(model: str) -> GEval:
    return GEval(
        name="Confidence Calibration",
        model=model,
        criteria=(
            "When the classifier returns the correct label, confidence should be "
            ">= 0.5. When the label is wrong, confidence should ideally be < 0.7. "
            "Evaluate whether the confidence value is well-calibrated given the result."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
    )


# ── Test cases ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", CLASSIFIER_CASES, ids=[c["id"] for c in CLASSIFIER_CASES])
def test_document_classifier_correctness(case, judge_model):
    """Each document must be classified into the correct label and dataset."""
    from orchestrator.document_classifier import classify

    result = asyncio.run(classify(case["text"]))

    actual_output = f"label={result.type_label}, dataset={result.dataset_name}, confidence={result.confidence:.2f}"
    expected_output = f"label={case['expected_label']}, dataset={case['expected_dataset']}"

    test_case = LLMTestCase(
        input=case["text"],
        actual_output=actual_output,
        expected_output=expected_output,
    )

    assert_test(test_case, [_correctness_metric(judge_model)])


@pytest.mark.parametrize("case", CLASSIFIER_CASES, ids=[c["id"] for c in CLASSIFIER_CASES])
def test_document_classifier_label_exact(case):
    """Deterministic check — no LLM judge needed — verifies the label directly."""
    from orchestrator.document_classifier import classify

    result = asyncio.run(classify(case["text"]))

    assert result.type_label == case["expected_label"], (
        f"[{case['id']}] Expected label '{case['expected_label']}', got '{result.type_label}' "
        f"(confidence={result.confidence:.2f})"
    )
    assert result.dataset_name == case["expected_dataset"], (
        f"[{case['id']}] Expected dataset '{case['expected_dataset']}', got '{result.dataset_name}'"
    )


@pytest.mark.parametrize("case", CLASSIFIER_CASES, ids=[c["id"] for c in CLASSIFIER_CASES])
def test_document_classifier_confidence_range(case):
    """Confidence must be in [0.0, 1.0] and above a minimum floor of 0.2."""
    from orchestrator.document_classifier import classify

    result = asyncio.run(classify(case["text"]))

    assert 0.0 <= result.confidence <= 1.0, (
        f"[{case['id']}] Confidence {result.confidence} out of range [0, 1]"
    )
    assert result.confidence >= 0.2, (
        f"[{case['id']}] Confidence {result.confidence} is too low (< 0.2)"
    )
