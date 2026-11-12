"""
Shared pytest fixtures for DeepEval evaluation suite.

Judge model: DeepEval's built-in AzureOpenAI, wired to the Azure AI Foundry
endpoint from .env (LLM_API_KEY / LLM_ENDPOINT / LLM_MODEL).

Environment variables (all already set in .env):
  LLM_API_KEY   — Azure AI Foundry API key
  LLM_ENDPOINT  — Azure AI Foundry base URL
  LLM_MODEL     — model/deployment name (e.g. openai/gpt-4.1-nano)

Optional overrides:
  JUDGE_DEPLOYMENT  — deployment name override (strips provider prefix from LLM_MODEL)
  JUDGE_API_VERSION — Azure API version (default: 2025-01-01-preview)
  BASE_URL          — web service base URL (default: http://localhost:8000)
  EVAL_TIMEOUT      — HTTP timeout seconds (default: 60)
  SKIP_LIVE         — set "1" to skip tests that need a running stack
  COGNIFY_WAIT      — seconds to wait after seeding RAG facts (default: 15)
"""
import os
import time
import uuid

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
EVAL_TIMEOUT = float(os.getenv("EVAL_TIMEOUT", "60"))
SKIP_LIVE = os.getenv("SKIP_LIVE", "0") == "1"

# ── Wire DeepEval to Azure AI Foundry ─────────────────────────────────────────
# DeepEval's AzureOpenAI reads these standard env vars.
_api_key = os.getenv("LLM_API_KEY", "")
_raw_model = os.getenv("LLM_MODEL", "gpt-4.1-nano")
_deployment = os.getenv("JUDGE_DEPLOYMENT") or (
    _raw_model.split("/")[-1] if "/" in _raw_model else _raw_model
)
# Strip /models or /responses suffix to get the OpenAI-compat base URL
_raw_endpoint = os.getenv("LLM_ENDPOINT", "").rstrip("/")
for _sfx in ("/responses", "/chat/completions", "/models"):
    if _raw_endpoint.endswith(_sfx):
        _raw_endpoint = _raw_endpoint[: -len(_sfx)]
        break
_endpoint = _raw_endpoint
_api_version = os.getenv("JUDGE_API_VERSION", "2025-01-01-preview")

# Export so DeepEval's AzureOpenAI picks them up automatically
os.environ.setdefault("AZURE_OPENAI_API_KEY", _api_key)
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", _endpoint)
os.environ.setdefault("OPENAI_API_VERSION", _api_version)


def _make_judge():
    """Return a DeepEval AzureOpenAIModel judge instance."""
    from deepeval.models import AzureOpenAIModel

    return AzureOpenAIModel(
        model=_deployment,
        deployment_name=_deployment,
        api_key=_api_key,
        base_url=_endpoint,
        api_version=_api_version,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stack_is_up() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def http_client():
    with httpx.Client(base_url=BASE_URL, timeout=EVAL_TIMEOUT) as client:
        yield client


@pytest.fixture(scope="session")
def judge_model():
    """Session-scoped DeepEval AzureOpenAI judge."""
    return _make_judge()


@pytest.fixture(scope="session")
def seeded_rag_user(http_client):
    """
    Seeds RAG test facts into the live stack once per session and returns
    the user_id used. Waits for cognify to process before any query tests run.
    """
    if SKIP_LIVE or not _stack_is_up():
        pytest.skip("Stack not reachable — skipping RAG seed")

    from tests.deepeval.dataset import RAG_SEED_FACTS

    user_id = f"eval_rag_{uuid.uuid4().hex[:8]}"
    for seed in RAG_SEED_FACTS:
        resp = http_client.post(
            "/api/v1/memory/record",
            json={"user_id": user_id, "fact": seed["fact"]},
        )
        assert resp.status_code == 202, f"Seed failed: {resp.text}"

    time.sleep(int(os.getenv("COGNIFY_WAIT", "15")))
    return user_id
