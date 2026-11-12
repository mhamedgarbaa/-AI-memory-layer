"""
Document Classifier — maps a document to a typed dataset using keyword scoring + LLM fallback.
"""
import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Dataset taxonomy ───────────────────────────────────────────────────────────

TAXONOMY: list[dict] = [
    {
        "label": "policy",
        "dataset_name": "ds_policy",
        "keywords": ["policy", "policies", "procedure", "guideline", "guidelines", "rule", "rules", "compliance", "standard", "standards", "code of conduct"],
    },
    {
        "label": "regulation",
        "dataset_name": "ds_regulation",
        "keywords": ["law", "laws", "regulation", "regulations", "regulatory", "decree", "article", "articles", "gdpr", "iso", "legislation", "directive", "legal"],
    },
    {
        "label": "employees",
        "dataset_name": "ds_employees",
        "keywords": ["employee", "employees", "staff", "hr", "human resources", "salary", "salaries", "hiring", "onboarding", "performance review", "payroll", "headcount", "workforce"],
    },
    {
        "label": "market",
        "dataset_name": "ds_market",
        "keywords": ["revenue", "market", "sales", "competitor", "competitors", "q1", "q2", "q3", "q4", "forecast", "growth", "customer", "clients", "pipeline", "deal", "deals"],
    },
    {
        "label": "workflow",
        "dataset_name": "ds_workflow",
        "keywords": ["process", "workflow", "bpi", "bpmn", "step", "steps", "task", "tasks", "pipeline", "activity", "activities", "sequence", "automation"],
    },
    {
        "label": "contracts",
        "dataset_name": "ds_contracts",
        "keywords": ["contract", "contracts", "agreement", "agreements", "supplier", "suppliers", "vendor", "vendors", "sla", "service level", "terms", "clause", "clauses", "procurement"],
    },
    {
        "label": "reports",
        "dataset_name": "ds_reports",
        "keywords": ["report", "reports", "analysis", "findings", "dashboard", "kpi", "summary", "summaries", "metrics", "results", "audit", "review"],
    },
]

_SYSTEM_PROMPT = """You are a document classifier for an enterprise knowledge base.
Classify the document into exactly one category and return ONLY a JSON object — no markdown, no explanation.

Schema:
{
  "type_label": "<label>",
  "confidence": <float 0.0–1.0>
}

Categories:
- policy      → internal policies, procedures, guidelines, compliance rules
- regulation  → laws, regulations, GDPR, ISO standards, legal directives
- employees   → HR documents, staff information, salaries, hiring, performance
- market      → revenue figures, sales data, competitors, forecasts, customer data
- workflow    → business processes, BPMN diagrams, automation workflows, task sequences
- contracts   → supplier/vendor contracts, SLAs, procurement agreements
- reports     → analysis reports, KPI dashboards, audit findings, executive summaries

If uncertain, choose "reports" as the default.
Return only valid JSON."""


@dataclass
class DocumentType:
    dataset_name: str  # e.g. "ds_policy"
    type_label: str    # e.g. "policy"
    confidence: float


def _keyword_score(text: str) -> tuple[str, str, float]:
    """Score each category by keyword hits. Returns (label, dataset_name, confidence)."""
    lower = text.lower()
    best_label, best_dataset, best_score = "reports", "ds_reports", 0

    for entry in TAXONOMY:
        hits = sum(1 for kw in entry["keywords"] if kw in lower)
        if hits > best_score:
            best_score = hits
            best_label   = entry["label"]
            best_dataset = entry["dataset_name"]

    confidence = min(best_score / 5.0, 0.9) if best_score > 0 else 0.0
    return best_label, best_dataset, confidence


async def classify(text: str) -> DocumentType:
    """Classify document type. Uses keyword scoring; falls back to LLM if confidence < 0.6."""
    label, dataset_name, kw_confidence = _keyword_score(text)

    if kw_confidence >= 0.6:
        return DocumentType(dataset_name=dataset_name, type_label=label, confidence=kw_confidence)

    api_key   = os.getenv("LLM_API_KEY", "")
    raw_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    model     = raw_model.split("/")[-1] if "/" in raw_model else raw_model
    endpoint  = os.getenv("LLM_ENDPOINT", "").rstrip("/") or None

    if not api_key:
        return DocumentType(dataset_name=dataset_name, type_label=label, confidence=max(kw_confidence, 0.3))

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=endpoint)

        response = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text[:1500]},
            ],
            temperature=0,
            max_completion_tokens=80,
        )
        raw  = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        llm_label      = data.get("type_label", "reports")
        llm_confidence = float(data.get("confidence", 0.7))

        # Map label back to dataset_name
        matched = next((e for e in TAXONOMY if e["label"] == llm_label), None)
        if matched:
            return DocumentType(
                dataset_name=matched["dataset_name"],
                type_label=llm_label,
                confidence=llm_confidence,
            )

    except Exception as exc:
        logger.warning("DocumentClassifier LLM failed (%s), using keyword result", exc)

    return DocumentType(dataset_name=dataset_name, type_label=label, confidence=max(kw_confidence, 0.3))


async def ensure_dataset_exists(dataset_name: str) -> None:
    """Create a dataset in mcp-wrapper if it does not yet exist (idempotent)."""
    from orchestrator.mcp_client import wrapper
    try:
        await wrapper.call("create_dataset", {"dataset_name": dataset_name})
        logger.debug("Dataset ready: %s", dataset_name)
    except Exception as exc:
        # mcp-wrapper returns an error if dataset already exists — safe to ignore
        logger.debug("create_dataset '%s' response: %s", dataset_name, exc)
