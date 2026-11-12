"""
Query Analyzer — classifies a query/text into intent + search type + temporal flag.
Uses keyword regex first, then LLM for intent and search-type selection.
"""
import json
import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Temporal detection patterns ────────────────────────────────────────────────

_TEMPORAL_PATTERNS = re.compile(
    r"""
    \b(20[0-9]{2})\b                        # years 2000-2099
    | \b(Q[1-4])\b                           # Q1/Q2/Q3/Q4
    | \b(january|february|march|april|may|june|july|august
         |september|october|november|december)\b
    | \b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b
    | \b(yesterday|today|tomorrow)\b
    | \b(last\s+week|last\s+month|last\s+year|last\s+quarter)\b
    | \b(this\s+week|this\s+month|this\s+year|this\s+quarter)\b
    | \b(next\s+week|next\s+month|next\s+year)\b
    | \b(recently|currently|formerly|previously|historically)\b
    | \d{4}-\d{2}-\d{2}                      # ISO date
    | \d{2}/\d{2}/\d{4}                      # DD/MM/YYYY
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Valid search types ─────────────────────────────────────────────────────────

SEARCH_TYPES = [
    "GRAPH_COMPLETION",
    "GRAPH_COMPLETION_COT",
    "GRAPH_COMPLETION_CONTEXT_EXTENSION",
    "GRAPH_COMPLETION_DECOMPOSITION",
    "GRAPH_SUMMARY_COMPLETION",
    "TEMPORAL",
    "TRIPLET_COMPLETION",
    "CYPHER",
    "NATURAL_LANGUAGE",
    "RAG_COMPLETION",
    "SUMMARIES",
    "CHUNKS",
    "CHUNKS_LEXICAL",
    "FEELING_LUCKY",
]

_SYSTEM_PROMPT = f"""You are an intelligent query router for a knowledge graph memory system.
Classify the user's text and return ONLY a JSON object — no markdown, no explanation.

Schema:
{{
  "intent": "save" | "search" | "improve" | "delete",
  "search_type": "<one of the allowed search types>",
  "confidence": <float 0.0–1.0>
}}

Rules for intent:
- "save"    → user wants to store/add/remember a fact or document
- "search"  → user wants to retrieve/find/query information
- "improve" → user wants to improve/update/refine stored knowledge
- "delete"  → user wants to remove/forget/delete information
- Default:    "search"

Allowed search types and when to use them:
- TEMPORAL              → events, dates, when did X happen, historical queries
- GRAPH_SUMMARY_COMPLETION → overview, summarize, what is X, high-level explanation
- GRAPH_COMPLETION      → relationship queries, how does X relate to Y, factual questions (DEFAULT)
- GRAPH_COMPLETION_COT  → complex multi-step reasoning
- CYPHER                → precise structured queries, find exactly, specific attribute lookups
- CHUNKS                → raw text retrieval, semantic similarity
- CHUNKS_LEXICAL        → keyword-based text search
- FEELING_LUCKY         → any ambiguous query
- Default:                GRAPH_COMPLETION

Always return valid JSON only."""


@dataclass
class QueryIntent:
    intent: str       # save | search | improve | delete
    search_type: str  # one of SEARCH_TYPES
    is_temporal: bool
    confidence: float


def _detect_temporal(text: str) -> bool:
    return bool(_TEMPORAL_PATTERNS.search(text))


async def analyze(text: str) -> QueryIntent:
    """Classify text intent and best search type using LLM with temporal keyword pre-pass."""
    is_temporal = _detect_temporal(text)

    api_key   = os.getenv("LLM_API_KEY", "")
    raw_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    model     = raw_model.split("/")[-1] if "/" in raw_model else raw_model
    endpoint  = os.getenv("LLM_ENDPOINT", "").rstrip("/") or None

    if not api_key:
        # No LLM configured — return safe default
        search_type = "TEMPORAL" if is_temporal else "GRAPH_COMPLETION"
        return QueryIntent(intent="search", search_type=search_type, is_temporal=is_temporal, confidence=0.5)

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=endpoint)

        response = await client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text[:1000]},
            ],
            temperature=0,
            max_completion_tokens=100,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        intent      = data.get("intent", "search")
        search_type = data.get("search_type", "GRAPH_COMPLETION")
        confidence  = float(data.get("confidence", 0.8))

        if intent not in ("save", "search", "improve", "delete"):
            intent = "search"
        if search_type not in SEARCH_TYPES:
            search_type = "GRAPH_COMPLETION"

        # Temporal override: if we detected temporal markers but LLM didn't pick TEMPORAL, honour the flag
        if is_temporal and intent == "search" and search_type == "GRAPH_COMPLETION":
            search_type = "TEMPORAL"

        return QueryIntent(
            intent=intent,
            search_type=search_type,
            is_temporal=is_temporal,
            confidence=confidence,
        )

    except Exception as exc:
        logger.warning("QueryAnalyzer LLM failed (%s), using keyword fallback", exc)
        search_type = "TEMPORAL" if is_temporal else "GRAPH_COMPLETION"
        return QueryIntent(intent="search", search_type=search_type, is_temporal=is_temporal, confidence=0.4)
