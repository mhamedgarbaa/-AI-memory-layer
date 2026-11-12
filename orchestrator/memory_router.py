"""
Memory Router — maps a QueryIntent to the single best mcp-wrapper search_type.
"""
import logging
from dataclasses import dataclass

from orchestrator.query_analyzer import QueryIntent

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    search_type: str          # one of the 14 mcp-wrapper search types
    needs_memify: bool        # reserved for future temporal enrichment trigger
    dataset_hint: str | None  # optional dataset scoping hint


def route(intent: QueryIntent, dataset_name: str | None = None) -> RouteResult:
    """
    Pick the single best search_type for this query intent.

    Priority rules:
    1. Temporal content always routes to TEMPORAL.
    2. LLM-selected GRAPH_SUMMARY_COMPLETION / CYPHER / COT types are honoured.
    3. Default: GRAPH_COMPLETION.
    """
    search_type = intent.search_type

    # Rule 1: any temporal signal → TEMPORAL search
    if intent.is_temporal:
        search_type = "TEMPORAL"
        logger.debug("Memory Router: temporal signal detected → TEMPORAL")

    # Rule 2: honour specific LLM-chosen types that carry meaning
    elif search_type in (
        "GRAPH_SUMMARY_COMPLETION",
        "CYPHER",
        "GRAPH_COMPLETION_COT",
        "GRAPH_COMPLETION_DECOMPOSITION",
        "GRAPH_COMPLETION_CONTEXT_EXTENSION",
        "CHUNKS",
        "CHUNKS_LEXICAL",
        "RAG_COMPLETION",
        "SUMMARIES",
        "FEELING_LUCKY",
    ):
        pass  # keep LLM choice

    # Rule 3: default
    else:
        search_type = "GRAPH_COMPLETION"

    logger.debug(
        "Memory Router: intent=%s is_temporal=%s → search_type=%s",
        intent.intent, intent.is_temporal, search_type,
    )

    return RouteResult(
        search_type=search_type,
        needs_memify=False,   # kept False for performance; can be enabled later
        dataset_hint=dataset_name,
    )
