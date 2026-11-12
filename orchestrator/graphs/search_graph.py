"""
Search pipeline as a LangGraph StateGraph.

Flow:
                           ┌─ route_temporal ─┐
  analyze ──(temporal?) ───┤                  ├──► execute_search
                           └─── route_llm ────┘
"""
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

# Hint → search_type mapping for caller overrides
_HINT_MAP: dict[str, str] = {
    "temporal":  "TEMPORAL",
    "factual":   "GRAPH_COMPLETION",
    "summary":   "GRAPH_SUMMARY_COMPLETION",
    "chunks":    "CHUNKS",
    "cypher":    "CYPHER",
    "rag":       "RAG_COMPLETION",
    "cot":       "GRAPH_COMPLETION_COT",
    "lexical":   "CHUNKS_LEXICAL",
}


# ── State schema ──────────────────────────────────────────────────────────────

class SearchState(TypedDict):
    # Inputs
    query:      str
    hint:       str        # optional caller hint
    session_id: str

    # Analyzer outputs
    intent:      str       # save | search | improve | delete
    search_type: str       # LLM-selected search type
    is_temporal: bool
    confidence:  float

    # Router output (final search type after all overrides)
    final_search_type: str

    # Result
    result: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def analyze_node(state: SearchState) -> dict:
    """Classify query intent + detect temporal signal via LLM + keyword regex."""
    from orchestrator.query_analyzer import analyze

    intent = await analyze(state["query"])
    logger.info(
        "[search/analyze] intent=%s search_type=%s temporal=%s confidence=%.2f",
        intent.intent, intent.search_type, intent.is_temporal, intent.confidence,
    )
    return {
        "intent":      intent.intent,
        "search_type": intent.search_type,
        "is_temporal": intent.is_temporal,
        "confidence":  intent.confidence,
    }


async def route_temporal_node(state: SearchState) -> dict:
    """Temporal path: force search_type to TEMPORAL."""
    logger.info("[search/route_temporal] temporal signal → TEMPORAL")
    return {"final_search_type": "TEMPORAL"}


async def route_llm_node(state: SearchState) -> dict:
    """Non-temporal path: use LLM-selected search_type via MemoryRouter."""
    from orchestrator.memory_router import route

    # Re-use the QueryIntent dataclass fields from state
    from orchestrator.query_analyzer import QueryIntent
    intent = QueryIntent(
        intent=state["intent"],
        search_type=state["search_type"],
        is_temporal=False,     # already handled by the temporal branch
        confidence=state["confidence"],
    )
    route_result = route(intent)

    # Apply caller hint if provided (final override)
    hint = state.get("hint", "").strip().lower()
    final_type = _HINT_MAP.get(hint, route_result.search_type)

    logger.info("[search/route_llm] llm=%s → final=%s (hint=%s)", state["search_type"], final_type, hint or "none")
    return {"final_search_type": final_type}


async def execute_search_node(state: SearchState) -> dict:
    """Execute the search against mcp-wrapper using the resolved search type."""
    from orchestrator.mcp_client import wrapper

    search_type = state["final_search_type"]
    logger.info("[search/execute] query='%s...' type=%s", state["query"][:60], search_type)

    result = await wrapper.call(
        "search",
        {
            "search_query": state["query"],
            "search_type":  search_type,
        },
        timeout=90,
    )
    return {"result": result or "No results found."}


# ── Conditional edge ──────────────────────────────────────────────────────────

def _should_route_temporal(state: SearchState) -> str:
    # Hint "temporal" also forces the temporal branch
    hint = state.get("hint", "").strip().lower()
    if state["is_temporal"] or hint == "temporal":
        return "route_temporal"
    return "route_llm"


# ── Graph construction (compiled once at import time) ─────────────────────────

def _build() -> object:
    builder = StateGraph(SearchState)

    builder.add_node("analyze",        analyze_node)
    builder.add_node("route_temporal", route_temporal_node)
    builder.add_node("route_llm",      route_llm_node)
    builder.add_node("execute_search", execute_search_node)

    builder.set_entry_point("analyze")

    # Branch on temporal detection
    builder.add_conditional_edges(
        "analyze",
        _should_route_temporal,
        {"route_temporal": "route_temporal", "route_llm": "route_llm"},
    )

    builder.add_edge("route_temporal", "execute_search")
    builder.add_edge("route_llm",      "execute_search")
    builder.add_edge("execute_search", END)

    return builder.compile()


search_graph = _build()


# ── Public entry point ────────────────────────────────────────────────────────

async def run_search(query: str, hint: str = "", session_id: str = "") -> str:
    """Invoke the search graph and return the result text."""
    initial_state: SearchState = {
        "query":             query,
        "hint":              hint,
        "session_id":        session_id,
        # Safe defaults for output fields
        "intent":            "search",
        "search_type":       "GRAPH_COMPLETION",
        "is_temporal":       False,
        "confidence":        0.0,
        "final_search_type": "GRAPH_COMPLETION",
        "result":            "",
    }
    final = await search_graph.ainvoke(initial_state)
    return final.get("result", "No results found.")
