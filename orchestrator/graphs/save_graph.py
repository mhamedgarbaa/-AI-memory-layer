"""
Save pipeline as a LangGraph StateGraph.

Flow:
                        ┌─ override_hint ─┐
  classify ──(hint?) ───┤                 ├──► ensure_dataset ──► check_version ──► cognify
                        └────────────────►┘
"""
import json
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class SaveState(TypedDict):
    # Inputs (set before graph runs)
    data:            str
    hint:            str          # optional dataset type hint from caller
    session_id:      str

    # Classifier outputs
    dataset_name:           str
    type_label:             str
    classifier_confidence:  float

    # Temporal detection
    is_temporal: bool

    # Version manager outputs
    version_number:  int
    version_action:  str          # "create" | "update"
    versioned_text:  str

    # Final result text (JSON string)
    result: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def classify_node(state: SaveState) -> dict:
    """Classify document type → dataset name using keyword + LLM."""
    from orchestrator.document_classifier import classify
    from orchestrator.query_analyzer import _detect_temporal

    doc_type    = await classify(state["data"])
    is_temporal = _detect_temporal(state["data"])

    logger.info(
        "[save/classify] type=%s dataset=%s confidence=%.2f temporal=%s",
        doc_type.type_label, doc_type.dataset_name, doc_type.confidence, is_temporal,
    )
    return {
        "dataset_name":          doc_type.dataset_name,
        "type_label":            doc_type.type_label,
        "classifier_confidence": doc_type.confidence,
        "is_temporal":           is_temporal,
    }


async def override_hint_node(state: SaveState) -> dict:
    """Apply caller's dataset type hint, overriding the classifier result."""
    from orchestrator.document_classifier import TAXONOMY

    hint    = state["hint"].strip().lower()
    matched = next((e for e in TAXONOMY if e["label"] == hint), None)
    if matched:
        logger.info("[save/override_hint] %s → %s", state["dataset_name"], matched["dataset_name"])
        return {
            "dataset_name":          matched["dataset_name"],
            "type_label":            matched["label"],
            "classifier_confidence": 1.0,
        }
    # Unknown hint — keep classifier result
    logger.warning("[save/override_hint] Unknown hint '%s', keeping classifier result", hint)
    return {}


async def ensure_dataset_node(state: SaveState) -> dict:
    """Create the target dataset in mcp-wrapper if it doesn't exist yet."""
    from orchestrator.document_classifier import ensure_dataset_exists

    await ensure_dataset_exists(state["dataset_name"])
    logger.info("[save/ensure_dataset] dataset=%s ready", state["dataset_name"])
    return {}


async def check_version_node(state: SaveState) -> dict:
    """Search for existing content and tag the new text with a version number."""
    from orchestrator.version_manager import check_and_version

    vr = await check_and_version(state["data"], state["dataset_name"])
    logger.info(
        "[save/check_version] action=%s version=%d dataset=%s",
        vr.action, vr.version_number, vr.dataset_name,
    )
    return {
        "version_number": vr.version_number,
        "version_action": vr.action,
        "versioned_text": vr.versioned_text,
    }


async def cognify_node(state: SaveState) -> dict:
    """Kick off cognify and return immediately — mcp-wrapper runs the heavy
    docker-exec pipeline in background, so the caller is not blocked."""
    from orchestrator.mcp_client import wrapper

    cognify_out = await wrapper.call(
        "cognify",
        {
            "data":         state["versioned_text"],
            "temporal":     state["is_temporal"],
            "dataset_name": state["dataset_name"],
        },
        timeout=30,  # background-fire path returns in <1s; allow 30s safety margin
    )

    result = json.dumps({
        "status":                "processing",
        "dataset":               state["dataset_name"],
        "type":                  state["type_label"],
        "version":               state["version_number"],
        "action":                state["version_action"],
        "is_temporal":           state["is_temporal"],
        "classifier_confidence": round(state["classifier_confidence"], 2),
        "cognify":               (cognify_out or "")[:200],
    }, ensure_ascii=False)

    logger.info("[save/cognify] queued — %s", result[:120])
    return {"result": result}


# ── Conditional edge ──────────────────────────────────────────────────────────

def _should_override_hint(state: SaveState) -> str:
    return "override_hint" if state.get("hint", "").strip() else "ensure_dataset"


# ── Graph construction (compiled once at import time) ─────────────────────────

def _build() -> object:
    builder = StateGraph(SaveState)

    builder.add_node("classify",       classify_node)
    builder.add_node("override_hint",  override_hint_node)
    builder.add_node("ensure_dataset", ensure_dataset_node)
    builder.add_node("check_version",  check_version_node)
    builder.add_node("cognify",        cognify_node)

    builder.set_entry_point("classify")

    # Branch: apply hint if caller provided one
    builder.add_conditional_edges(
        "classify",
        _should_override_hint,
        {"override_hint": "override_hint", "ensure_dataset": "ensure_dataset"},
    )

    builder.add_edge("override_hint",  "ensure_dataset")
    builder.add_edge("ensure_dataset", "check_version")
    builder.add_edge("check_version",  "cognify")
    builder.add_edge("cognify",        END)

    return builder.compile()


save_graph = _build()


# ── Public entry point ────────────────────────────────────────────────────────

async def run_save(data: str, hint: str = "", session_id: str = "") -> str:
    """Invoke the save graph and return the final result JSON string."""
    initial_state: SaveState = {
        "data":                  data,
        "hint":                  hint,
        "session_id":            session_id,
        # Pre-fill all output fields with safe defaults
        "dataset_name":          "",
        "type_label":            "",
        "classifier_confidence": 0.0,
        "is_temporal":           False,
        "version_number":        1,
        "version_action":        "create",
        "versioned_text":        data,
        "result":                "",
    }
    final = await save_graph.ainvoke(initial_state)
    return final.get("result", "")
