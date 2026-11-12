"""
Orchestrator Agent — intelligent MCP gateway.

Exposes 4 tools to clients (save, search, improve, delete).
save and search run as LangGraph StateGraphs; improve and delete are direct pass-throughs.

  save   → graphs/save_graph.py   (classify → override_hint? → ensure_dataset → check_version → cognify)
  search → graphs/search_graph.py (analyze → route_temporal | route_llm → execute_search)

Port: 8003
"""
import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ── Logging ────────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("ORCHESTRATOR_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("orchestrator")

# ── Session config ─────────────────────────────────────────────────────────────

SESSION_TTL = int(os.getenv("ORCHESTRATOR_SESSION_TTL", "3600"))

# In-memory session store
# { session_id: {agent_id, created_at, last_active, dataset_name} }
_sessions: dict[str, dict] = {}


def _new_session(agent_id: str) -> str:
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "agent_id":    agent_id,
        "created_at":  time.monotonic(),
        "last_active": time.monotonic(),
        "dataset_name": "main_dataset",
    }
    return sid


def _touch_session(sid: str) -> None:
    if sid in _sessions:
        _sessions[sid]["last_active"] = time.monotonic()


def _evict_sessions() -> None:
    now = time.monotonic()
    stale = [sid for sid, s in _sessions.items() if now - s["last_active"] > SESSION_TTL]
    for sid in stale:
        del _sessions[sid]
    if stale:
        logger.debug("Evicted %d stale sessions", len(stale))


# ── Tool catalogue ─────────────────────────────────────────────────────────────

ORCHESTRATOR_TOOL_LIST = [
    {
        "name": "save",
        "description": (
            "Save a document or fact into the knowledge graph. "
            "Automatically classifies the content into the correct typed dataset "
            "(policy, regulation, employees, market, workflow, contracts, or reports), "
            "detects temporal content, and handles document versioning."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "The text content to save.",
                },
                "hint": {
                    "type": "string",
                    "description": "Optional dataset type hint (e.g. 'policy', 'market'). Overrides auto-classification.",
                },
            },
            "required": ["data"],
        },
        "category": "memory_write",
        "typical_latency_seconds": 35,
    },
    {
        "name": "search",
        "description": (
            "Search the knowledge graph. "
            "Automatically selects the best search strategy "
            "(temporal, graph completion, summary, cypher, etc.) based on query analysis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "hint": {
                    "type": "string",
                    "description": "Optional hint: 'temporal', 'factual', 'summary', 'chunks'.",
                },
            },
            "required": ["query"],
        },
        "category": "memory_read",
        "typical_latency_seconds": 8,
    },
    {
        "name": "improve",
        "description": (
            "Run the quality improvement loop: apply feedback weights, "
            "persist sessions into the graph, sync traces, and rebuild the session index."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Dataset to improve. Defaults to 'all' (runs on every dataset).",
                    "default": "all",
                },
            },
            "required": [],
        },
        "category": "memory_manage",
        "typical_latency_seconds": 60,
    },
    {
        "name": "delete",
        "description": (
            "Delete knowledge from the graph. "
            "Pass 'all' to wipe everything, a dataset name (e.g. 'ds_policy') to delete a dataset, "
            "or a UUID to delete a single item."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "'all' | '<dataset_name>' | '<data-uuid>'",
                },
                "mode": {
                    "type": "string",
                    "enum": ["soft", "hard"],
                    "default": "soft",
                    "description": "Soft delete (reversible) or hard delete (permanent).",
                },
            },
            "required": ["target"],
        },
        "category": "memory_manage",
        "typical_latency_seconds": 5,
    },
]

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


# ── Tool implementations ───────────────────────────────────────────────────────

async def _tool_save(args: dict, session_id: str) -> str:
    """
    LangGraph pipeline:
      classify ──(hint?) ──► [override_hint] ──► ensure_dataset ──► check_version ──► cognify
    """
    from orchestrator.graphs.save_graph import run_save

    data = args.get("data", "").strip()
    hint = args.get("hint", "").strip()

    if not data:
        return "Error: 'data' argument is required."

    return await run_save(data=data, hint=hint, session_id=session_id)


async def _tool_search(args: dict, session_id: str) -> str:
    """
    LangGraph pipeline:
      analyze ──(temporal?) ──► [route_temporal | route_llm] ──► execute_search
    """
    from orchestrator.graphs.search_graph import run_search

    query = args.get("query", "").strip()
    hint  = args.get("hint", "").strip()

    if not query:
        return "Error: 'query' argument is required."

    return await run_search(query=query, hint=hint, session_id=session_id)


async def _tool_improve(args: dict, session_id: str) -> str:
    """Proxy to mcp-wrapper improve, optionally scoped to one dataset."""
    from orchestrator.mcp_client import wrapper

    dataset_name = args.get("dataset_name", "all").strip()

    if dataset_name and dataset_name != "all":
        try:
            await wrapper.call("switch_dataset", {"dataset_name": dataset_name})
        except Exception as exc:
            logger.warning("switch_dataset before improve failed: %s", exc)
        result = await wrapper.call("improve", {"dataset_name": dataset_name}, timeout=120)
    else:
        result = await wrapper.call("improve", {}, timeout=120)

    return result or "Improve loop completed."


async def _tool_delete(args: dict, session_id: str) -> str:
    """Route to prune / delete_dataset / delete based on target type."""
    from orchestrator.mcp_client import wrapper

    target = args.get("target", "").strip()
    mode   = args.get("mode", "soft")

    if not target:
        return "Error: 'target' argument is required."

    if target.lower() == "all":
        result = await wrapper.call("prune", {})
        return result or "All knowledge pruned."

    if _UUID_RE.match(target):
        result = await wrapper.call("delete", {"data_id": target, "mode": mode})
        return result or f"Item {target} deleted ({mode})."

    # Treat as dataset name
    result = await wrapper.call("delete_dataset", {"dataset_name": target})
    return result or f"Dataset '{target}' deleted."


# ── Dispatch table ─────────────────────────────────────────────────────────────

_TOOL_HANDLERS = {
    "save":    _tool_save,
    "search":  _tool_search,
    "improve": _tool_improve,
    "delete":  _tool_delete,
}


async def dispatch(tool: str, args: dict, session_id: str = "") -> str:
    handler = _TOOL_HANDLERS.get(tool)
    if handler is None:
        return f"Unknown tool: {tool}"
    return await handler(args, session_id)


# ── SSE helpers ────────────────────────────────────────────────────────────────

def _sse_result(req_id, text: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id":      req_id,
        "result":  {"content": [{"type": "text", "text": text}]},
    }
    return f"data: {json.dumps(payload)}\n\n"


def _sse_error(req_id, code: int, message: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id":      req_id,
        "error":   {"code": code, "message": message},
    }
    return f"data: {json.dumps(payload)}\n\n"


# ── Lifespan (session eviction background task) ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _evict_loop():
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            _evict_sessions()

    task = asyncio.create_task(_evict_loop())
    logger.info("Orchestrator Agent started on port 8003")
    yield
    task.cancel()
    from orchestrator.mcp_client import wrapper
    await wrapper.close()


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="Orchestrator Agent", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    from orchestrator.mcp_client import wrapper
    cb_open = wrapper._cb_is_open()
    return {
        "status":          "ok",
        "service":         "orchestrator",
        "sessions":        len(_sessions),
        "circuit_breaker": "open" if cb_open else "closed",
        "tools":           [t["name"] for t in ORCHESTRATOR_TOOL_LIST],
    }


@app.get("/tools")
async def list_tools():
    return {"tools": ORCHESTRATOR_TOOL_LIST}


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    method  = body.get("method", "")
    req_id  = body.get("id")
    params  = body.get("params", {})
    sid     = request.headers.get("mcp-session-id", "")

    # ── initialize ─────────────────────────────────────────────────────────
    if method == "initialize":
        agent_name = params.get("clientInfo", {}).get("name", "unknown")
        new_sid    = _new_session(agent_name)
        payload    = {
            "jsonrpc": "2.0",
            "id":      req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities":    {},
                "serverInfo":      {"name": "orchestrator", "version": "1.0.0"},
            },
        }
        return JSONResponse(
            content=payload,
            headers={"mcp-session-id": new_sid},
        )

    # ── notifications/initialized ──────────────────────────────────────────
    if method == "notifications/initialized":
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {}})

    # ── tools/list ─────────────────────────────────────────────────────────
    if method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id":      req_id,
            "result":  {"tools": ORCHESTRATOR_TOOL_LIST},
        })

    # ── tools/call ─────────────────────────────────────────────────────────
    if method == "tools/call":
        if sid and sid not in _sessions:
            return StreamingResponse(
                iter([_sse_error(req_id, -32600, "Unknown or expired session. Call initialize first.")]),
                media_type="text/event-stream",
            )

        _touch_session(sid)
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})

        async def _stream():
            try:
                result_text = await dispatch(tool_name, tool_args, sid)
                yield _sse_result(req_id, result_text)
            except Exception as exc:
                logger.error("Tool '%s' error: %s", tool_name, exc, exc_info=True)
                yield _sse_error(req_id, -32603, str(exc))

        return StreamingResponse(_stream(), media_type="text/event-stream")

    # ── unknown method ─────────────────────────────────────────────────────
    return JSONResponse({
        "jsonrpc": "2.0",
        "id":      req_id,
        "error":   {"code": -32601, "message": f"Method not found: {method}"},
    }, status_code=404)
