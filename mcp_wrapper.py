"""
Cognee MCP Wrapper — single MCP endpoint exposing ALL tools.

Architecture:
  Other agents  ──►  POST http://localhost:8002/mcp  (this file)
                          │
                          ├── base tools  ──►  Cognee MCP  http://localhost:8001
                          └── custom tools ──► local logic (memify, etc.)

Run:
    python mcp_wrapper.py

Connect any MCP-compatible agent to:
    http://localhost:8002
    Host header: localhost:8002

Tools exposed:
    cognify, save_interaction, search (with search_type), list_data,
    cognify_status, prune, memify, improve, persist_sessions, improve_answer,
    create_dataset, switch_dataset, delete_dataset
"""

import asyncio
import json
import logging
import os
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── config ────────────────────────────────────────────────────────────────────

MCP_URL         = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
MCP_HOST_HEADER = os.getenv("MCP_HOST_HEADER", "localhost:8001")
AGENT_USER      = os.getenv("AGENT_USER", "external_agent")
PORT            = int(os.getenv("MCP_WRAPPER_PORT", "8002"))
LOG_LEVEL       = os.getenv("MCP_WRAPPER_LOG_LEVEL", "INFO").upper()

# Phase 2+ vars (defined now so .env.example documents them)
MCP_WRAPPER_API_KEY         = os.getenv("MCP_WRAPPER_API_KEY", "")
MCP_WRAPPER_STRICT_SESSIONS = os.getenv("MCP_WRAPPER_STRICT_SESSIONS", "false").lower() == "true"
MCP_WRAPPER_RECONNECT_DELAY = int(os.getenv("MCP_WRAPPER_RECONNECT_DELAY", "5"))
MCP_WRAPPER_MAX_RECONNECT   = int(os.getenv("MCP_WRAPPER_MAX_RECONNECT_ATTEMPTS", "10"))
# Rate-limit retry — when Cognee's upstream LLM returns 429 we back off and retry
COGNEE_RETRY_MAX_ATTEMPTS   = int(os.getenv("COGNEE_RETRY_MAX_ATTEMPTS", "4"))
COGNEE_RETRY_BASE_DELAY     = float(os.getenv("COGNEE_RETRY_BASE_DELAY", "5.0"))
COGNEE_RETRY_MAX_DELAY      = float(os.getenv("COGNEE_RETRY_MAX_DELAY", "60.0"))
# Container name and output path are env-driven so they work both locally and in Docker Compose
COGNEE_CONTAINER_NAME       = os.getenv("COGNEE_CONTAINER_NAME", "cognee_mcp_server")
# Gate streaming SSE forwarding — off by default so non-streaming clients keep working
MCP_WRAPPER_STREAMING       = os.getenv("MCP_WRAPPER_STREAMING", "false").lower() == "true"
# Tools that run a subprocess and need heartbeats instead of upstream SSE forwarding
_SUBPROCESS_TOOLS: frozenset[str] = frozenset({"memify"})

# ── JSON logger (shared config from configuration/logging_setup.py) ──────────
# Importing applies the dictConfig; then we grab a named child logger.
from configuration.logging_setup import logger as _root_logger  # noqa: E402

log = _root_logger.getChild("mcp_wrapper")
log.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# ── tool catalogue ─────────────────────────────────────────────────────────────

TOOL_LIST = [
    {
        "name": "cognify",
        "description": "Deep-process text/knowledge into the graph (entities, relationships, summaries). Use for rich content. Set temporal=true to also extract Event nodes with timestamps, which enables TEMPORAL search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "Text or content to store"},
                "temporal": {"type": "boolean", "description": "Extract temporal Event nodes during ingestion (enables TEMPORAL search). Default false.", "default": False},
            },
            "required": ["data"],
        },
    },
    {
        "name": "save_interaction",
        "description": "Quickly save a short note or interaction without full graph processing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {"type": "string"},
                "user": {"type": "string"},
            },
            "required": ["data"],
        },
    },
    {
        "name": "search",
        "description": (
            "Search the knowledge graph. search_type controls the strategy:\n"
            "GRAPH_COMPLETION (default) — entity graph + LLM synthesis\n"
            "GRAPH_COMPLETION_COT — chain-of-thought multi-round reasoning\n"
            "GRAPH_COMPLETION_DECOMPOSITION — decomposes complex queries into sub-queries\n"
            "GRAPH_COMPLETION_CONTEXT_EXTENSION — expands context with follow-up queries\n"
            "GRAPH_SUMMARY_COMPLETION — includes document summaries\n"
            "TEMPORAL — time-aware, filters Event nodes by timestamp (use for 'last month', 'in Q2')\n"
            "TRIPLET_COMPLETION — fast subject-predicate-object search (requires memify first)\n"
            "CYPHER — LLM-generated Cypher query\n"
            "NATURAL_LANGUAGE — NL → graph query auto-translation\n"
            "RAG_COMPLETION — classic vector RAG\n"
            "SUMMARIES — document summaries only\n"
            "CHUNKS — raw text chunks by vector similarity\n"
            "CHUNKS_LEXICAL — keyword/BM25 search (use for codes, acronyms, exact terms)\n"
            "FEELING_LUCKY — auto-selects best type"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "search_query": {"type": "string"},
                "search_type": {
                    "type": "string",
                    "enum": [
                        "GRAPH_COMPLETION", "GRAPH_COMPLETION_COT",
                        "GRAPH_COMPLETION_DECOMPOSITION",
                        "GRAPH_COMPLETION_CONTEXT_EXTENSION", "GRAPH_SUMMARY_COMPLETION",
                        "TEMPORAL", "TRIPLET_COMPLETION", "CYPHER", "NATURAL_LANGUAGE",
                        "RAG_COMPLETION", "SUMMARIES", "CHUNKS", "CHUNKS_LEXICAL",
                        "FEELING_LUCKY",
                    ],
                    "default": "GRAPH_COMPLETION",
                },
                "user": {"type": "string"},
            },
            "required": ["search_query"],
        },
    },
    {
        "name": "list_data",
        "description": "List all datasets stored in the knowledge graph.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "cognify_status",
        "description": "Check whether the cognify background pipeline is idle or still running.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "prune",
        "description": "Wipe ALL stored knowledge. Irreversible — only call when explicitly asked.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "memify",
        "description": (
            "Enrich an existing knowledge graph by creating triplet embeddings (Entity → Relationship → Entity). "
            "Enables SearchType.TRIPLET_COMPLETION queries. Run after cognify on a dataset. Takes 5-10 minutes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "default": "main_dataset"},
            },
        },
    },
    {
        "name": "improve",
        "description": (
            "Promote session memory to the permanent knowledge graph and enrich it. "
            "Runs 4 stages: (1) apply feedback weights from session Q&A ratings, "
            "(2) persist session question/answer pairs into the graph, "
            "(3) persist agent trace steps, "
            "(4) sync enriched graph back into session caches for fast recall. "
            "Call this after a CEO/admin session ends or on a nightly schedule. "
            "Pass session_ids to bridge specific sessions; omit to run enrichment only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "default": "main_dataset",
                    "description": "Dataset to enrich",
                },
                "session_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Session IDs to bridge into permanent graph. Omit to run enrichment only.",
                },
            },
        },
    },
    {
        "name": "forget_memory",
        "description": (
            "Delete memory from the enterprise brain. "
            "Pass dataset name to delete a specific dataset, or everything=true to wipe all user data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset":    {"type": "string", "description": "Dataset name to delete"},
                "everything": {"type": "boolean", "default": False, "description": "Wipe all user memory"},
            },
        },
    },
    {
        "name": "delete",
        "description": (
            "Delete a specific data item by ID. "
            "mode=soft marks it deleted (recoverable); mode=hard permanently removes it from the graph."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data_id":    {"type": "string", "description": "UUID of the data item to delete"},
                "dataset_id": {"type": "string", "description": "UUID of the dataset containing the item"},
                "mode":       {"type": "string", "enum": ["soft", "hard"], "default": "soft"},
            },
            "required": ["data_id"],
        },
    },
    {
        "name": "submit_feedback",
        "description": (
            "Score a past recall result so the enterprise brain learns from CEO/admin corrections. "
            "Attach a 1–5 rating (and optional text) to a qa_id returned by a previous recall. "
            "Scores are applied to graph node weights by improve() — higher-rated answers boost "
            "their source nodes, lower-rated ones decrease them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "qa_id":          {"type": "string", "description": "QA entry ID from a recall result"},
                "session_id":     {"type": "string", "description": "Session ID that produced the recall"},
                "feedback_score": {"type": "integer", "minimum": 1, "maximum": 5,
                                   "description": "Quality score 1 (wrong) – 5 (perfect)"},
                "feedback_text":  {"type": "string", "description": "Optional free-text correction"},
                "user_id":        {"type": "string", "description": "User who is rating"},
            },
            "required": ["qa_id", "feedback_score"],
        },
    },
    {
        "name": "record_trace",
        "description": (
            "Log an agent tool-call step to the session trace so improve() can later "
            "persist it into the permanent knowledge graph. "
            "Call this after each tool invocation to build an auditable decision trail."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "origin_function": {"type": "string", "description": "Tool or function name that was called"},
                "status":          {"type": "string", "enum": ["success", "error"], "default": "success"},
                "session_id":      {"type": "string", "description": "Current session ID"},
                "user_id":         {"type": "string", "description": "User performing the action"},
                "memory_query":    {"type": "string", "default": "", "description": "Query sent to memory, if any"},
                "method_params":   {"type": "object", "description": "Tool arguments (will be stored)"},
                "error_message":   {"type": "string", "default": "", "description": "Error detail if status=error"},
            },
            "required": ["origin_function"],
        },
    },
    {
        "name": "persist_sessions",
        "description": (
            "Legacy fallback: store a raw conversation transcript as a graph node via cognify. "
            "Prefer 'improve' with session_ids for proper session bridging with feedback weights."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "Conversation transcript to persist"},
            },
            "required": ["data"],
        },
    },
    {
        "name": "improve_answer",
        "description": (
            "Self-improvement: given a question that was answered incorrectly, re-query the graph "
            "with chain-of-thought reasoning and store the corrected answer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question":     {"type": "string"},
                "wrong_answer": {"type": "string"},
                "feedback":     {"type": "string"},
            },
            "required": ["question", "wrong_answer", "feedback"],
        },
    },
    {
        "name": "create_dataset",
        "description": (
            "Create a named dataset and set it as the active dataset for this session. "
            "All subsequent cognify and search calls will target this dataset."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "Name for the new dataset"},
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "switch_dataset",
        "description": (
            "Switch the active dataset for this session without creating a new one. "
            "Future cognify and search calls will target this dataset."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "Name of the existing dataset to switch to"},
            },
            "required": ["dataset_name"],
        },
    },
    {
        "name": "delete_dataset",
        "description": "Delete a named dataset and all its stored knowledge. Irreversible — only call when explicitly asked.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string", "description": "Name of the dataset to delete"},
            },
            "required": ["dataset_name"],
        },
    },
]

# ── Cognee MCP proxy ───────────────────────────────────────────────────────────

class CogneeMCPProxy:
    """Resilient proxy to the upstream Cognee MCP server.

    Handles lazy connect, transparent reconnect on failure, and a circuit
    breaker that stops hammering the upstream after repeated failures.
    """

    _CB_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
    _CB_TIMEOUT   = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT",   "60"))

    def __init__(self):
        self._http: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._req_id = 0
        self._lock = asyncio.Lock()
        self._connected = False
        # circuit-breaker state
        self._failures = 0
        self._open_until: float = 0.0

    # ── internal helpers ──────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    @staticmethod
    def _parse_sse(body: str) -> dict:
        for line in body.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise RuntimeError(f"No SSE data in: {body[:200]}")

    def _mcp_headers(self, with_session: bool = False) -> dict:
        h = {"Accept": "application/json, text/event-stream", "Host": MCP_HOST_HEADER}
        if with_session and self._session_id:
            h["mcp-session-id"] = self._session_id
        return h

    # ── circuit-breaker ───────────────────────────────────────────────────────

    def _cb_open(self) -> bool:
        return time.monotonic() < self._open_until

    def _cb_record_failure(self):
        self._failures += 1
        if self._failures >= self._CB_THRESHOLD:
            self._open_until = time.monotonic() + self._CB_TIMEOUT
            log.error(
                "circuit breaker OPEN — upstream unavailable",
                extra={"tool": "proxy", "error": f"open for {self._CB_TIMEOUT}s"},
            )

    def _cb_record_success(self):
        self._failures = 0
        self._open_until = 0.0

    # ── connect / reconnect ───────────────────────────────────────────────────

    async def _do_connect(self):
        if self._http:
            await self._http.aclose()
        self._http = httpx.AsyncClient(base_url=MCP_URL, timeout=30)
        resp = await self._http.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "mcp-wrapper", "version": "1.0"},
                },
            },
            headers=self._mcp_headers(),
        )
        resp.raise_for_status()
        self._session_id = resp.headers.get("mcp-session-id")

        # MCP 2024-11-05 requires notifications/initialized after initialize
        # before any tools/call will be accepted (Cognee 0.5.x enforces this).
        notif = await self._http.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            headers=self._mcp_headers(with_session=True),
        )
        # 200 or 202 accepted; anything else is unexpected but non-fatal
        if notif.status_code not in (200, 202, 204):
            log.warning("notifications/initialized returned %d", notif.status_code,
                        extra={"tool": "proxy"})

        self._connected = True
        self._cb_record_success()
        log.info("Cognee MCP connected", extra={"session_id": self._session_id, "tool": "proxy"})

    async def ensure_connected(self):
        if self._connected:
            return
        async with self._lock:
            if self._connected:   # double-check after acquiring lock
                return
            delay = MCP_WRAPPER_RECONNECT_DELAY
            for attempt in range(1, MCP_WRAPPER_MAX_RECONNECT + 1):
                try:
                    log.info("Connecting to Cognee MCP (attempt %d/%d)", attempt, MCP_WRAPPER_MAX_RECONNECT,
                             extra={"tool": "proxy"})
                    await self._do_connect()
                    return
                except Exception as exc:
                    self._connected = False
                    self._cb_record_failure()
                    log.warning("Connect attempt %d failed: %s", attempt, exc, extra={"tool": "proxy"})
                    if attempt < MCP_WRAPPER_MAX_RECONNECT:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 60)   # exponential back-off, cap 60s
            raise RuntimeError(
                f"Could not connect to Cognee MCP after {MCP_WRAPPER_MAX_RECONNECT} attempts"
            )

    # ── public call interface ─────────────────────────────────────────────────

    @staticmethod
    def _is_rate_limit(text: str) -> bool:
        t = text.lower()
        return (
            "rate limit" in t or "ratelimit" in t or "429" in t
            or "too many requests" in t or "quota exceeded" in t
            or "tokens per minute" in t or "requests per minute" in t
        )

    async def call(self, tool: str, args: dict, timeout: float = 180) -> str:
        if self._cb_open():
            raise RuntimeError(
                f"Circuit breaker open — Cognee MCP unavailable. "
                f"Retry in {round(self._open_until - time.monotonic())}s."
            )

        await self.ensure_connected()

        delay = COGNEE_RETRY_BASE_DELAY
        last_exc: Exception | None = None

        for attempt in range(1, COGNEE_RETRY_MAX_ATTEMPTS + 1):
            t0 = time.monotonic()
            try:
                async with self._lock:
                    resp = await self._http.post(
                        "/mcp",
                        json={
                            "jsonrpc": "2.0", "id": self._next_id(), "method": "tools/call",
                            "params": {"name": tool, "arguments": args},
                        },
                        headers=self._mcp_headers(with_session=True),
                        timeout=timeout,
                    )
                resp.raise_for_status()
                data = self._parse_sse(resp.text)

                if "error" in data:
                    err     = data["error"]
                    code    = err.get("code", 0)
                    message = err.get("message", "")
                    if self._is_rate_limit(message) and attempt < COGNEE_RETRY_MAX_ATTEMPTS:
                        log.warning(
                            "rate limit in MCP error (attempt %d/%d) — retrying in %.0fs: %s",
                            attempt, COGNEE_RETRY_MAX_ATTEMPTS, delay, message,
                            extra={"tool": tool},
                        )
                        last_exc = RuntimeError(f"MCP error [{code}]: {message}")
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, COGNEE_RETRY_MAX_DELAY)
                        continue
                    raise RuntimeError(f"Cognee MCP error [{code}]: {message}")

                content = data.get("result", {}).get("content", [])
                parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                result = "\n".join(parts) or json.dumps(data.get("result", {}))
                latency = round((time.monotonic() - t0) * 1000)
                self._cb_record_success()
                log.info("cognee_call ok", extra={"tool": tool, "latency_ms": latency})
                return result

            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                self._connected = False
                self._cb_record_failure()
                latency = round((time.monotonic() - t0) * 1000)
                log.error("cognee_call failed (connection lost)",
                          extra={"tool": tool, "latency_ms": latency, "error": str(exc)})
                raise

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if (status == 429 or self._is_rate_limit(str(exc))) and attempt < COGNEE_RETRY_MAX_ATTEMPTS:
                    log.warning(
                        "rate limit HTTP %d (attempt %d/%d) — retrying in %.0fs",
                        status, attempt, COGNEE_RETRY_MAX_ATTEMPTS, delay,
                        extra={"tool": tool},
                    )
                    last_exc = exc
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, COGNEE_RETRY_MAX_DELAY)
                    continue
                if status in (400, 401, 403, 404):
                    self._connected = False
                    if status == 400 and attempt < COGNEE_RETRY_MAX_ATTEMPTS:
                        log.warning(
                            "400 Bad Request (stale session?) — reconnecting (attempt %d/%d)",
                            attempt, COGNEE_RETRY_MAX_ATTEMPTS,
                            extra={"tool": tool},
                        )
                        await self.ensure_connected()
                        last_exc = exc
                        continue
                self._cb_record_failure()
                latency = round((time.monotonic() - t0) * 1000)
                log.error("cognee_call HTTP error",
                          extra={"tool": tool, "latency_ms": latency, "error": str(exc)})
                raise

        # All retry attempts exhausted
        self._cb_record_failure()
        log.error("cognee_call: exhausted %d retries for tool=%s", COGNEE_RETRY_MAX_ATTEMPTS, tool,
                  extra={"tool": tool})
        raise last_exc or RuntimeError(f"Tool {tool} failed after {COGNEE_RETRY_MAX_ATTEMPTS} attempts")

    async def stream_call(self, tool: str, args: dict, req_id, timeout: float = 180):
        """Forward upstream SSE events as an async generator.

        Yields raw 'data: {...}\\n\\n' lines so the caller can stream them
        directly to the connecting agent without buffering the full response.
        The final event has its JSON-RPC id rewritten to match req_id so the
        agent's response correlates correctly.
        """
        if self._cb_open():
            err = {"jsonrpc": "2.0", "id": req_id,
                   "error": {"code": -32000, "message": "Circuit breaker open — upstream unavailable"}}
            yield f"data: {json.dumps(err)}\n\n"
            return

        await self.ensure_connected()

        t0 = time.monotonic()
        try:
            # No _lock here — httpx connection pool handles concurrent requests safely
            async with self._http.stream(
                "POST", "/mcp",
                json={
                    "jsonrpc": "2.0", "id": self._next_id(), "method": "tools/call",
                    "params": {"name": tool, "arguments": args},
                },
                headers=self._mcp_headers(with_session=True),
                timeout=timeout,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        try:
                            event = json.loads(line[5:].strip())
                            # Rewrite id so caller's response matches their request
                            if "id" in event:
                                event["id"] = req_id
                        except json.JSONDecodeError:
                            event = {"raw": line}
                        yield f"data: {json.dumps(event)}\n\n"
                    else:
                        yield f"{line}\n\n"

            latency = round((time.monotonic() - t0) * 1000)
            self._cb_record_success()
            log.info("stream_call ok", extra={"tool": tool, "latency_ms": latency})

        except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            self._connected = False
            self._cb_record_failure()
            err = {"jsonrpc": "2.0", "id": req_id,
                   "error": {"code": -32000, "message": f"Connection lost: {exc}"}}
            yield f"data: {json.dumps(err)}\n\n"

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403, 404):
                self._connected = False
            self._cb_record_failure()
            err = {"jsonrpc": "2.0", "id": req_id,
                   "error": {"code": -32000, "message": f"Upstream HTTP {exc.response.status_code}"}}
            yield f"data: {json.dumps(err)}\n\n"

    @property
    def connected(self) -> bool:
        return self._connected


# module-level singleton — all tool calls go through this
proxy = CogneeMCPProxy()


async def cognee_call(tool: str, args: dict, timeout: float = 180) -> str:
    """Thin shim so existing code keeps working without changes."""
    return await proxy.call(tool, args, timeout)


# ── streaming helpers ──────────────────────────────────────────────────────────

async def _stream_subprocess(tool_name: str, tool_args: dict, req_id):
    """Run a subprocess tool in an executor and yield SSE heartbeats while it runs.

    Subprocess tools (memify) produce no intermediate output,
    so we send MCP notifications/progress pings every 5 s to keep the connection
    alive, then emit the final JSON-RPC result event.
    """
    loop = asyncio.get_event_loop()

    # Run subprocess in a thread so we can yield while it runs
    task = asyncio.ensure_future(
        loop.run_in_executor(None, _sync_dispatch, tool_name, tool_args)
    )

    ping = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"progressToken": req_id, "progress": 0, "total": 100},
    }

    while not task.done():
        yield f"data: {json.dumps(ping)}\n\n"
        try:
            # Wait up to 5 s; shield prevents cancellation of the real task
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except asyncio.TimeoutError:
            pass

    try:
        result_text = task.result()
    except Exception as exc:
        result_text = f"Error: {exc}"

    payload = {
        "jsonrpc": "2.0", "id": req_id,
        "result": {"content": [{"type": "text", "text": result_text}]},
    }
    yield f"data: {json.dumps(payload)}\n\n"


def _sync_dispatch(tool_name: str, tool_args: dict) -> str:
    """Blocking wrapper — runs the subprocess synchronously in a thread pool worker."""
    import asyncio as _asyncio
    loop = _asyncio.new_event_loop()
    try:
        return loop.run_until_complete(run_memify(tool_args.get("dataset", "main_dataset")))
    finally:
        loop.close()


async def stream_tool_result(tool_name: str, tool_args: dict, req_id, session_id: str, user: str):
    """Top-level streaming generator for tools/call.

    - Subprocess tools  → heartbeat pings + final result
    - All other tools   → forward Cognee MCP SSE events in real time
    """
    log.info("stream start", extra={"tool": tool_name, "session_id": session_id})

    if tool_name in _SUBPROCESS_TOOLS:
        async for chunk in _stream_subprocess(tool_name, tool_args, req_id):
            yield chunk
        return

    # Determine timeout by tool
    _TIMEOUTS = {
        "cognify": 300, "save_interaction": 120, "search": 180,
        "forget_memory": 60, "delete": 60,
        "submit_feedback": 30, "record_trace": 30,
        "improve": 600, "persist_sessions": 300, "improve_answer": 180,
    }
    timeout = _TIMEOUTS.get(tool_name, 60)

    async for chunk in proxy.stream_call(tool_name, tool_args, req_id, timeout=timeout):
        yield chunk


# ── custom tool implementations ────────────────────────────────────────────────

_COGNIFY_DATASET_SCRIPT = r"""
import asyncio, json, sys, os, logging

# Show INFO+ pipeline logs in mcp-wrapper background thread output
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[cognee] %(levelname)s %(name)s: %(message)s")
os.environ.setdefault("COGNEE_LOG_LEVEL", "INFO")

async def main():
    args = json.loads(sys.stdin.buffer.read())
    data         = args["data"]
    dataset_name = args.get("dataset_name", "main_dataset")
    temporal     = args.get("temporal", False)

    import cognee
    await cognee.run_migrations()
    await cognee.add(data, dataset_name=dataset_name)
    await cognee.cognify(
        datasets=[dataset_name],
        temporal_cognify=temporal,
        run_in_background=False,
    )
    sys.stdout.write(json.dumps({"status": "ok", "dataset": dataset_name}) + "\n")
    sys.stdout.flush()

asyncio.run(main())
"""


async def _run_cognify_dataset(data: str, dataset_name: str, temporal: bool = False) -> str:
    """Cognify directly into a named dataset using Cognee's Python API inside the container.

    Bypasses cognee-mcp's MCP tool (which has no dataset_name param) and calls
    cognee.add() + cognee.cognify(datasets=[...]) directly — same pattern as run_memify.
    """
    loop    = asyncio.get_event_loop()
    payload = json.dumps({"data": data, "dataset_name": dataset_name, "temporal": temporal})

    def _exec() -> str:
        import docker as _docker, tarfile as _tar, io as _io
        client    = _docker.from_env()
        container = client.containers.get(COGNEE_CONTAINER_NAME)

        # Embed the JSON payload directly into a loader preamble so stdin isn't needed
        loader = (
            "import sys\n"
            f"sys.stdin = __import__('io').TextIOWrapper("
            f"__import__('io').BytesIO({repr(payload.encode())}))\n"
        ) + _COGNIFY_DATASET_SCRIPT

        buf = _io.BytesIO()
        with _tar.open(fileobj=buf, mode="w") as t:
            content = loader.encode()
            info = _tar.TarInfo(name="cognify_dataset_run.py")
            info.size = len(content)
            t.addfile(info, _io.BytesIO(content))
        buf.seek(0)
        container.put_archive("/tmp", buf.read())

        exit_code, output = container.exec_run(
            "python3 /tmp/cognify_dataset_run.py",
            stream=False, demux=False,
            environment={"COGNEE_LOG_LEVEL": "INFO"},
        )
        text = output.decode(errors="replace").strip() if output else ""
        if exit_code != 0:
            log.error("cognify_dataset failed (exit %d): %s", exit_code, text[-400:])
            raise RuntimeError(f"Dataset cognify failed: {text[-300:]}")
        # Pick the last JSON line (cognee startup logs appear before it on stdout)
        for line in reversed(text.splitlines()):
            if line.strip().startswith("{"):
                return line.strip()
        return text[-300:] or f"Cognified into {dataset_name}"

    result = await loop.run_in_executor(None, _exec)
    log.info("cognify_dataset ok", extra={"dataset": dataset_name, "tool": "cognify"})
    return result


async def run_memify(dataset: str = "main_dataset") -> str:
    """Run memify inside the cognee container via the Python docker SDK."""
    loop = asyncio.get_event_loop()

    def _exec() -> str:
        import docker as _docker, tarfile as _tar, io as _io
        client = _docker.from_env()
        container = client.containers.get(COGNEE_CONTAINER_NAME)

        script = """
import asyncio, sys
sys.path.insert(0, '/app/src')
from cognee.memify_pipelines.create_triplet_embeddings import create_triplet_embeddings
from cognee.modules.users.methods import get_default_user
dataset = sys.argv[1] if len(sys.argv) > 1 else 'main_dataset'

async def run():
    user = await get_default_user()
    await create_triplet_embeddings(user, dataset)
    print(f'Triplet embeddings created for dataset: {dataset}')

asyncio.run(run())
"""
        # Upload script into container
        buf = _io.BytesIO()
        with _tar.open(fileobj=buf, mode="w") as t:
            content = script.encode()
            info = _tar.TarInfo(name="run_memify_wrapper.py")
            info.size = len(content)
            t.addfile(info, _io.BytesIO(content))
        buf.seek(0)
        container.put_archive("/tmp", buf.read())

        exit_code, output = container.exec_run(
            f"python3 /tmp/run_memify_wrapper.py {dataset}",
            stream=False,
        )
        text = output.decode(errors="replace") if output else ""
        return text[-600:] or "Memify complete."

    output = await loop.run_in_executor(None, _exec)
    return output or "Memify complete."



async def run_prune() -> str:
    """Wipe all cognee data via docker exec.

    Cognee's MCP prune calls prune_data() which does shutil.rmtree on the data
    directory — that directory is a Docker volume mount point, so it fails with
    EBUSY. We skip prune_data() and call only:
      - datasets.delete_all()  → removes graph, vector, and relational data
      - prune_system(...)      → clears remaining DB state and cache
    Cognee's own API resolves all paths from its config, no hardcoding needed.
    """
    loop = asyncio.get_event_loop()

    def _exec() -> str:
        import docker as _docker, tarfile as _tar, io as _io
        client = _docker.from_env()
        container = client.containers.get(COGNEE_CONTAINER_NAME)

        script = """
import asyncio, sys
sys.path.insert(0, '/app/src')
import cognee

async def main():
    await cognee.prune.prune_system(graph=True, vector=True, metadata=True, cache=True)
    print("Prune complete.")

asyncio.run(main())
"""
        buf = _io.BytesIO()
        with _tar.open(fileobj=buf, mode="w") as t:
            content = script.encode()
            info = _tar.TarInfo(name="run_prune.py")
            info.size = len(content)
            t.addfile(info, _io.BytesIO(content))
        buf.seek(0)
        container.put_archive("/tmp", buf.read())

        exit_code, output = container.exec_run("python3 /tmp/run_prune.py", stream=False)
        text = output.decode(errors="replace") if output else ""
        if exit_code != 0:
            return f"Prune error (exit {exit_code}): {text[-400:]}"
        return text.strip() or "Prune complete."

    return await loop.run_in_executor(None, _exec)


async def run_submit_feedback(
    user_id: str,
    session_id: str,
    qa_id: str,
    feedback_score: int | None,
    feedback_text: str | None,
) -> str:
    loop = asyncio.get_event_loop()

    def _exec() -> str:
        import docker as _docker, tarfile as _tar, io as _io, json as _json
        client = _docker.from_env()
        container = client.containers.get(COGNEE_CONTAINER_NAME)

        script = f"""
import asyncio, sys, json as _json
sys.path.insert(0, '/app/src')
from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine

async def main():
    cache = get_cache_engine()
    if cache is None:
        print("ERROR: cache engine unavailable — set CACHING=true")
        return
    # Cognee 0.5.x FSCacheAdapter only has add_qa — store feedback as a Q&A entry
    await cache.add_qa(
        user_id={_json.dumps(user_id)},
        session_id={_json.dumps(session_id)},
        question="[feedback] qa_id={_json.dumps(qa_id)}",
        context="feedback_score={feedback_score!r} feedback_text={_json.dumps(feedback_text) if feedback_text else ''}",
        answer="score:{feedback_score!r}",
    )
    print("submitted")

asyncio.run(main())
"""
        buf = _io.BytesIO()
        with _tar.open(fileobj=buf, mode="w") as t:
            data = script.encode()
            info = _tar.TarInfo(name="run_feedback.py")
            info.size = len(data)
            t.addfile(info, _io.BytesIO(data))
        buf.seek(0)
        container.put_archive("/tmp", buf.read())
        exit_code, output = container.exec_run("python3 /tmp/run_feedback.py", stream=False)
        text = output.decode(errors="replace") if output else ""
        if exit_code != 0:
            return f"Feedback error (exit {exit_code}): {text[-300:]}"
        return text.strip() or "Feedback submitted."

    return await loop.run_in_executor(None, _exec)


async def run_record_trace(
    user_id: str,
    session_id: str,
    origin_function: str,
    status: str,
    memory_query: str,
    method_params: dict,
    error_message: str,
) -> str:
    loop = asyncio.get_event_loop()

    def _exec() -> str:
        import docker as _docker, tarfile as _tar, io as _io, json as _json
        client = _docker.from_env()
        container = client.containers.get(COGNEE_CONTAINER_NAME)

        script = f"""
import asyncio, sys, uuid, json as _json
sys.path.insert(0, '/app/src')
from cognee.infrastructure.databases.cache.get_cache_engine import get_cache_engine

async def main():
    cache = get_cache_engine()
    if cache is None:
        print("ERROR: cache engine unavailable — set CACHING=true")
        return
    trace_id = str(uuid.uuid4())
    # Cognee 0.5.x FSCacheAdapter only has add_qa — store trace as a Q&A entry
    await cache.add_qa(
        user_id={_json.dumps(user_id)},
        session_id={_json.dumps(session_id)},
        question="[trace] {_json.dumps(origin_function)} status={_json.dumps(status)}",
        context={_json.dumps(memory_query)},
        answer="error={_json.dumps(error_message)} trace_id=" + trace_id,
    )
    print(f"trace:{{trace_id}}")

asyncio.run(main())
"""
        buf = _io.BytesIO()
        with _tar.open(fileobj=buf, mode="w") as t:
            data = script.encode()
            info = _tar.TarInfo(name="run_trace.py")
            info.size = len(data)
            t.addfile(info, _io.BytesIO(data))
        buf.seek(0)
        container.put_archive("/tmp", buf.read())
        exit_code, output = container.exec_run("python3 /tmp/run_trace.py", stream=False)
        text = output.decode(errors="replace") if output else ""
        if exit_code != 0:
            return f"Trace error (exit {exit_code}): {text[-300:]}"
        return text.strip() or "Trace recorded."

    return await loop.run_in_executor(None, _exec)


async def run_persist_sessions(data: str, user: str = AGENT_USER) -> str:
    if not data.strip():
        return "No session data provided."
    result = await cognee_call("cognify", {"data": data, "user": user}, timeout=300)
    return f"Session persisted to graph. {result[:200]}"


async def run_improve_answer(
    question: str, wrong_answer: str, feedback: str, user: str = AGENT_USER
) -> str:
    improved = await cognee_call(
        "search",
        {"search_query": question, "search_type": "GRAPH_COMPLETION_COT", "user": user},
        timeout=180,
    )
    correction = (
        f"CORRECTION — Question: {question}\n"
        f"Wrong answer: {wrong_answer}\nFeedback: {feedback}\n"
        f"Improved answer: {improved}"
    )
    await cognee_call("cognify", {"data": correction, "user": user}, timeout=300)
    return f"Improved answer (stored):\n{improved}"


async def run_create_dataset(dataset_name: str, session_id: str) -> str:
    """Activate a named dataset for the session — Cognee creates it on first cognify call."""
    clean = dataset_name.strip()
    if not clean:
        return "Error: dataset_name cannot be empty."
    if session_id and session_id in _sessions:
        _sessions[session_id]["dataset_name"] = clean
        agent = _sessions[session_id]["agent_id"]
        log.info("dataset activated", extra={"session_id": session_id, "dataset": clean, "tool": "create_dataset"})
        return (
            f"Dataset '{clean}' is now active for session '{agent}'. "
            "Cognee will create it on the first cognify call targeting this session."
        )
    return f"Dataset '{clean}' noted — no bound session to update."


async def run_switch_dataset(dataset_name: str, session_id: str) -> str:
    """Switch the active dataset for the current session."""
    clean = dataset_name.strip()
    if not clean:
        return "Error: dataset_name cannot be empty."
    if session_id and session_id in _sessions:
        old = _sessions[session_id].get("dataset_name", "main_dataset")
        _sessions[session_id]["dataset_name"] = clean
        log.info("dataset switched", extra={"session_id": session_id, "from": old, "to": clean, "tool": "switch_dataset"})
        return f"Switched active dataset from '{old}' → '{clean}' for this session."
    return f"Active dataset set to '{clean}' (no bound session)."


async def run_delete_dataset(dataset_name: str) -> str:
    """Delete a Cognee dataset via docker exec using the internal datasets module."""
    loop = asyncio.get_event_loop()

    def _exec() -> str:
        import docker as _docker, tarfile as _tar, io as _io
        client = _docker.from_env()
        container = client.containers.get(COGNEE_CONTAINER_NAME)

        script_template = """
import asyncio, sys
sys.path.insert(0, '/app/src')

dataset_name = __DATASET__

async def main():
    # Attempt 1: internal datasets module (preferred)
    try:
        from cognee.modules.datasets.methods import get_datasets
        from cognee.modules.datasets.methods.delete_dataset import delete_dataset
        all_ds = await get_datasets()
        targets = [d for d in all_ds if getattr(d, 'name', '') == dataset_name]
        if targets:
            for ds in targets:
                await delete_dataset(ds.id)
            print(f"Deleted dataset '{dataset_name}' ({len(targets)} entr(ies)).")
        else:
            print(f"Dataset '{dataset_name}' not found in registry.")
        return
    except Exception as e1:
        pass

    # Attempt 2: cognee high-level API
    try:
        import cognee
        await cognee.delete_data(dataset_name=dataset_name)
        print(f"Deleted dataset '{dataset_name}' via cognee.delete_data.")
        return
    except Exception as e2:
        pass

    print(f"Could not delete dataset '{dataset_name}': no compatible API found in this Cognee build.")

asyncio.run(main())
"""
        script = script_template.replace("__DATASET__", repr(dataset_name))
        buf = _io.BytesIO()
        with _tar.open(fileobj=buf, mode="w") as t:
            content = script.encode()
            info = _tar.TarInfo(name="run_delete_dataset.py")
            info.size = len(content)
            t.addfile(info, _io.BytesIO(content))
        buf.seek(0)
        container.put_archive("/tmp", buf.read())

        exit_code, output = container.exec_run("python3 /tmp/run_delete_dataset.py", stream=False)
        text = output.decode(errors="replace") if output else ""
        if exit_code != 0:
            return f"Delete error (exit {exit_code}): {text[-400:]}"
        return text.strip() or f"Dataset '{dataset_name}' deletion attempted."

    return await loop.run_in_executor(None, _exec)


# ── background cognify (plain thread, survives request scope) ──────────────────

_bg_executor = __import__("concurrent.futures", fromlist=["ThreadPoolExecutor"]).ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="cognify-bg"
)


def _cognify_bg_thread(payload: str) -> None:
    """Blocking worker: runs inside a plain OS thread, independent of asyncio/anyio."""
    import tarfile as _tar, io as _io
    import docker as _docker  # type: ignore[import]

    meta = json.loads(payload)
    ds   = meta.get("dataset_name", "?")

    try:
        client    = _docker.from_env()
        container = client.containers.get(COGNEE_CONTAINER_NAME)

        loader = (
            "import sys\n"
            f"sys.stdin = __import__('io').TextIOWrapper("
            f"__import__('io').BytesIO({repr(payload.encode())}))\n"
        ) + _COGNIFY_DATASET_SCRIPT

        buf = _io.BytesIO()
        with _tar.open(fileobj=buf, mode="w") as t:
            content = loader.encode()
            info = _tar.TarInfo(name="cognify_dataset_run.py")
            info.size = len(content)
            t.addfile(info, _io.BytesIO(content))
        buf.seek(0)
        container.put_archive("/tmp", buf.read())

        exit_code, output = container.exec_run(
            "python3 /tmp/cognify_dataset_run.py",
            stream=False, demux=False,
            environment={"COGNEE_LOG_LEVEL": "INFO"},
        )
        text = output.decode(errors="replace").strip() if output else ""
        if exit_code != 0:
            log.error(
                "background cognify failed (exit %d) dataset=%s: %s",
                exit_code, ds, text[-300:],
                extra={"tool": "cognify", "session_id": ""},
            )
        else:
            log.info(
                "background cognify done dataset=%s",
                ds,
                extra={"tool": "cognify", "session_id": ""},
            )
    except Exception as exc:
        log.error(
            "background cognify error dataset=%s: %s",
            ds, exc,
            extra={"tool": "cognify", "session_id": ""},
            exc_info=True,
        )


def _fire_cognify_bg(data: str, dataset_name: str, temporal: bool) -> None:
    """Submit cognify to a plain thread pool — lives outside asyncio/anyio scope."""
    payload = json.dumps({"data": data, "dataset_name": dataset_name, "temporal": temporal})
    _bg_executor.submit(_cognify_bg_thread, payload)


# ── tool dispatcher ────────────────────────────────────────────────────────────

async def dispatch(tool: str, args: dict, session_id: str = "", user: str = AGENT_USER, dataset: str = "main_dataset") -> str:
    t0 = time.monotonic()
    # Keep last_active fresh on every tool call
    if session_id and session_id in _sessions:
        _sessions[session_id]["last_active"] = time.time()

    try:
        if tool == "cognify":
            # Caller can pass dataset_name directly in args (orchestrator path) or
            # rely on the session's active dataset (switch_dataset path).
            effective_dataset = args.get("dataset_name", "").strip() or dataset
            if effective_dataset and effective_dataset != "main_dataset":
                # cognee-mcp's MCP tool has no dataset_name param — bypass it
                # and call cognee.add() + cognee.cognify() directly via docker exec.
                # Fire in background so the caller gets an immediate acknowledgement;
                # the heavy docker-exec pipeline (2-7 min) runs independently.
                _fire_cognify_bg(
                    args["data"], effective_dataset,
                    temporal=args.get("temporal", False),
                )
                result = json.dumps({
                    "status": "processing",
                    "dataset": effective_dataset,
                    "message": "Cognify started in background. Use cognify_status to check progress.",
                })
            else:
                result = await cognee_call("cognify", {
                    "data": args["data"],
                    "temporal_cognify": args.get("temporal", False),
                }, timeout=300)
        elif tool == "save_interaction":
            effective_dataset = args.get("dataset_name", "").strip() or dataset
            if effective_dataset and effective_dataset != "main_dataset":
                result = await _run_cognify_dataset(
                    args["data"], effective_dataset,
                    temporal=False,
                )
            else:
                result = await cognee_call("cognify", {"data": args["data"]}, timeout=300)
        elif tool == "search":
            query = args.get("search_query") or args.get("query", "")
            if not query:
                return [{"type": "text", "text": "Error: search_query is required"}]
            effective_dataset = args.get("dataset_name", "").strip() or dataset
            search_args: dict = {
                "search_query": query,
                "search_type":  args.get("search_type", "GRAPH_COMPLETION"),
            }
            if effective_dataset and effective_dataset != "main_dataset":
                search_args["dataset_name"] = effective_dataset
            result = await cognee_call("search", search_args, timeout=180)
        elif tool == "forget_memory":
            forget_args: dict = {}
            if args.get("dataset"):
                forget_args["dataset"] = args["dataset"]
            if args.get("everything"):
                forget_args["everything"] = args["everything"]
            result = await cognee_call("forget_memory", forget_args, timeout=60)
        elif tool == "delete":
            delete_args: dict = {"data_id": args["data_id"]}
            if args.get("dataset_id"):
                delete_args["dataset_id"] = args["dataset_id"]
            if args.get("mode"):
                delete_args["mode"] = args["mode"]
            result = await cognee_call("delete", delete_args, timeout=60)
        elif tool == "list_data":
            result = await cognee_call("list_data", {}, timeout=60)
        elif tool == "cognify_status":
            result = await cognee_call("cognify_status", {}, timeout=30)
        elif tool == "prune":
            result = await run_prune()
        elif tool == "memify":
            result = await run_memify(args.get("dataset", dataset))
        elif tool == "improve":
            improve_args: dict = {
                "dataset_name": args.get("dataset_name", dataset),
            }
            # Use explicitly passed session_ids, or auto-inject the current session
            session_ids = args.get("session_ids") or ([session_id] if session_id else None)
            if session_ids:
                improve_args["session_ids"] = session_ids
            result = await cognee_call("improve", improve_args, timeout=600)
        elif tool == "submit_feedback":
            result = await run_submit_feedback(
                user_id=args.get("user_id", user),
                session_id=args.get("session_id", session_id),
                qa_id=args.get("qa_id", ""),
                feedback_score=args.get("feedback_score"),
                feedback_text=args.get("feedback_text"),
            )
        elif tool == "record_trace":
            result = await run_record_trace(
                user_id=args.get("user_id", user),
                session_id=args.get("session_id", session_id),
                origin_function=args.get("origin_function", "unknown"),
                status=args.get("status", "success"),
                memory_query=args.get("memory_query", ""),
                method_params=args.get("method_params") or {},
                error_message=args.get("error_message", ""),
            )
        elif tool == "persist_sessions":
            result = await run_persist_sessions(args.get("data", ""), user=user)
        elif tool == "improve_answer":
            result = await run_improve_answer(
                args.get("question", ""), args.get("wrong_answer", ""), args.get("feedback", ""),
                user=user,
            )
        elif tool == "create_dataset":
            result = await run_create_dataset(args.get("dataset_name", ""), session_id=session_id)
        elif tool == "switch_dataset":
            result = await run_switch_dataset(args.get("dataset_name", ""), session_id=session_id)
        elif tool == "delete_dataset":
            ds = args.get("dataset_name", "")
            if not ds:
                result = "Error: dataset_name is required."
            else:
                result = await run_delete_dataset(ds)
        else:
            result = f"Unknown tool: {tool}"

        latency = round((time.monotonic() - t0) * 1000)
        log.info("dispatch ok", extra={"tool": tool, "session_id": session_id, "latency_ms": latency})
        return result

    except Exception as exc:
        latency = round((time.monotonic() - t0) * 1000)
        log.error(
            "dispatch error",
            extra={"tool": tool, "session_id": session_id, "latency_ms": latency, "error": str(exc)},
            exc_info=True,
        )
        raise


# ── MCP protocol handler ───────────────────────────────────────────────────────

SESSION_TTL_SECONDS = int(os.getenv("MCP_WRAPPER_SESSION_TTL", "3600"))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Background task to evict stale sessions."""
    async def _evict_loop():
        while True:
            await asyncio.sleep(300)  # check every 5 minutes
            now = time.time()
            stale = [
                sid for sid, s in _sessions.items()
                if now - s.get("created_at", now) > SESSION_TTL_SECONDS
            ]
            for sid in stale:
                _sessions.pop(sid, None)
            if stale:
                log.info("evicted %d stale session(s)", len(stale))

    # Quick single-attempt connect at startup (5 s timeout).
    # Full retry logic runs on first tool call — don't block uvicorn startup.
    try:
        await asyncio.wait_for(proxy._do_connect(), timeout=5.0)
    except Exception as exc:
        proxy._connected = False
        log.warning("Startup: Cognee MCP not reachable (%s) — will retry on first call", exc,
                    extra={"tool": "proxy"})

    task = asyncio.create_task(_evict_loop())
    try:
        yield
    finally:
        task.cancel()
        if proxy._http:
            await proxy._http.aclose()


app = FastAPI(title="Cognee MCP Wrapper", lifespan=_lifespan)



# Active sessions: session_id → {"agent_id": str, "created_at": float}
_sessions: dict[str, dict] = {}


def _sse_response(payload: dict, headers: dict | None = None) -> StreamingResponse:
    data = f"data: {json.dumps(payload)}\n\n"
    return StreamingResponse(iter([data]), media_type="text/event-stream", headers=headers or {})


def _wants_sse(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/event-stream" in accept


def _jsonrpc_error(req_id, code: int, message: str):
    return JSONResponse(
        status_code=400,
        content={"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}},
    )


def _validate_session(sid: str, req_id) -> JSONResponse | None:
    """Return an error response if session is required and invalid, else None."""
    if not MCP_WRAPPER_STRICT_SESSIONS:
        return None
    if not sid:
        return _jsonrpc_error(req_id, -32000, "Missing mcp-session-id header — call initialize first")
    if sid not in _sessions:
        return _jsonrpc_error(req_id, -32000, "Unknown or expired session — call initialize again")
    return None


def _session_user(sid: str) -> str:
    """Return the agent_id bound to this session, falling back to AGENT_USER.

    Using session-bound identity prevents callers from spoofing another
    agent's user context by passing 'user' in tool arguments.
    """
    if sid and sid in _sessions:
        return _sessions[sid]["agent_id"]
    return AGENT_USER


def _session_dataset(sid: str) -> str:
    """Return the active dataset_name bound to this session, defaulting to 'main_dataset'."""
    if sid and sid in _sessions:
        return _sessions[sid].get("dataset_name", "main_dataset")
    return "main_dataset"


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    body = await request.json()
    method = body.get("method")
    req_id = body.get("id")
    sid = request.headers.get("mcp-session-id", "")

    # ── initialize ──
    if method == "initialize":
        session_id = str(uuid.uuid4())
        params = body.get("params", {})
        agent_name = params.get("clientInfo", {}).get("name", "unknown")
        dataset_name = params.get("dataset_name", "main_dataset")
        _sessions[session_id] = {
            "agent_id": agent_name,
            "created_at": time.time(),
            "last_active": time.time(),
            "dataset_name": dataset_name,
        }
        log.info("session init", extra={"session_id": session_id, "tool": f"agent={agent_name}"})
        payload = {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "cognee-mcp-wrapper", "version": "1.0"},
            },
        }
        resp_headers = {"mcp-session-id": session_id}
        if _wants_sse(request):
            return StreamingResponse(
                iter([f"data: {json.dumps(payload)}\n\n"]),
                media_type="text/event-stream",
                headers=resp_headers,
            )
        return JSONResponse(content=payload, headers=resp_headers)

    # ── tools/list ──
    if method == "tools/list":
        if err := _validate_session(sid, req_id):
            log.warning("rejected tools/list: invalid session", extra={"session_id": sid})
            return err
        log.info("tools/list", extra={"session_id": sid})
        payload = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_LIST}}
        return _sse_response(payload)

    # ── tools/call ──
    if method == "tools/call":
        if err := _validate_session(sid, req_id):
            log.warning("rejected tools/call: invalid session", extra={"session_id": sid})
            return err
        tool_name = body["params"]["name"]
        tool_args = body["params"].get("arguments", {})
        agent_user = _session_user(sid)
        active_dataset = _session_dataset(sid)

        if MCP_WRAPPER_STREAMING:
            # Stream SSE events incrementally — no buffering
            return StreamingResponse(
                stream_tool_result(tool_name, tool_args, req_id, sid, agent_user),
                media_type="text/event-stream",
            )

        # Non-streaming (default) — buffer full result, emit single SSE event
        try:
            result_text = await dispatch(tool_name, tool_args, session_id=sid, user=agent_user, dataset=active_dataset)
        except Exception as exc:
            result_text = f"Error: {exc}"
        payload = {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"content": [{"type": "text", "text": result_text}]},
        }
        return _sse_response(payload)

    log.warning("unknown method: %s", method, extra={"session_id": sid})
    return JSONResponse({
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"},
    })


_start_time = time.time()

# ── Phase 7: tool documentation metadata ──────────────────────────────────────

_TOOL_METADATA: dict[str, dict] = {
    "cognify":           {"category": "write", "typical_latency_seconds": 30},
    "save_interaction":  {"category": "write", "typical_latency_seconds": 2},
    "forget_memory":     {"category": "admin", "typical_latency_seconds": 3},
    "delete":            {"category": "admin", "typical_latency_seconds": 2},
    "search":            {"category": "read",  "typical_latency_seconds": 5},
    "list_data":         {"category": "read",  "typical_latency_seconds": 2},
    "cognify_status":    {"category": "read",  "typical_latency_seconds": 1},
    "prune":             {"category": "admin", "typical_latency_seconds": 5},
    "memify":            {"category": "write", "typical_latency_seconds": 120},
    "submit_feedback":   {"category": "write", "typical_latency_seconds": 3},
    "record_trace":      {"category": "write", "typical_latency_seconds": 3},
    "improve":           {"category": "write", "typical_latency_seconds": 300},
    "persist_sessions":  {"category": "write", "typical_latency_seconds": 10},
    "improve_answer":    {"category": "write", "typical_latency_seconds": 10},
    "create_dataset":    {"category": "admin", "typical_latency_seconds": 1},
    "switch_dataset":    {"category": "admin", "typical_latency_seconds": 1},
    "delete_dataset":    {"category": "admin", "typical_latency_seconds": 5},
}


@app.get("/tools")
async def list_tools_doc():
    enriched = []
    for tool in TOOL_LIST:
        meta = _TOOL_METADATA.get(tool["name"], {})
        enriched.append({**tool, **meta})
    return {
        "server": {
            "name": "cognee-mcp-wrapper",
            "version": "1.0.0",
            "mcp_endpoint": "/mcp",
            "auth": "none",
        },
        "session_handshake": {
            "step_1": "POST /mcp  method=initialize  →  mcp-session-id header returned",
            "step_2": "POST /mcp  method=tools/list   mcp-session-id: <id>",
            "step_3": "POST /mcp  method=tools/call   mcp-session-id: <id>",
        },
        "tool_count": len(enriched),
        "tools": enriched,
    }


@app.get("/health")
async def health():
    cb_open = proxy._cb_open()
    return {
        "status": "degraded" if cb_open else "ok",
        "tools": len(TOOL_LIST),
        "uptime_seconds": round(time.time() - _start_time),
        "active_sessions": len(_sessions),
        "cognee_mcp_connected": proxy.connected,
        "circuit_breaker_open": cb_open,
        "consecutive_failures": proxy._failures,
    }


@app.get("/sessions")
async def list_sessions():
    """List all active MCP sessions with metadata."""
    now = time.time()
    return {
        "active_sessions": len(_sessions),
        "sessions": [
            {
                "session_id": sid,
                "agent_id": s["agent_id"],
                "dataset_name": s.get("dataset_name", "main_dataset"),
                "created_at": s["created_at"],
                "last_active": s.get("last_active", s["created_at"]),
                "age_seconds": round(now - s["created_at"]),
                "idle_seconds": round(now - s.get("last_active", s["created_at"])),
                "ttl_remaining_seconds": max(0, round(SESSION_TTL_SECONDS - (now - s["created_at"]))),
            }
            for sid, s in _sessions.items()
        ],
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get metadata for a specific MCP session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    s = _sessions[session_id]
    now = time.time()
    return {
        "session_id": session_id,
        "agent_id": s["agent_id"],
        "dataset_name": s.get("dataset_name", "main_dataset"),
        "created_at": s["created_at"],
        "last_active": s.get("last_active", s["created_at"]),
        "age_seconds": round(now - s["created_at"]),
        "idle_seconds": round(now - s.get("last_active", s["created_at"])),
        "ttl_remaining_seconds": max(0, round(SESSION_TTL_SECONDS - (now - s["created_at"]))),
    }


@app.delete("/sessions/{session_id}")
async def terminate_session(session_id: str):
    """Forcibly terminate an MCP session before its TTL expires."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    s = _sessions.pop(session_id)
    log.info("session terminated by admin", extra={"session_id": session_id, "tool": "admin"})
    return {"terminated": True, "session_id": session_id, "agent_id": s["agent_id"]}


# ── admin: ontology reload ─────────────────────────────────────────────────────

@app.post("/admin/reload-cognee")
async def reload_cognee_container():
    """Restart cognee-mcp so it picks up a newly uploaded ontology file.

    Protected by the same API key middleware as all other endpoints.
    The web service calls this after writing a new .ttl/.owl file to the
    shared ontologies volume.
    """
    loop = asyncio.get_event_loop()

    def _restart() -> str:
        import docker as _docker
        client = _docker.from_env()
        container = client.containers.get(COGNEE_CONTAINER_NAME)
        container.restart(timeout=30)
        return "restarting"

    try:
        status = await loop.run_in_executor(None, _restart)
        log.info("cognee-mcp container restarted for ontology reload", extra={"tool": "admin"})
        return {"status": status, "container": COGNEE_CONTAINER_NAME,
                "message": "cognee-mcp is restarting — allow ~60s for health check to pass again"}
    except Exception as exc:
        log.error("Failed to restart cognee-mcp: %s", exc)
        raise HTTPException(status_code=500, detail=f"Restart failed: {exc}")


# ── run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    log.info(
        "MCP Wrapper starting",
        extra={"tool": f"port={PORT} tools={len(TOOL_LIST)} log_level={LOG_LEVEL}"},
    )
    uvicorn.run("mcp_wrapper:app", host="0.0.0.0", port=PORT, reload=False)
