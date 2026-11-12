"""
Async client for mcp-wrapper:8002.
Handles MCP session handshake, tool calls, SSE response parsing, and circuit breaking.
"""
import asyncio
import json
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_MCP_WRAPPER_URL = os.getenv("MCP_WRAPPER_URL", "http://mcp-wrapper:8002")

# Circuit breaker thresholds
_CB_THRESHOLD = 3    # failures before opening
_CB_TIMEOUT   = 60.0 # seconds before half-open retry


class WrapperClient:
    """Singleton async client that holds one MCP session with mcp-wrapper."""

    def __init__(self, base_url: str = _MCP_WRAPPER_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._session_id: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._request_counter = 0
        self._init_lock: asyncio.Lock | None = None  # created lazily (needs running loop)

        # Circuit breaker state
        self._cb_failures = 0
        self._cb_opened_at: float = 0.0

    # ── Circuit breaker ────────────────────────────────────────────────────

    def _cb_is_open(self) -> bool:
        if self._cb_failures < _CB_THRESHOLD:
            return False
        if time.monotonic() - self._cb_opened_at > _CB_TIMEOUT:
            # Half-open: allow one probe
            return False
        return True

    def _cb_record_success(self) -> None:
        self._cb_failures = 0

    def _cb_record_failure(self) -> None:
        self._cb_failures += 1
        if self._cb_failures >= _CB_THRESHOLD:
            self._cb_opened_at = time.monotonic()

    # ── HTTP client ────────────────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(180.0),
                headers={"Accept": "application/json, text/event-stream"},
            )
        return self._client

    def _next_id(self) -> str:
        self._request_counter += 1
        return f"orch-{self._request_counter}"

    def _session_headers(self) -> dict:
        h: dict = {"Content-Type": "application/json"}
        if self._session_id:
            h["mcp-session-id"] = self._session_id
        return h

    # ── MCP session handshake ──────────────────────────────────────────────

    async def ensure_session(self) -> None:
        """Initialize MCP session with mcp-wrapper if not already done."""
        if self._session_id:
            return
        # Lazily create the lock inside a running event loop
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._session_id:   # double-checked after acquiring the lock
                return
            client = self._get_client()
            payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "orchestrator", "version": "1.0"},
                },
            }
            try:
                resp = await client.post("/mcp", json=payload, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                self._session_id = resp.headers.get("mcp-session-id")
                if not self._session_id:
                    raise RuntimeError("mcp-wrapper did not return mcp-session-id")

                # Required handshake step
                await client.post("/mcp", json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }, headers=self._session_headers())

                logger.info("MCP session established: %s", self._session_id)
                self._cb_record_success()
            except Exception as exc:
                self._cb_record_failure()
                raise RuntimeError(f"Failed to initialize MCP session: {exc}") from exc

    # ── SSE response parser ────────────────────────────────────────────────

    def _parse_sse(self, text: str) -> str:
        """Extract text content from SSE response lines."""
        for raw in text.splitlines():
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            candidate = line[5:].strip()
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if "error" in data:
                err = data["error"]
                raise RuntimeError(f"MCP tool error: {err.get('message', err)}")
            content = data.get("result", {}).get("content", [])
            parts = [item.get("text", "") for item in content if item.get("type") == "text"]
            return " ".join(parts)
        return ""

    # ── Tool call ──────────────────────────────────────────────────────────

    async def call(self, tool: str, args: dict[str, Any], timeout: float = 180) -> str:
        """Call a tool on mcp-wrapper, return result text.

        Retries once on 400 (stale/expired session) after re-initializing.
        """
        if self._cb_is_open():
            raise RuntimeError("circuit_breaker_open: mcp-wrapper unreachable")

        for attempt in range(2):
            await self.ensure_session()
            client = self._get_client()

            payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": tool, "arguments": args},
            }
            try:
                resp = await client.post(
                    "/mcp",
                    json=payload,
                    headers=self._session_headers(),
                    timeout=timeout,
                )
                # 400 means session expired/unknown — reset and retry once
                if resp.status_code == 400 and attempt == 0:
                    logger.warning(
                        "mcp-wrapper returned 400 for tool '%s' (stale session?) — re-initializing",
                        tool,
                    )
                    self._session_id = None
                    continue
                resp.raise_for_status()
                result = self._parse_sse(resp.text)
                self._cb_record_success()
                return result
            except RuntimeError:
                raise
            except Exception as exc:
                self._cb_record_failure()
                self._session_id = None
                raise RuntimeError(f"mcp-wrapper call '{tool}' failed: {exc}") from exc

        raise RuntimeError(f"mcp-wrapper call '{tool}' failed after session refresh")

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Module-level singleton — shared across all requests
wrapper = WrapperClient()
