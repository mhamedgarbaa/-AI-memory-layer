"""
LangChain agent connected to Cognee Memory Layer via MCP Wrapper (:8002).

Acts as the memory interface for the enterprise brain — full temporal knowledge
graph access (store, recall, feedback, trace, improve) via a ReAct loop.

Install deps:
    pip install langchain-mcp-adapters langchain-groq langgraph python-dotenv httpx

Run:
    python langchain_agent.py
"""

import asyncio
import os
import uuid
import warnings
from datetime import datetime

import httpx
from dotenv import load_dotenv
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
MCP_URL      = os.getenv("ORCHESTRATOR_URL", "http://localhost:8003") + "/mcp"
GROQ_API_KEY = os.getenv("GROQ_API_KEY",  "")
GROQ_MODEL   = os.getenv("GROQ_MODEL",    "meta-llama/llama-4-scout-17b-16e-instruct")
USER_ID      = os.getenv("AGENT_USER_ID", "ceo")

SKIP_TRACE: set = set()  # orchestrator has no trace tool

SYSTEM_PROMPT = f"""You are the Enterprise Brain — temporal knowledge assistant for CEO/Admin use.
You are connected to a Cognee knowledge graph via the Orchestrator (4 tools).
Today: {datetime.now().strftime("%Y-%m-%d")}  |  User: {USER_ID}

## MANDATORY RULES

1. **Search before answering** — Always call `search` before answering any factual question.
   Never answer from your own training knowledge alone.

2. **Store important information** — When the user shares facts, decisions, or context:
   call `save`. The orchestrator auto-classifies the content into the correct typed dataset
   (policy, regulation, employees, market, workflow, contracts, reports).
   Use `hint` to override: save(data="...", hint="market").

3. **Improve** — At the end of a session call `improve` to consolidate learnings into the graph.

4. **Be concise** — 2–3 sentences. State whether the answer came from memory or your own knowledge.

## TOOL GUIDE

| Intent | Tool | Notes |
|--------|------|-------|
| Answer any factual question | `search` | auto-routes to best search strategy |
| Time-sensitive / "when" query | `search` | hint="temporal" to force TEMPORAL search |
| Store a document or fact | `save` | auto-classified; use hint= to override dataset |
| Consolidate session learnings | `improve` | run at end of session or after corrections |
| Remove wrong or outdated info | `delete` | target= "all" / dataset name / item UUID |
"""


# ─── Background MCP call ──────────────────────────────────────────────────────

async def _bg_call(http: httpx.AsyncClient, session_id: str, tool: str, args: dict) -> None:
    """Fire-and-forget MCP tool call on the background session."""
    try:
        await http.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/call",
                "params": {"name": tool, "arguments": args},
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": session_id,
            },
            timeout=30,
        )
    except Exception:
        pass


# ─── Observability callback ───────────────────────────────────────────────────

class CogneeObservabilityCallback(AsyncCallbackHandler):
    """
    Auto-records a Cognee trace after every MCP tool call and refreshes
    the graph visualization after write operations.
    Uses LangChain's AsyncCallbackHandler — no monkey-patching required.
    """

    def __init__(self, http: httpx.AsyncClient, session_id: str, user_id: str) -> None:
        self.http       = http
        self.session_id = session_id
        self.user_id    = user_id
        self._pending: dict[str, dict] = {}

    async def on_tool_start(
        self, serialized: dict, input_str: str, *, run_id, **_kwargs
    ) -> None:
        self._pending[str(run_id)] = {
            "tool":  serialized.get("name", "unknown"),
            "input": str(input_str)[:500],
        }

    async def on_tool_end(self, _output: str, *, run_id, **_kwargs) -> None:
        info      = self._pending.pop(str(run_id), {})
        tool_name = info.get("tool", "unknown")
        if tool_name in SKIP_TRACE:
            return
        asyncio.create_task(_bg_call(self.http, self.session_id, "record_trace", {
            "user_id":         self.user_id,
            "session_id":      self.session_id,
            "origin_function": tool_name,
            "status":          "success",
            "memory_query":    info.get("input", ""),
            "error_message":   "",
        }))

    async def on_tool_error(self, error: BaseException, *, run_id, **_kwargs) -> None:
        info      = self._pending.pop(str(run_id), {})
        tool_name = info.get("tool", "unknown")
        if tool_name in SKIP_TRACE:
            return
        asyncio.create_task(_bg_call(self.http, self.session_id, "record_trace", {
            "user_id":         self.user_id,
            "session_id":      self.session_id,
            "origin_function": tool_name,
            "status":          "error",
            "memory_query":    info.get("input", ""),
            "error_message":   str(error)[:500],
        }))


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set in .env")
        return

    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2)

    # Discover all MCP tools from the wrapper
    mcp_client = MultiServerMCPClient({
        "cognee": {"url": MCP_URL, "transport": "streamable_http"},
    })
    # Orchestrator exposes exactly 4 tools — use all of them
    AGENT_TOOLS = {"save", "search", "improve", "delete"}
    all_tools = await mcp_client.get_tools()
    tools = [t for t in all_tools if t.name in AGENT_TOOLS]
    print(f"[orchestrator] {len(tools)} tools: {[t.name for t in tools]}")

    # Dedicated HTTP client for background trace + visualize calls
    base_url = MCP_URL.rsplit("/mcp", 1)[0]
    http     = httpx.AsyncClient(base_url=base_url, timeout=30)

    # Open a persistent background MCP session
    init_resp = await http.post(
        "/mcp",
        json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities":    {},
                "clientInfo":      {"name": "enterprise-brain-bg", "version": "1.0"},
            },
        },
        headers={"Accept": "application/json, text/event-stream"},
    )
    bg_session_id = init_resp.headers.get("mcp-session-id", "")
    print(f"[cognee] Background session: {bg_session_id[:8]}...")

    callback = CogneeObservabilityCallback(http, bg_session_id, USER_ID)
    agent    = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

    print("\n── Enterprise Brain Agent ──────────────────────────────")
    print(f"  Graph viewer : http://localhost:8000/graph")
    print(f"  Swagger UI   : http://localhost:8000/api/docs")
    print(f"  Type 'quit' to exit and run the improve loop\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input or user_input.lower() in {"quit", "exit", "q"}:
                break

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"callbacks": [callback]},
            )
            answer = result["messages"][-1].content
            print(f"\nAgent: {answer}\n")

            # Optional inline feedback
            fb = input("Rate this answer [1-5] or Enter to skip: ").strip()
            if fb in {"1", "2", "3", "4", "5"}:
                asyncio.create_task(_bg_call(http, bg_session_id, "submit_feedback", {
                    "user_id":        USER_ID,
                    "session_id":     bg_session_id,
                    "qa_id":          str(uuid.uuid4()),
                    "feedback_score": int(fb),
                }))
                print(f"  [feedback {fb}/5 recorded]\n")
            else:
                print()

    finally:
        print("[orchestrator] Running improve loop...")
        await _bg_call(http, bg_session_id, "improve", {})
        await asyncio.sleep(2)
        await http.aclose()
        print("[orchestrator] Session closed.")


if __name__ == "__main__":
    asyncio.run(main())
