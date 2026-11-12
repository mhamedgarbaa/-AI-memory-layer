"""
Interactive agent with two backends:

  --backend orchestrator  (default)  connects to :8003 — 4 intelligent tools
                                     save / search / improve / delete
                                     Auto-classifies documents, detects temporal,
                                     routes search to the best strategy.

  --backend wrapper                  connects to :8002 — original 18-tool gateway
                                     (backward-compatible mode)

Run:
    python agent_chat.py                           # orchestrator mode
    python agent_chat.py --backend wrapper         # direct wrapper mode
    GROQ_API_KEY=<key> python agent_chat.py
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any

import httpx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── env ───────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
AGENT_USER   = os.getenv("AGENT_USER", "chat_user")

MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR BACKEND  (:8003) — 4 intelligent tools
# ══════════════════════════════════════════════════════════════════════════════

ORCHESTRATOR_SYSTEM_PROMPT = """You are an intelligent memory assistant backed by a LangGraph Orchestrator Agent.
The orchestrator automatically classifies documents, detects temporal content, versions documents,
and routes queries to the best search strategy — you do not need to specify datasets or search types manually.

You have 4 tools:

WRITE:
- **save**: Store any document or fact. The orchestrator will:
    * classify it into the correct dataset (policy, regulation, employees, market, workflow, contracts, reports)
    * detect dates/events and flag it as temporal
    * check for existing versions and tag the new one
    * You can pass a hint (e.g. hint="market") to override classification.

READ:
- **search**: Query the knowledge graph. The orchestrator will:
    * analyse your query with an LLM
    * automatically pick the best search strategy (TEMPORAL, GRAPH_COMPLETION, GRAPH_SUMMARY, CYPHER, etc.)
    * You can pass hint="temporal", "factual", "summary", or "chunks" to guide routing.

MANAGE:
- **improve**: Run the quality improvement loop — applies feedback weights, persists sessions, syncs traces.
  Pass dataset_name="all" (default) or a specific dataset name.
- **delete**: Remove knowledge. Pass target="all" to wipe everything, a dataset name like "ds_policy",
  or a UUID to delete a single item. Optional mode="soft" (default) or "hard".

Guidelines:
- Use **save** when the user shares a fact, document, or information to remember.
- Use **search** before answering any question — let the orchestrator pick the right search type.
- Use **improve** when the user asks to refine, reinforce, or improve stored knowledge.
- Use **delete** only when explicitly asked to remove something.
- After each tool call, summarize the result in 1-2 sentences.
- If search returns nothing useful, say so honestly and answer from your own knowledge.
"""

ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save",
            "description": (
                "Save a document or fact into the knowledge graph. "
                "The orchestrator auto-classifies it into the right dataset, "
                "detects temporal content, and handles versioning automatically. "
                "Optionally pass hint to override classification (e.g. 'market', 'policy', 'contracts')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The full text content to save.",
                    },
                    "hint": {
                        "type": "string",
                        "description": "Optional dataset type hint: policy | regulation | employees | market | workflow | contracts | reports",
                    },
                },
                "required": ["data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the knowledge graph with a natural language query. "
                "The orchestrator analyses the query and automatically picks the best search strategy. "
                "Optionally pass hint='temporal' for time-based queries, 'summary' for overviews, "
                "'factual' for specific facts, or 'chunks' for raw text retrieval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or query.",
                    },
                    "hint": {
                        "type": "string",
                        "description": "Optional routing hint: temporal | factual | summary | chunks | cypher",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "improve",
            "description": (
                "Run the quality improvement loop: applies feedback weights, "
                "persists sessions into the graph, syncs traces, and rebuilds the session index."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Dataset to improve. Default 'all' runs on every dataset.",
                        "default": "all",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete",
            "description": (
                "Delete knowledge from the graph. "
                "target='all' wipes everything, a dataset name (e.g. 'ds_policy') deletes a full dataset, "
                "a UUID deletes a single item. Use mode='hard' for permanent deletion."
            ),
            "parameters": {
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
                    },
                },
                "required": ["target"],
            },
        },
    },
]


async def dispatch_orchestrator(mcp: "MCPClient", name: str, args: dict) -> str:
    if name == "save":
        return await mcp.call("save", {
            "data": args["data"],
            **( {"hint": args["hint"]} if args.get("hint") else {} ),
        }, timeout=180)

    if name == "search":
        return await mcp.call("search", {
            "query": args["query"],
            **( {"hint": args["hint"]} if args.get("hint") else {} ),
        }, timeout=120)

    if name == "improve":
        return await mcp.call("improve", {
            "dataset_name": args.get("dataset_name", "all"),
        }, timeout=180)

    if name == "delete":
        return await mcp.call("delete", {
            "target": args["target"],
            "mode":   args.get("mode", "soft"),
        }, timeout=60)

    return f"Unknown orchestrator tool: {name}"


# ══════════════════════════════════════════════════════════════════════════════
# WRAPPER BACKEND  (:8002) — original 18-tool gateway
# ══════════════════════════════════════════════════════════════════════════════

WRAPPER_SYSTEM_PROMPT = """You are a helpful memory assistant connected to a Cognee knowledge graph via the MCP Wrapper.

You have access to the following tools:

WRITE:
- **cognify**: Deep-process text into the graph (entity extraction, relationships). Use for facts, documents, rich content. Takes ~30s.
- **save_interaction**: Quickly save a short note without full graph processing. Takes ~2s.
- **memify**: Enrich stored chunks with temporal Event nodes (timestamps). Enables TEMPORAL search.
- **persist_sessions**: Store a conversation transcript as graph nodes for future recall.
- **improve_answer**: Re-query with chain-of-thought reasoning and store the corrected answer.

READ:
- **search**: Search the knowledge graph. Supports search_type: GRAPH_COMPLETION (default), TEMPORAL, CHUNKS, RAG_COMPLETION, CYPHER, and more.
- **list_data**: List all datasets currently stored in the graph.
- **cognify_status**: Check whether the cognify pipeline is idle or still running.
- **visualize_graph**: Render the full knowledge graph as an interactive HTML file.

ADMIN:
- **prune**: Wipe ALL stored knowledge. Irreversible.

Guidelines:
- When a user shares information to remember, use cognify (rich) or save_interaction (short).
- When a user asks a question, ALWAYS call search first before answering.
- For time-related queries, use search_type=TEMPORAL after running memify.
- Be concise. Summarize tool results in 1-2 sentences.
"""

WRAPPER_TOOLS = [
    {"type": "function", "function": {
        "name": "cognify",
        "description": "Store and deeply process knowledge into the Cognee graph. Set temporal=true when content contains dates/times.",
        "parameters": {"type": "object", "properties": {
            "data": {"type": "string"},
            "temporal": {"type": "boolean", "default": False},
        }, "required": ["data"]},
    }},
    {"type": "function", "function": {
        "name": "save_interaction",
        "description": "Quickly save a short note to memory (no deep graph processing).",
        "parameters": {"type": "object", "properties": {
            "data": {"type": "string"},
        }, "required": ["data"]},
    }},
    {"type": "function", "function": {
        "name": "search",
        "description": "Search the knowledge graph. search_type options: GRAPH_COMPLETION, TEMPORAL, RAG_COMPLETION, CHUNKS, SUMMARIES, CYPHER, GRAPH_COMPLETION_COT.",
        "parameters": {"type": "object", "properties": {
            "query":       {"type": "string"},
            "search_type": {"type": "string", "enum": [
                "GRAPH_COMPLETION", "GRAPH_COMPLETION_COT", "TEMPORAL",
                "RAG_COMPLETION", "CHUNKS", "SUMMARIES", "CYPHER",
            ], "default": "GRAPH_COMPLETION"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "list_data",
        "description": "List all datasets stored in the graph.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "prune",
        "description": "Wipe ALL stored knowledge. Irreversible.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "cognify_status",
        "description": "Check whether the cognify pipeline is idle or running.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "memify",
        "description": "Enrich graph with temporal Event nodes. Run after cognify.",
        "parameters": {"type": "object", "properties": {
            "dataset": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "visualize_graph",
        "description": "Render the Neo4j graph as interactive HTML.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "persist_sessions",
        "description": "Store a conversation transcript as knowledge graph nodes.",
        "parameters": {"type": "object", "properties": {
            "data": {"type": "string"},
        }, "required": ["data"]},
    }},
    {"type": "function", "function": {
        "name": "improve_answer",
        "description": "Re-query with chain-of-thought and store the corrected answer.",
        "parameters": {"type": "object", "properties": {
            "question":     {"type": "string"},
            "wrong_answer": {"type": "string"},
            "feedback":     {"type": "string"},
        }, "required": ["question", "wrong_answer", "feedback"]},
    }},
]


async def dispatch_wrapper(mcp: "MCPClient", name: str, args: dict) -> str:
    if name == "cognify":
        call_args: dict = {"data": args["data"]}
        if args.get("temporal") is True or str(args.get("temporal", "")).lower() == "true":
            call_args["temporal"] = True
        return await mcp.call("cognify", call_args, timeout=300)
    if name == "save_interaction":
        return await mcp.call("save_interaction", {"data": args["data"]}, timeout=120)
    if name == "search":
        return await mcp.call("search", {
            "search_query": args["query"],
            "search_type":  args.get("search_type", "GRAPH_COMPLETION"),
        }, timeout=180)
    if name == "list_data":
        return await mcp.call("list_data", {}, timeout=60)
    if name == "prune":
        return await mcp.call("prune", {}, timeout=60)
    if name == "cognify_status":
        return await mcp.call("cognify_status", {}, timeout=30)
    if name == "memify":
        return await mcp.call("memify", {"dataset": args.get("dataset", "main_dataset")}, timeout=300)
    if name == "visualize_graph":
        return await mcp.call("visualize_graph", {}, timeout=60)
    if name == "persist_sessions":
        return await mcp.call("persist_sessions", {"data": args["data"]}, timeout=120)
    if name == "improve_answer":
        return await mcp.call("improve_answer", {
            "question":     args["question"],
            "wrong_answer": args["wrong_answer"],
            "feedback":     args["feedback"],
        }, timeout=120)
    return f"Unknown wrapper tool: {name}"


# ══════════════════════════════════════════════════════════════════════════════
# MCP CLIENT (shared)
# ══════════════════════════════════════════════════════════════════════════════

class MCPClient:
    def __init__(self, client: httpx.AsyncClient):
        self._http = client
        self.session_id: str | None = None
        self._id = 0

    def _nid(self) -> int:
        self._id += 1
        return self._id

    @staticmethod
    def _sse(body: str) -> dict:
        for line in body.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise RuntimeError(f"No SSE data line: {body[:200]}")

    async def connect(self) -> None:
        resp = await self._http.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": self._nid(), "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "agent-chat", "version": "2.0"},
                },
            },
            headers=MCP_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        self.session_id = resp.headers.get("mcp-session-id")
        if not self.session_id:
            raise RuntimeError("No mcp-session-id from initialize")

    async def call(self, tool: str, args: dict[str, Any], timeout: float = 120) -> str:
        resp = await self._http.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": self._nid(), "method": "tools/call",
                "params": {"name": tool, "arguments": args},
            },
            headers={**MCP_HEADERS, "mcp-session-id": self.session_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = self._sse(resp.text)
        if "error" in data:
            err = data["error"]
            raise RuntimeError(f"MCP error [{err.get('code')}]: {err.get('message')}")
        content = data.get("result", {}).get("content", [])
        parts = [
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(parts) if parts else json.dumps(data.get("result", {}))


# ══════════════════════════════════════════════════════════════════════════════
# AGENTIC CHAT LOOP
# ══════════════════════════════════════════════════════════════════════════════

SLASH_HELP = """
Slash commands:
  /save <text>          Save text via the save tool
  /search <query>       Search the graph
  /search:temporal <q>  Search with temporal hint
  /search:summary <q>   Search with summary hint
  /delete <target>      Delete all | <dataset> | <uuid>
  /improve              Run improve loop
  /help                 Show this help
  /quit                 Exit
"""


async def chat_loop(mcp: MCPClient, backend: str) -> None:
    groq = Groq(api_key=GROQ_API_KEY)

    if backend == "orchestrator":
        tools         = ORCHESTRATOR_TOOLS
        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT
        dispatch_fn   = dispatch_orchestrator
        tool_names    = "save / search / improve / delete"
    else:
        tools         = WRAPPER_TOOLS
        system_prompt = WRAPPER_SYSTEM_PROMPT
        dispatch_fn   = dispatch_wrapper
        tool_names    = "cognify / search / save_interaction / list_data / prune + more"

    history: list[dict] = []

    print("\n" + "=" * 62)
    print(f"  Orchestrator Memory Agent  (backend: {backend})")
    print(f"  Endpoint : {mcp._http.base_url}")
    print(f"  Model    : {GROQ_MODEL}")
    print(f"  Tools    : {tool_names}")
    print(f"  Session  : {mcp.session_id[:24]}...")
    print("  Type /help for slash commands or just chat naturally.")
    print("=" * 62 + "\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue

        # ── Slash commands ─────────────────────────────────────────────────
        if raw.startswith("/"):
            cmd, _, rest = raw.partition(" ")
            cmd = cmd.lower()

            if cmd in ("/quit", "/exit", "/bye"):
                print("Bye.")
                break

            if cmd == "/help":
                print(SLASH_HELP)
                continue

            if cmd == "/save" and rest:
                print(f"\n  [save] saving...")
                try:
                    result = await mcp.call("save", {"data": rest}, timeout=180) \
                        if backend == "orchestrator" else \
                        await mcp.call("cognify", {"data": rest}, timeout=300)
                    print(f"  [save] {result[:200]}\n")
                except Exception as exc:
                    print(f"  [save] ERROR: {exc}\n")
                continue

            if cmd.startswith("/search"):
                hint = ""
                if ":" in cmd:
                    hint = cmd.split(":", 1)[1]
                query = rest or "summarize stored knowledge"
                print(f"\n  [search] querying (hint={hint or 'auto'})...")
                try:
                    if backend == "orchestrator":
                        call_args: dict = {"query": query}
                        if hint:
                            call_args["hint"] = hint
                        result = await mcp.call("search", call_args, timeout=120)
                    else:
                        stype_map = {"temporal": "TEMPORAL", "summary": "GRAPH_SUMMARY_COMPLETION", "chunks": "CHUNKS"}
                        result = await mcp.call("search", {
                            "search_query": query,
                            "search_type":  stype_map.get(hint, "GRAPH_COMPLETION"),
                        }, timeout=180)
                    print(f"  [search] {result[:400]}\n")
                except Exception as exc:
                    print(f"  [search] ERROR: {exc}\n")
                continue

            if cmd == "/improve":
                print(f"\n  [improve] running...")
                try:
                    result = await mcp.call("improve", {"dataset_name": rest or "all"}, timeout=180) \
                        if backend == "orchestrator" else \
                        await mcp.call("improve", {}, timeout=180)
                    print(f"  [improve] {result[:200]}\n")
                except Exception as exc:
                    print(f"  [improve] ERROR: {exc}\n")
                continue

            if cmd == "/delete" and rest:
                confirm = input(f"  Delete '{rest}'? (yes/no): ").strip().lower()
                if confirm == "yes":
                    try:
                        result = await mcp.call("delete", {"target": rest}, timeout=60) \
                            if backend == "orchestrator" else \
                            await mcp.call("prune", {}, timeout=60)
                        print(f"  [delete] {result[:200]}\n")
                    except Exception as exc:
                        print(f"  [delete] ERROR: {exc}\n")
                else:
                    print("  Cancelled.\n")
                continue

            print(f"  Unknown command '{cmd}'. Type /help.\n")
            continue

        # ── Natural language agentic loop ──────────────────────────────────
        history.append({"role": "user", "content": raw})

        while True:
            response = groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system_prompt}] + history,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2048,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                history.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    print(f"\n  [tool: {tc.function.name}] {json.dumps(args, ensure_ascii=False)[:100]}")
                    try:
                        result = await dispatch_fn(mcp, tc.function.name, args)
                        print(f"  [tool: {tc.function.name}] {result[:140].replace(chr(10), ' ')}")
                    except Exception as exc:
                        result = f"Tool error: {exc}"
                        print(f"  [tool: {tc.function.name}] ERROR: {exc}")

                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            answer = msg.content or "(no response)"
            history.append({"role": "assistant", "content": answer})
            print(f"\nAgent: {answer}\n")
            break


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    parser = argparse.ArgumentParser(description="Cognee memory agent chat")
    parser.add_argument(
        "--backend",
        choices=["orchestrator", "wrapper"],
        default="orchestrator",
        help="orchestrator=:8003 (4-tool, default)  wrapper=:8002 (18-tool)",
    )
    parser.add_argument("--url", default=None, help="Override MCP URL")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("ERROR: set GROQ_API_KEY in .env or environment.")
        sys.exit(1)

    default_ports = {"orchestrator": "http://localhost:8003", "wrapper": "http://localhost:8002"}
    mcp_url = args.url or os.getenv("MCP_URL", default_ports[args.backend])

    async with httpx.AsyncClient(base_url=mcp_url, timeout=30) as http:
        mcp = MCPClient(http)
        print(f"Connecting to {args.backend} at {mcp_url} ...")
        await mcp.connect()
        await chat_loop(mcp, args.backend)


if __name__ == "__main__":
    asyncio.run(main())
