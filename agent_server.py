"""
Cognee Memory Agent — web UI server
Runs on http://localhost:8080

Layout:
  Left  — chat with the Llama-4 agent (Groq), backed by Cognee MCP tools
  Right — upload files (PDF, CSV, Excel, images, text) or paste text to cognify
          via the Orchestrator Agent (:8003) which auto-classifies into the right dataset

Run:
    python agent_server.py
"""

import asyncio
import csv
import io
import json
import os
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from groq import AsyncGroq
from pydantic import BaseModel

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MCP_URL            = os.getenv("MCP_URL", "http://localhost:8002")
MCP_HOST_HEADER    = os.getenv("MCP_HOST_HEADER", "localhost:8002")
ORCHESTRATOR_URL   = os.getenv("ORCHESTRATOR_URL", "http://localhost:8003")
WEB_API_URL        = os.getenv("WEB_API_URL", "http://localhost:8000")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
AGENT_USER         = os.getenv("AGENT_USER", "chat_user")

MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Host": MCP_HOST_HEADER,
}

SYSTEM_PROMPT = """You are a helpful memory assistant connected to a Cognee knowledge graph.

## Storage — automatic classification
When you store content with `cognify` or `save_interaction`, the Orchestrator automatically
classifies it into the right typed dataset (policy, regulation, market, contracts, etc.).
You do NOT need to choose a dataset. Optionally pass a `hint` if you are certain of the type:
  policy | regulation | employees | market | workflow | contracts | reports

## Datasets available
ds_policy · ds_regulation · ds_employees · ds_market · ds_workflow · ds_contracts · ds_reports

## Tools — READ (return data):
- search: Query the knowledge graph. ALWAYS call this before answering factual questions.
- list_data: Show what datasets are stored.
- cognify_status: Check if the pipeline is still running.

## Tools — WRITE (actions, return status only):
- cognify: Store rich content — the orchestrator classifies and routes it to the right dataset.
  Optional: pass `hint` (e.g. "market") to guide classification.
- save_interaction: Save a short note quickly. Same auto-classification applies.
- memify: Extract temporal Event nodes after cognify. Takes 1-3 min.
- visualize_graph: Render the graph as HTML.
- prune: Wipe ALL memory (only when explicitly asked).

## Rules:
- ALWAYS call search to answer factual questions — never answer from memory alone.
- memify and visualize_graph are background operations; do NOT read data from their result.
- Be concise. Summarize search results; don't dump raw output.
- If search returns nothing relevant, say so clearly.
"""


# ── MCP client ────────────────────────────────────────────────────────────────

class MCPClient:
    def __init__(self, base_url: str, host_header: str | None = None):
        self.base_url = base_url
        self._host_header = host_header
        self._http: httpx.AsyncClient | None = None
        self.session_id: str | None = None
        self._id = 0
        self._lock = asyncio.Lock()

    @property
    def _headers(self) -> dict:
        h = {"Accept": "application/json, text/event-stream"}
        if self._host_header:
            h["Host"] = self._host_header
        return h

    async def connect(self):
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=30)
        self._id = 0
        resp = await self._http.post(
            "/mcp",
            json={
                "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "agent-ui", "version": "1.0"},
                },
            },
            headers=self._headers,
        )
        resp.raise_for_status()
        self.session_id = resp.headers.get("mcp-session-id")

    async def close(self):
        if self._http:
            await self._http.aclose()

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    @staticmethod
    def _sse(body: str) -> dict:
        for line in body.splitlines():
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise RuntimeError(f"No SSE data: {body[:200]}")

    async def call(self, tool: str, args: dict, timeout: float = 120) -> str:
        async with self._lock:
            resp = await self._http.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0", "id": self._next_id(), "method": "tools/call",
                    "params": {"name": tool, "arguments": args},
                },
                headers={**self._headers, "mcp-session-id": self.session_id},
                timeout=timeout,
            )
        resp.raise_for_status()
        data = self._sse(resp.text)
        content = data.get("result", {}).get("content", [])
        parts = [
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(parts) or json.dumps(data.get("result", {}))


# ── app state ─────────────────────────────────────────────────────────────────

mcp            = MCPClient(MCP_URL, MCP_HOST_HEADER)   # :8002 — chat tools
orchestrator   = MCPClient(ORCHESTRATOR_URL)            # :8003 — save/search/improve/delete
groq_client: AsyncGroq | None = None
sessions: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client
    groq_client = AsyncGroq(api_key=GROQ_API_KEY)
    await mcp.connect()
    print(f"MCP wrapper connected   session={mcp.session_id}")
    try:
        await orchestrator.connect()
        print(f"Orchestrator connected  session={orchestrator.session_id}")
    except Exception as exc:
        print(f"Orchestrator not reachable ({exc}) — uploads will fallback to mcp-wrapper")
    yield
    await mcp.close()
    await orchestrator.close()


app = FastAPI(lifespan=lifespan)


# ── text extraction helpers ───────────────────────────────────────────────────

def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())
    except ImportError:
        return "[PDF extraction requires pypdf — run: pip install pypdf]"


def _extract_csv(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return text
    header = rows[0]
    lines = [", ".join(header)]
    for row in rows[1:500]:   # cap at 500 rows to avoid huge context
        lines.append(", ".join(row))
    if len(rows) > 501:
        lines.append(f"... ({len(rows) - 501} more rows truncated)")
    return "\n".join(lines)


def _extract_excel(data: bytes) -> str:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        parts = []
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i >= 500:
                    rows.append("... (truncated)")
                    break
                rows.append(", ".join("" if v is None else str(v) for v in row))
            parts.append(f"[Sheet: {sheet}]\n" + "\n".join(rows))
        return "\n\n".join(parts)
    except ImportError:
        return "[Excel extraction requires openpyxl — run: pip install openpyxl]"


def _extract_image_note(filename: str) -> str:
    return (
        f"[Image file: {filename}]\n"
        "Note: image content was uploaded. "
        "For visual analysis, describe this image and its key contents."
    )


def extract_text(filename: str, content_type: str, data: bytes) -> str:
    name_lower = filename.lower()

    if name_lower.endswith(".pdf") or "pdf" in content_type:
        return _extract_pdf(data)

    if name_lower.endswith(".csv") or "csv" in content_type:
        return _extract_csv(data)

    if name_lower.endswith((".xlsx", ".xls")) or "excel" in content_type or "spreadsheet" in content_type:
        return _extract_excel(data)

    if name_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")) or content_type.startswith("image/"):
        return _extract_image_note(filename)

    # JSON, markdown, txt, plain text
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return f"[Could not decode file: {filename}]"


# ── tool definitions (chat tools — mcp-wrapper) ───────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cognify",
            "description": (
                "Store and deeply process knowledge into the graph. "
                "The orchestrator automatically classifies the content into the right typed dataset. "
                "Optionally pass a hint if you are certain of the type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string"},
                    "hint": {
                        "type": "string",
                        "description": "Optional classification hint: policy | regulation | employees | market | workflow | contracts | reports",
                    },
                },
                "required": ["data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_interaction",
            "description": (
                "Quickly save a short note. "
                "The orchestrator classifies it automatically. Optionally pass a hint."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string"},
                    "hint": {
                        "type": "string",
                        "description": "Optional classification hint: policy | regulation | employees | market | workflow | contracts | reports",
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
                "Search the knowledge graph. Choose search_type based on the question:\n"
                "- GRAPH_COMPLETION: default — entity/relationship graph traversal + LLM synthesis\n"
                "- GRAPH_COMPLETION_COT: chain-of-thought multi-round reasoning (complex questions)\n"
                "- GRAPH_COMPLETION_CONTEXT_EXTENSION: expands context with follow-up queries\n"
                "- GRAPH_SUMMARY_COMPLETION: includes text summaries in context\n"
                "- TEMPORAL: time-aware — filters Event nodes by timestamp (use for 'when' questions)\n"
                "- CYPHER: LLM generates a Cypher query (structured graph questions)\n"
                "- NATURAL_LANGUAGE: NL to graph query auto-translation\n"
                "- RAG_COMPLETION: classic vector RAG\n"
                "- SUMMARIES: retrieve document summaries only\n"
                "- CHUNKS: retrieve raw text chunks\n"
                "- FEELING_LUCKY: auto-selects best search type"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "search_type": {
                        "type": "string",
                        "enum": [
                            "GRAPH_COMPLETION", "GRAPH_COMPLETION_COT",
                            "GRAPH_COMPLETION_CONTEXT_EXTENSION", "GRAPH_SUMMARY_COMPLETION",
                            "TEMPORAL", "CYPHER", "NATURAL_LANGUAGE",
                            "RAG_COMPLETION", "SUMMARIES", "CHUNKS", "FEELING_LUCKY",
                        ],
                        "default": "GRAPH_COMPLETION",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_data",
            "description": "List all datasets in the knowledge graph.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cognify_status",
            "description": "Check whether the cognify background pipeline is idle or still processing.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prune",
            "description": "Wipe ALL stored knowledge. Only call when the user explicitly asks.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visualize_graph",
            "description": "Render the full Neo4j knowledge graph as an interactive HTML visualization.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memify",
            "description": (
                "Temporal graph enrichment: reads DocumentChunk nodes, extracts Event nodes with "
                "timestamps via LLM, writes them to Neo4j. Call after cognify to enable TEMPORAL "
                "search. Takes 1-3 minutes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "description": "Dataset name (default: main_dataset)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "persist_sessions",
            "description": (
                "Self-improvement: converts recent conversation sessions into new graph nodes. "
                "Call periodically to keep the graph up to date with recent interactions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset": {"type": "string", "description": "Dataset name (default: main_dataset)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "improve_answer",
            "description": (
                "Self-improvement: when a previous answer was wrong or incomplete, provide the "
                "original question, the wrong answer, and user feedback. The system re-queries "
                "the graph with chain-of-thought reasoning to generate a corrected answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question":     {"type": "string"},
                    "wrong_answer": {"type": "string"},
                    "feedback":     {"type": "string"},
                },
                "required": ["question", "wrong_answer", "feedback"],
            },
        },
    },
]


# ── tool dispatch (chat) ───────────────────────────────────────────────────────

async def _orchestrator_save(data: str, hint: str = "") -> str:
    """Route a write through the orchestrator for auto-classification.
    Falls back to direct mcp-wrapper cognify if orchestrator is unavailable."""
    if orchestrator.session_id:
        return await orchestrator.call(
            "save",
            {"data": data, "hint": hint},
            timeout=300,
        )
    # fallback: no classification, goes to main_dataset
    return await mcp.call("cognify", {"data": data, "user": AGENT_USER}, timeout=300)


async def dispatch(tool: str, args: dict) -> str:
    if tool == "cognify":
        return await _orchestrator_save(args.get("data", ""), args.get("hint", ""))
    if tool == "save_interaction":
        return await _orchestrator_save(args.get("data", ""), args.get("hint", ""))
    if tool == "search":
        return await mcp.call(
            "search",
            {
                "search_query": args.get("query", ""),
                "search_type":  args.get("search_type", "GRAPH_COMPLETION"),
                "user": AGENT_USER,
            },
            timeout=180,
        )
    if tool == "list_data":
        return await mcp.call("list_data", {}, timeout=60)
    if tool == "cognify_status":
        return await mcp.call("cognify_status", {}, timeout=30)
    if tool == "prune":
        return await mcp.call("prune", {}, timeout=60)
    if tool == "visualize_graph":
        return await run_visualization()
    if tool == "memify":
        return await run_memify(args.get("dataset", "main_dataset"))
    if tool == "persist_sessions":
        return await run_persist_sessions(args.get("dataset", "main_dataset"))
    if tool == "improve_answer":
        return await run_improve_answer(
            args.get("question", ""),
            args.get("wrong_answer", ""),
            args.get("feedback", ""),
        )
    return f"Unknown tool: {tool}"


async def run_memify(dataset: str = "main_dataset") -> str:
    loop = asyncio.get_event_loop()

    def _exec():
        result = subprocess.run(
            ["docker", "exec", "cognee_mcp_server", "bash", "-c",
             f"python3 /tmp/run_memify.py {dataset}"],
            capture_output=True, text=True, timeout=300,
        )
        return result.stdout + result.stderr

    output = await loop.run_in_executor(None, _exec)
    return "\n".join(output.strip().splitlines()[-6:]) or "Memify complete."


async def run_persist_sessions(dataset: str = "main_dataset") -> str:
    if not sessions:
        return "No sessions to persist yet."
    all_text = []
    for history in sessions.values():
        turns = []
        for msg in history:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and content:
                turns.append(f"User: {content}")
            elif role == "assistant" and content:
                turns.append(f"Assistant: {content}")
        if turns:
            all_text.append("\n".join(turns))
    if not all_text:
        return "Sessions are empty — nothing to persist."
    combined = "\n\n---\n\n".join(all_text)
    result = await mcp.call("cognify", {"data": combined, "user": AGENT_USER}, timeout=300)
    return f"Persisted {len(all_text)} session(s) to graph. {result[:200]}"


async def run_improve_answer(question: str, wrong_answer: str, feedback: str) -> str:
    if not question:
        return "Error: question is required."
    improved = await mcp.call(
        "search",
        {"search_query": question, "search_type": "GRAPH_COMPLETION_COT", "user": AGENT_USER},
        timeout=180,
    )
    correction = (
        f"CORRECTION — Question: {question}\n"
        f"Wrong answer: {wrong_answer}\n"
        f"Feedback: {feedback}\n"
        f"Improved answer: {improved}"
    )
    await mcp.call("save_interaction", {"data": correction, "user": AGENT_USER}, timeout=120)
    return f"Improved answer (stored in graph):\n{improved}"


COGNEE_CONTAINER = os.getenv("COGNEE_CONTAINER_NAME", "cognee_mcp_server")
GRAPH_HTML_PATH  = os.getenv("GRAPH_OUTPUT_PATH", "c:/Users/mhame/Cognee-layer/cognee_graph_neo4j.html")


async def run_visualization() -> str:
    loop = asyncio.get_event_loop()

    def _exec():
        subprocess.run(
            ["docker", "exec", COGNEE_CONTAINER, "bash", "-c", "python3 /tmp/visualize.py"],
            capture_output=True, timeout=60,
        )
        result = subprocess.run(
            ["docker", "cp", f"{COGNEE_CONTAINER}:/tmp/cognee_graph.html", GRAPH_HTML_PATH],
            capture_output=True, timeout=10,
        )
        return result.returncode

    rc = await loop.run_in_executor(None, _exec)
    if rc == 0:
        return "Graph rendered. Open it at: http://localhost:8080/api/graph"
    return f"Visualization failed — check docker logs {COGNEE_CONTAINER}"


# ── API models ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str
    session_id: str = ""

class CognifyRequest(BaseModel):
    content: str


# ── streaming chat endpoint ───────────────────────────────────────────────────

async def agent_stream(message: str, session_id: str) -> AsyncGenerator[str, None]:
    history = sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": message})

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    MAX_ROUNDS = 8
    for _round in range(MAX_ROUNDS):
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=2048,
            timeout=60,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            history.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                yield sse("tool_start", {"tool": tool_name, "args": args})
                result = await dispatch(tool_name, args)
                yield sse("tool_done", {"tool": tool_name, "result": result[:400]})

                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        answer = msg.content or ""
        history.append({"role": "assistant", "content": answer})
        yield sse("answer", {"text": answer})
        return

    yield sse("answer", {"text": "I ran into a loop and could not produce a final answer. Please try rephrasing your question."})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    sid = req.session_id or str(uuid.uuid4())
    return StreamingResponse(
        agent_stream(req.message, sid),
        media_type="text/event-stream",
        headers={"X-Session-Id": sid},
    )


# ── upload endpoint — delegates to web container ingest API ──────────────────

@app.post("/api/upload")
async def upload_file(
    file:         UploadFile = File(...),
    hint:         str        = Form(""),
    dataset_name: str        = Form(""),
):
    """
    Forward file to web container POST /api/v1/ingest/upload.
    With dataset_name → mcp-wrapper direct (add → cognify on named dataset).
    Without dataset_name → orchestrator auto-classify.
    """
    filename = file.filename or "upload"
    data     = await file.read()

    if not data:
        return {"status": "error", "message": "Empty file."}

    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            form: dict = {}
            if hint.strip():
                form["hint"] = hint.strip().lower()
            if dataset_name.strip():
                form["dataset_name"] = dataset_name.strip()

            resp = await client.post(
                f"{WEB_API_URL}/api/v1/ingest/upload",
                files={"file": (filename, data, file.content_type or "application/octet-stream")},
                data=form,
            )
            resp.raise_for_status()
            return resp.json()

    except httpx.HTTPStatusError as exc:
        return {"status": "error", "filename": filename, "message": exc.response.text}
    except Exception as exc:
        return {"status": "error", "filename": filename, "message": str(exc)}


# ── legacy text cognify endpoint ─────────────────────────────────────────────

@app.post("/api/cognify")
async def cognify_doc(req: CognifyRequest):
    """Forward raw text to web container POST /api/v1/ingest/cognify."""
    if not req.content.strip():
        return {"status": "error", "message": "Content is empty."}
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            form: dict = {"data": req.content}
            if getattr(req, "hint", ""):
                form["hint"] = req.hint.strip().lower()
            if getattr(req, "dataset_name", ""):
                form["dataset_name"] = req.dataset_name.strip()

            resp = await client.post(
                f"{WEB_API_URL}/api/v1/ingest/cognify",
                data=form,
            )
            resp.raise_for_status()
            return resp.json()

    except httpx.HTTPStatusError as exc:
        return {"status": "error", "message": exc.response.text}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.get("/api/graph", response_class=HTMLResponse)
async def serve_graph():
    await run_visualization()
    try:
        with open(GRAPH_HTML_PATH, encoding="utf-8") as f:
            content = f.read()
        from fastapi.responses import HTMLResponse as HR
        return HR(content=content, headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"})
    except FileNotFoundError:
        return "<h2>No graph yet — ask the agent to visualize_graph first.</h2>"


@app.get("/api/status")
async def cognify_status():
    try:
        result = await mcp.call("cognify_status", {}, timeout=30)
        idle = result.strip() in ("{}", "")
        return {"idle": idle, "raw": result[:200]}
    except Exception as exc:
        return {"idle": False, "raw": str(exc)}


@app.get("/api/datasets")
async def datasets():
    try:
        result = await mcp.call("list_data", {}, timeout=30)
        return {"result": result[:1000]}
    except Exception as exc:
        return {"error": str(exc)}


# ── frontend ──────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Cognee Agent</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{
    --bg:#0f1117;--surface:#1a1d27;--surface2:#242736;
    --border:#2e3147;--accent:#6c63ff;--accent2:#4ecca3;
    --text:#e8eaf6;--muted:#6b7280;--danger:#ef4444;--warn:#f59e0b;
  }
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
       height:100vh;display:flex;flex-direction:column;overflow:hidden}

  header{background:var(--surface);border-bottom:1px solid var(--border);
         padding:12px 24px;display:flex;align-items:center;gap:12px;flex-shrink:0}
  .logo{width:32px;height:32px;background:linear-gradient(135deg,var(--accent),var(--accent2));
        border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px}
  header h1{font-size:16px;font-weight:600}
  .badge{background:var(--surface2);border:1px solid var(--border);border-radius:20px;
         padding:2px 10px;font-size:11px;color:var(--muted);margin-left:auto}
  .dot{width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block;margin-right:6px}

  main{flex:1;display:flex;gap:0;overflow:hidden}

  /* ── chat panel ── */
  .chat-panel{flex:1;display:flex;flex-direction:column;border-right:1px solid var(--border)}
  .panel-header{padding:14px 20px;border-bottom:1px solid var(--border);
                font-size:12px;font-weight:600;text-transform:uppercase;
                letter-spacing:.08em;color:var(--muted);background:var(--surface)}
  .messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
  .messages::-webkit-scrollbar{width:4px}
  .messages::-webkit-scrollbar-track{background:transparent}
  .messages::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}

  .msg{display:flex;gap:10px;max-width:88%}
  .msg.user{align-self:flex-end;flex-direction:row-reverse}
  .avatar{width:30px;height:30px;border-radius:50%;flex-shrink:0;display:flex;
          align-items:center;justify-content:center;font-size:13px;font-weight:700}
  .msg.user .avatar{background:var(--accent)}
  .msg.agent .avatar{background:var(--surface2);border:1px solid var(--border)}
  .bubble{padding:10px 14px;border-radius:14px;font-size:14px;line-height:1.55;white-space:pre-wrap}
  .msg.user .bubble{background:var(--accent);border-radius:14px 4px 14px 14px}
  .msg.agent .bubble{background:var(--surface2);border:1px solid var(--border);
                     border-radius:4px 14px 14px 14px}

  .tool-pill{display:inline-flex;align-items:center;gap:6px;background:var(--surface);
             border:1px solid var(--border);border-radius:20px;padding:4px 12px;
             font-size:11px;color:var(--muted);margin:4px 0;align-self:flex-start}
  .tool-pill .icon{font-size:12px}
  .tool-pill.done{border-color:var(--accent2);color:var(--accent2)}
  .tool-pill.running{border-color:var(--warn);color:var(--warn)}

  .chat-input{padding:16px 20px;border-top:1px solid var(--border);background:var(--surface);
              display:flex;gap:10px;align-items:flex-end}
  textarea.chat-box{flex:1;background:var(--surface2);border:1px solid var(--border);
                    border-radius:10px;color:var(--text);font-size:14px;resize:none;
                    padding:10px 14px;outline:none;font-family:inherit;min-height:44px;max-height:120px;
                    line-height:1.5}
  textarea.chat-box:focus{border-color:var(--accent)}
  textarea.chat-box::placeholder{color:var(--muted)}
  .send-btn{background:var(--accent);border:none;border-radius:10px;
            width:44px;height:44px;cursor:pointer;display:flex;align-items:center;
            justify-content:center;font-size:18px;flex-shrink:0;transition:opacity .2s}
  .send-btn:hover{opacity:.85}
  .send-btn:disabled{opacity:.4;cursor:not-allowed}

  /* ── cognify / upload panel ── */
  .cognify-panel{width:390px;flex-shrink:0;display:flex;flex-direction:column;background:var(--surface)}
  .cognify-body{flex:1;padding:18px;display:flex;flex-direction:column;gap:12px;overflow-y:auto}
  .field-label{font-size:11px;font-weight:600;text-transform:uppercase;
               letter-spacing:.08em;color:var(--muted);margin-bottom:6px}

  /* tabs */
  .tab-bar{display:flex;gap:0;border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:2px}
  .tab{flex:1;padding:7px;background:none;border:none;color:var(--muted);font-size:12px;
       font-weight:600;cursor:pointer;transition:background .15s,color .15s;text-align:center}
  .tab.active{background:var(--surface2);color:var(--text)}
  .tab-content{display:none;flex-direction:column;gap:10px}
  .tab-content.active{display:flex}

  /* drop zone */
  .drop-zone{border:2px dashed var(--border);border-radius:12px;padding:28px 16px;
             text-align:center;cursor:pointer;transition:border-color .2s,background .2s}
  .drop-zone:hover,.drop-zone.over{border-color:var(--accent2);background:rgba(78,204,163,.06)}
  .drop-zone .dz-icon{font-size:32px;margin-bottom:8px}
  .drop-zone .dz-title{font-size:13px;font-weight:600;color:var(--text);margin-bottom:4px}
  .drop-zone .dz-sub{font-size:11px;color:var(--muted)}
  .file-chip{display:flex;align-items:center;gap:8px;background:var(--surface2);
             border:1px solid var(--border);border-radius:8px;padding:8px 12px;
             font-size:12px;display:none}
  .file-chip .file-icon{font-size:18px}
  .file-chip .file-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .file-chip .file-remove{cursor:pointer;color:var(--muted);font-size:14px;flex-shrink:0}
  .file-chip .file-remove:hover{color:var(--danger)}

  /* hint selector */
  select.hint-select{width:100%;background:var(--surface2);border:1px solid var(--border);
                     border-radius:8px;color:var(--text);font-size:13px;padding:8px 10px;
                     outline:none;cursor:pointer}
  select.hint-select:focus{border-color:var(--accent2)}

  textarea.doc-box{background:var(--surface2);border:1px solid var(--border);
                   border-radius:10px;color:var(--text);font-size:13px;resize:none;
                   padding:12px 14px;outline:none;font-family:inherit;min-height:180px;
                   line-height:1.5;width:100%}
  textarea.doc-box:focus{border-color:var(--accent2)}
  textarea.doc-box::placeholder{color:var(--muted)}

  .cognify-btn{background:linear-gradient(135deg,var(--accent2),#36b37e);border:none;
               border-radius:10px;color:#0a2018;font-weight:700;font-size:14px;
               padding:12px;cursor:pointer;transition:opacity .2s;display:flex;
               align-items:center;justify-content:center;gap:8px;width:100%}
  .cognify-btn:hover{opacity:.9}
  .cognify-btn:disabled{opacity:.4;cursor:not-allowed}

  /* result card */
  .result-card{background:var(--surface2);border:1px solid var(--border);border-radius:10px;
               padding:12px 14px;font-size:12px;line-height:1.7;display:none}
  .result-card.ok{border-color:var(--accent2)}
  .result-card.err{border-color:var(--danger);color:var(--danger)}
  .result-card .rc-row{display:flex;justify-content:space-between;align-items:center}
  .result-card .rc-label{color:var(--muted)}
  .result-card .rc-value{font-weight:600;color:var(--text)}
  .result-card .rc-value.highlight{color:var(--accent2)}
  .result-card .rc-err{color:var(--danger)}

  .status-box{background:var(--surface2);border:1px solid var(--border);border-radius:10px;
              padding:12px 14px;font-size:12px;color:var(--muted);line-height:1.6;min-height:52px}
  .status-box.ok{border-color:var(--accent2);color:var(--accent2)}
  .status-box.err{border-color:var(--danger);color:var(--danger)}
  .status-box.running{border-color:var(--warn);color:var(--warn)}
  .spinner{display:inline-block;width:10px;height:10px;border:2px solid currentColor;
           border-top-color:transparent;border-radius:50%;animation:spin .6s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  .action-row{display:flex;gap:8px}
  .outline-btn{flex:1;background:none;border:1px solid var(--border);border-radius:8px;
               color:var(--muted);font-size:12px;padding:7px 12px;cursor:pointer;
               display:flex;align-items:center;justify-content:center;gap:6px;
               transition:border-color .2s;white-space:nowrap}
  .outline-btn:hover{border-color:var(--accent);color:var(--text)}
  .outline-btn.green:hover{border-color:var(--accent2);color:var(--accent2)}
  .datasets-result{background:var(--surface2);border:1px solid var(--border);border-radius:10px;
                   padding:12px;font-size:11px;color:var(--muted);white-space:pre-wrap;
                   max-height:160px;overflow-y:auto;display:none}
  .datasets-result.visible{display:block}
</style>
</head>
<body>

<header>
  <div class="logo">&#x1F9E0;</div>
  <h1>Cognee Memory Agent</h1>
  <span class="badge"><span class="dot"></span>llama-4-scout &middot; Neo4j &middot; Orchestrator</span>
</header>

<main>
  <!-- ── chat panel ── -->
  <div class="chat-panel">
    <div class="panel-header">&#x1F4AC; Chat</div>
    <div class="messages" id="messages">
      <div class="msg agent">
        <div class="avatar">&#x1F916;</div>
        <div class="bubble">Hi! I'm your Cognee memory assistant. Ask me anything, or tell me something to remember. I can search the knowledge graph, store new facts, and more.

Upload documents (PDF, CSV, Excel, images) in the panel on the right — they get auto-classified into the right dataset.</div>
      </div>
    </div>
    <div class="chat-input">
      <textarea class="chat-box" id="chatInput" placeholder="Ask something or tell me a fact to remember&#x2026;" rows="1"></textarea>
      <button class="send-btn" id="sendBtn" title="Send">&#x27A4;</button>
    </div>
  </div>

  <!-- ── upload / cognify panel ── -->
  <div class="cognify-panel">
    <div class="panel-header">&#x1F4C2; Add to Knowledge Graph</div>
    <div class="cognify-body">

      <!-- tabs -->
      <div class="tab-bar">
        <button class="tab active" id="tabUpload" onclick="switchTab('upload')">&#x2B06; Upload File</button>
        <button class="tab" id="tabPaste" onclick="switchTab('paste')">&#x1F4DD; Paste Text</button>
      </div>

      <!-- ── FILE UPLOAD TAB ── -->
      <div class="tab-content active" id="tabContentUpload">
        <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
          <input type="file" id="fileInput"
            accept=".pdf,.csv,.xlsx,.xls,.txt,.md,.json,.png,.jpg,.jpeg,.gif,.webp"
            style="display:none">
          <div class="dz-icon">&#x1F4C4;</div>
          <div class="dz-title">Drop file or click to browse</div>
          <div class="dz-sub">.pdf &middot; .csv &middot; .xlsx &middot; .txt &middot; .json &middot; .png &middot; .jpg</div>
        </div>
        <div class="file-chip" id="fileChip">
          <span class="file-icon" id="fileChipIcon">&#x1F4C4;</span>
          <span class="file-name" id="fileChipName"></span>
          <span class="file-remove" onclick="clearFile()" title="Remove">&#x2715;</span>
        </div>
        <div>
          <div class="field-label">Dataset hint (optional)</div>
          <select class="hint-select" id="uploadHint">
            <option value="">Auto-detect (recommended)</option>
            <option value="policy">Policy</option>
            <option value="regulation">Regulation</option>
            <option value="employees">Employees / HR</option>
            <option value="market">Market / Sales</option>
            <option value="workflow">Workflow / Process</option>
            <option value="contracts">Contracts / Suppliers</option>
            <option value="reports">Reports / Analysis</option>
          </select>
        </div>
        <button class="cognify-btn" id="uploadBtn" onclick="doUpload()">
          <span>&#x26A1;</span> Cognify File
        </button>
        <div class="result-card" id="uploadResult"></div>
      </div>

      <!-- ── PASTE TEXT TAB ── -->
      <div class="tab-content" id="tabContentPaste">
        <div>
          <div class="field-label">Paste document or text</div>
          <textarea class="doc-box" id="docInput"
            placeholder="Paste any document, article, notes, or data here.

The orchestrator will auto-classify it into the right dataset (policy, market, contracts&#x2026;) and extract entities and relationships into Neo4j."></textarea>
        </div>
        <div>
          <div class="field-label">Dataset hint (optional)</div>
          <select class="hint-select" id="pasteHint">
            <option value="">Auto-detect (recommended)</option>
            <option value="policy">Policy</option>
            <option value="regulation">Regulation</option>
            <option value="employees">Employees / HR</option>
            <option value="market">Market / Sales</option>
            <option value="workflow">Workflow / Process</option>
            <option value="contracts">Contracts / Suppliers</option>
            <option value="reports">Reports / Analysis</option>
          </select>
        </div>
        <button class="cognify-btn" id="cognifyBtn">
          <span>&#x26A1;</span> Cognify into Graph
        </button>
        <div class="status-box" id="statusBox">Ready</div>
      </div>

      <!-- shared actions -->
      <div class="action-row">
        <button class="outline-btn" id="datasetsBtn">&#x1F4C2; Datasets</button>
        <button class="outline-btn green" id="graphBtn">&#x1F578; Graph</button>
      </div>
      <div class="datasets-result" id="datasetsResult"></div>

    </div>
  </div>
</main>

<script>
const messagesEl  = document.getElementById('messages');
const chatInput   = document.getElementById('chatInput');
const sendBtn     = document.getElementById('sendBtn');
const docInput    = document.getElementById('docInput');
const cognifyBtn  = document.getElementById('cognifyBtn');
const statusBox   = document.getElementById('statusBox');
const datasetsBtn = document.getElementById('datasetsBtn');
const datasetsResult = document.getElementById('datasetsResult');
const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const fileChip    = document.getElementById('fileChip');
const uploadBtn   = document.getElementById('uploadBtn');
const uploadResult = document.getElementById('uploadResult');

let sessionId = '';
let busy      = false;
let selectedFile = null;

// ── tab switching ──

function switchTab(tab) {
  document.getElementById('tabUpload').classList.toggle('active', tab === 'upload');
  document.getElementById('tabPaste').classList.toggle('active', tab === 'paste');
  document.getElementById('tabContentUpload').classList.toggle('active', tab === 'upload');
  document.getElementById('tabContentPaste').classList.toggle('active', tab === 'paste');
}

// ── file helpers ──

const ICONS = {
  pdf: '&#x1F4D5;', csv: '&#x1F4CA;', xlsx: '&#x1F4CA;', xls: '&#x1F4CA;',
  png: '&#x1F5BC;', jpg: '&#x1F5BC;', jpeg: '&#x1F5BC;', gif: '&#x1F5BC;',
  json: '&#x1F9E9;', md: '&#x1F4DD;', txt: '&#x1F4CB;',
};

function fileIcon(name) {
  const ext = (name.split('.').pop() || '').toLowerCase();
  return ICONS[ext] || '&#x1F4C4;';
}

function setFile(f) {
  selectedFile = f;
  document.getElementById('fileChipIcon').innerHTML = fileIcon(f.name);
  document.getElementById('fileChipName').textContent = f.name + '  (' + (f.size > 1048576
    ? (f.size / 1048576).toFixed(1) + ' MB' : Math.round(f.size / 1024) + ' KB') + ')';
  fileChip.style.display = 'flex';
  dropZone.style.display = 'none';
  uploadResult.style.display = 'none';
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  fileChip.style.display = 'none';
  dropZone.style.display = '';
  uploadResult.style.display = 'none';
}

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('over');
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

// ── upload ──

async function doUpload() {
  if (!selectedFile) { showUploadResult('err', 'No file selected.'); return; }
  uploadBtn.disabled = true;
  showUploadResult('running', '<span class="spinner"></span> Extracting and cognifying&hellip;');

  const fd = new FormData();
  fd.append('file', selectedFile);
  fd.append('hint', document.getElementById('uploadHint').value);

  try {
    const resp = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await resp.json();

    if (data.status === 'ok') {
      let html = '';
      if (data.dataset) {
        html += row('File',       escHtml(data.filename || selectedFile.name));
        html += row('Dataset',    '<span class="rc-value highlight">' + escHtml(data.dataset) + '</span>');
        html += row('Type',       escHtml(data.type || ''));
        if (data.version)     html += row('Version', 'v' + data.version + ' (' + escHtml(data.action || '') + ')');
        if (data.confidence)  html += row('Confidence', Math.round(data.confidence * 100) + '%');
        if (data.is_temporal) html += row('Temporal', 'Yes &mdash; Event nodes will be extracted');
      } else {
        html = '<div>' + escHtml(data.message || 'Saved.') + '</div>';
      }
      showUploadResult('ok', html);
      clearFile();
    } else {
      showUploadResult('err', '<span class="rc-err">Error: ' + escHtml(data.message || 'Unknown error') + '</span>');
    }
  } catch (err) {
    showUploadResult('err', '<span class="rc-err">' + escHtml(err.message) + '</span>');
  }
  uploadBtn.disabled = false;
}

function row(label, value) {
  return '<div class="rc-row"><span class="rc-label">' + label + '</span><span class="rc-value">' + value + '</span></div>';
}

function showUploadResult(cls, html) {
  uploadResult.className = 'result-card ' + cls;
  uploadResult.innerHTML = html;
  uploadResult.style.display = 'block';
}

// ── paste cognify ──

cognifyBtn.addEventListener('click', async () => {
  const content = docInput.value.trim();
  if (!content) { setStatus('Please paste some content first.', 'err'); return; }
  cognifyBtn.disabled = true;
  setStatus('<span class="spinner"></span> Sending to orchestrator&hellip;', 'running');

  try {
    const resp = await fetch('/api/cognify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content,
        hint: document.getElementById('pasteHint').value,
      }),
    });
    const data = await resp.json();
    if (data.status === 'ok') {
      let msg = data.dataset
        ? 'Saved to <strong>' + escHtml(data.dataset) + '</strong> (' + escHtml(data.type || '') + ')'
        : escHtml(data.message || 'Cognified successfully.');
      setStatus('&#x2713; ' + msg, 'ok');
      docInput.value = '';
    } else {
      setStatus('&#x2717; ' + escHtml(data.message || 'Unknown error'), 'err');
    }
  } catch (err) {
    setStatus('&#x2717; ' + escHtml(err.message), 'err');
  }
  cognifyBtn.disabled = false;
});

function setStatus(html, cls) {
  statusBox.className = 'status-box ' + cls;
  statusBox.innerHTML = html;
}

// ── chat ──

function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

function addMsg(role, html, id) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (id) div.id = id;
  const av = document.createElement('div');
  av.className = 'avatar';
  av.textContent = role === 'user' ? String.fromCodePoint(0x1F464) : String.fromCodePoint(0x1F916);
  const bub = document.createElement('div');
  bub.className = 'bubble';
  bub.innerHTML = html;
  div.appendChild(av);
  div.appendChild(bub);
  messagesEl.appendChild(div);
  scrollBottom();
  return bub;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function addToolPill(label, done) {
  const p = document.createElement('div');
  p.className = 'tool-pill ' + (done ? 'done' : 'running');
  p.innerHTML = '<span class="icon">' + (done ? '&#x2713;' : '<span class="spinner"></span>') + '</span>' + escHtml(label);
  messagesEl.appendChild(p);
  scrollBottom();
  return p;
}

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || busy) return;

  busy = true;
  sendBtn.disabled = true;
  chatInput.value = '';
  chatInput.style.height = 'auto';

  addMsg('user', escHtml(text));
  const agentBubble = addMsg('agent', '<span class="spinner"></span>');
  let agentText = '';
  const pills = {};

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    if (!sessionId) sessionId = resp.headers.get('X-Session-Id') || '';

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });

      const lines = buf.split('\n\n');
      buf = lines.pop();

      for (const chunk of lines) {
        const eventLine = chunk.split('\n').find(l => l.startsWith('event:'));
        const dataLine  = chunk.split('\n').find(l => l.startsWith('data:'));
        if (!eventLine || !dataLine) continue;

        const event = eventLine.slice(6).trim();
        let data;
        try { data = JSON.parse(dataLine.slice(5).trim()); } catch { continue; }

        if (event === 'tool_start') {
          const label = data.tool + '(' + Object.keys(data.args || {}).join(', ') + ')';
          pills[data.tool] = addToolPill(label, false);
          agentBubble.innerHTML = '<span class="spinner"></span>';
        }
        if (event === 'tool_done') {
          const pill = pills[data.tool];
          if (pill) {
            pill.className = 'tool-pill done';
            pill.querySelector('.icon').innerHTML = '&#x2713;';
          }
        }
        if (event === 'answer') {
          agentText = data.text;
          agentBubble.innerHTML = escHtml(agentText).replace(/\n/g, '<br>');
        }
      }
    }
    if (!agentText) agentBubble.innerHTML = '(no response)';

  } catch (err) {
    agentBubble.innerHTML = '<span style="color:var(--danger)">Error: ' + escHtml(err.message) + '</span>';
  }

  busy = false;
  sendBtn.disabled = false;
  scrollBottom();
}

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});
sendBtn.addEventListener('click', sendMessage);

// ── datasets / graph ──

document.getElementById('graphBtn').addEventListener('click', () => {
  window.open('/api/graph?t=' + Date.now(), '_blank');
});

datasetsBtn.addEventListener('click', async () => {
  datasetsResult.classList.toggle('visible');
  if (!datasetsResult.classList.contains('visible')) return;
  datasetsResult.textContent = 'Loading…';
  try {
    const resp = await fetch('/api/datasets');
    const data = await resp.json();
    datasetsResult.textContent = data.result || data.error || 'Empty.';
  } catch (err) {
    datasetsResult.textContent = 'Error: ' + err.message;
  }
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set (check .env or environment)")
        sys.exit(1)
    print("Starting agent UI on http://localhost:8080")
    uvicorn.run("agent_server:app", host="0.0.0.0", port=8080, reload=False)
