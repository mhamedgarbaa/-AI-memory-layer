## AI memory layer
# Cognee Memory MCP Stack

An enterprise-grade, graph-backed temporal memory microservice for AI agents.
Exposes two MCP entry-points: a **4-tool Orchestrator** (`:8003`) that auto-classifies
documents and routes queries, and a **raw 18-tool Gateway** (`:8002`) for direct control.

---

## Architecture

```
 ┌──────────────────────────────────────┐
 │   AI Agent / Web UI / CLI            │
 └────────────────┬─────────────────────┘
                  │ MCP  JSON-RPC 2.0 + SSE
                  ▼
 ┌──────────────────────────────────────┐
 │   Orchestrator Agent   :8003         │  ← 4 smart tools
 │                                      │    auto-classify → ds_policy / ds_market …
 │   save / search / improve / delete   │    temporal detection + versioning
 │   LangGraph StateGraphs              │    search-type auto-routing
 └────────────┬─────────────────────────┘
              │ JSON-RPC 2.0
              ▼
 ┌──────────────────────────────────────┐
 │   MCP Wrapper   :8002                │  ← 18 tools, session management
 │                                      │    circuit breaker, retry, docker-exec
 └────────────┬─────────────────────────┘
              │ proxy + docker exec
              ▼
 ┌─────────────────────────┐
 │  Cognee MCP   :8001     │  ← ECL pipeline: chunk→embed→extract→graph
 └────────────┬────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
  Neo4j 5          Postgres 15        LanceDB
  (graph)          (metadata)         (vectors, file-backed)

 ┌──────────────────────────────────────┐
 │  FastAPI Web   :8000                 │  ← REST API + graph UI
 └──────────────────────────────────────┘

 ┌──────────────────────────────────────┐
 │  Agent Server  :8080  (local)        │  ← Chat UI + file upload panel
 └──────────────────────────────────────┘
```

Six Docker services boot in dependency order:
`db` → `neo4j` → `cognee-mcp` → `mcp-wrapper` → `orchestrator` → `web`

---

## Quick Start

### Prerequisites

- Docker Desktop (or Docker + Compose v2)
- OpenAI or Azure OpenAI API key

### 1. Clone and configure

```bash
git clone <repo-url>
cd cognee-layer
cp .env.example .env   # edit .env
```

Minimum required in `.env`:

```env
LLM_PROVIDER=openai
LLM_MODEL=openai/gpt-4o-mini
LLM_ENDPOINT=https://api.openai.com/v1
LLM_API_KEY=<key>

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_ENDPOINT=https://api.openai.com/v1
EMBEDDING_API_KEY=<key>
EMBEDDING_DIMENSIONS=1536

POSTGRES_USER=workspace_user
POSTGRES_PASSWORD=workspace_pass
NEO4J_PASSWORD=neo4j_pass
```

### 2. Start the stack

```bash
docker compose up -d
```

Wait ~60 s for `cognee-mcp` to become healthy, then verify:

```bash
docker compose ps
curl http://localhost:8003/health   # Orchestrator
curl http://localhost:8002/health   # MCP Wrapper
curl http://localhost:8000/health   # FastAPI Web
```

### 3. Launch the web UI agent (local)

```bash
pip install -r requirements.txt
# Set GROQ_API_KEY in .env
python agent_server.py
# open http://localhost:8080
```

---

## Orchestrator Tools (`:8003`)

Connect any MCP agent to `http://localhost:8003`. Four tools replace all 18:

| Tool | Description |
|------|-------------|
| `save` | Store any text or document. Auto-classifies into the right typed dataset, detects temporal content, versions duplicates. |
| `search` | Query memory. Auto-detects temporal queries, picks best search strategy. |
| `improve` | Run the quality improvement loop across all datasets. |
| `delete` | Delete a dataset, a specific data item by UUID, or wipe all (`all`). |

### Typed datasets (auto-assigned)

| Dataset | Content type |
|---------|-------------|
| `ds_policy` | Policies, procedures, compliance, guidelines |
| `ds_regulation` | Laws, GDPR, ISO standards, legal directives |
| `ds_employees` | HR, staff, salaries, hiring, performance |
| `ds_market` | Revenue, sales, competitors, Q1–Q4 forecasts |
| `ds_workflow` | BPMN, processes, task sequences |
| `ds_contracts` | Supplier/vendor contracts, SLAs |
| `ds_reports` | Analysis reports, KPI dashboards, audit findings |

Pass `hint` to override: `{"data": "...", "hint": "contracts"}`.

### Session handshake (orchestrator)

```python
import httpx, json

BASE = "http://localhost:8003"
c = httpx.Client(timeout=180)

# 1 — initialize
r = c.post(f"{BASE}/mcp", json={"jsonrpc":"2.0","id":1,"method":"initialize",
    "params":{"protocolVersion":"2024-11-05","capabilities":{},
              "clientInfo":{"name":"my-agent","version":"1.0"}}})
sid = r.headers["mcp-session-id"]
h = {"mcp-session-id": sid}

# 2 — save (auto-classified)
c.post(f"{BASE}/mcp", headers=h, json={"jsonrpc":"2.0","id":2,"method":"tools/call",
    "params":{"name":"save","arguments":{"data":"Q3 2024 revenue was 12.4M, up 18% YoY."}}})

# 3 — search (auto-routed)
r = c.post(f"{BASE}/mcp", headers=h, json={"jsonrpc":"2.0","id":3,"method":"tools/call",
    "params":{"name":"search","arguments":{"query":"What was Q3 revenue?"}}})
for line in r.text.splitlines():
    if line.startswith("data:"):
        print(json.loads(line[5:])["result"]["content"][0]["text"])
```

---

## Raw Gateway Tools (`:8002`)

For direct control, connect to `http://localhost:8002`. Full 18-tool API:

### Write
| Tool | Latency | Description |
|------|---------|-------------|
| `cognify` | ~30 s | Deep-process text. Pass `dataset_name` to route to a typed dataset. |
| `save_interaction` | ~2 s | Fast-append a short note. Also accepts `dataset_name`. |
| `memify` | ~120 s | Extract temporal Event nodes from stored chunks. |
| `improve` | ~60 s | Feedback weights → graph persistence → trace sync → session index. |

### Read
| Tool | Latency | Description |
|------|---------|-------------|
| `search` | ~5 s | 14 search strategies (see below). Accepts `dataset_name` to scope. |
| `list_data` | ~2 s | List all datasets and data items. |
| `cognify_status` | ~1 s | Check pipeline status. |
| `visualize_graph` | ~15 s | Render Neo4j graph as interactive HTML. |

### Delete
| Tool | Latency | Description |
|------|---------|-------------|
| `delete` | ~3 s | Delete a data item or dataset (soft/hard). |
| `prune` | ~5 s | Wipe ALL knowledge — irreversible. |

### Dataset management
| Tool | Description |
|------|-------------|
| `create_dataset` | Register a dataset name for the session. |
| `switch_dataset` | Switch active dataset for subsequent calls. |
| `delete_dataset` | Remove a dataset and all its data. |

### 14 Search types

`GRAPH_COMPLETION` · `GRAPH_COMPLETION_COT` · `GRAPH_COMPLETION_CONTEXT_EXTENSION` ·
`GRAPH_COMPLETION_DECOMPOSITION` · `GRAPH_SUMMARY_COMPLETION` · `TEMPORAL` ·
`TRIPLET_COMPLETION` · `CYPHER` · `NATURAL_LANGUAGE` · `RAG_COMPLETION` ·
`SUMMARIES` · `CHUNKS` · `CHUNKS_LEXICAL` · `FEELING_LUCKY`

---

## File Upload & Cognify (agent_server.py)

The web UI at `:8080` supports uploading files directly to the knowledge graph:

- **Formats:** `.pdf`, `.csv`, `.xlsx`, `.txt`, `.json`, `.md`, `.png`, `.jpg`
- **PDF** — text extracted via `pypdf`, sent to orchestrator for classification
- **CSV / Excel** — rows read and forwarded as structured text
- **Images** — filename + note stored (full vision requires a shared volume)
- **Hint** — optional dropdown to override auto-classification

All uploads route through the orchestrator `save` tool — documents land in the correct typed dataset automatically.

---

## Integration Tests

```bash
# Orchestrator unit + integration tests
python scripts/test_orchestrator.py              # full suite (Docker required)
python scripts/test_orchestrator.py --unit-only  # unit tests only

# Stack integration tests
pytest tests/test_stack_integration.py -v

# Streaming smoke test
python scripts/test_streaming.py
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `list_data` only shows `main_dataset` | Data was saved without `dataset_name`. Re-save via orchestrator to get typed datasets. |
| Orchestrator not reachable | `docker compose up -d orchestrator` — it depends on mcp-wrapper healthy |
| `421 Invalid Host header` | Use MCP Wrapper (`:8002`), not Cognee MCP (`:8001`) directly |
| `circuit_breaker_open: true` | Cognee MCP unreachable — check `docker compose logs cognee-mcp` |
| `Unknown or expired session` | Call `initialize` first; include `mcp-session-id` on all subsequent requests |
| Embedding dimension mismatch | Wipe volumes: `docker volume rm cognee-layer_cognee_data cognee-layer_cognee_storage` |
| Ontology not picked up | `POST /admin/reload-cognee` on the wrapper |

---

## Neo4j Browser

```
http://localhost:7474
Login: neo4j / neo4j_pass
```

```cypher
// All entities
MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 100

// Temporal Event nodes (after memify)
MATCH (e:Event) RETURN e ORDER BY e.timestamp DESC LIMIT 20

// Data by typed dataset
MATCH (n:TextDocument) RETURN n.dataset_name, count(n) ORDER BY count(n) DESC
```
