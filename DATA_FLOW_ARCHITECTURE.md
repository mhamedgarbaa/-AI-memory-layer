# Data Flow Architecture

## High-Level Overview

```
┌──────────────────────────────────────────────────────────────────┐
│              EXTERNAL AGENTS / WEB UI / CLI                      │
│       (AI agents, LangChain, custom HTTP clients, agent_server)  │
└────┬─────────────────────────────────────────────────────────────┘
     │ HTTP JSON-RPC 2.0 + SSE
     ▼
┌──────────────────────────────────────────────────────────────────┐
│          ORCHESTRATOR AGENT  :8003  (orchestrator.py)            │
│                                                                  │
│  POST /mcp  — initialize / tools/list / tools/call               │
│  GET  /health                                                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  4 smart tools: save · search · improve · delete        │     │
│  │  LangGraph StateGraphs (save_graph, search_graph, ...)  │     │
│  │  Document classifier  → 7 typed datasets                │     │
│  │  Query analyzer       → intent + temporal detection     │     │
│  │  Version manager      → search-before-save dedup        │     │
│  │  Memory router        → best search_type per intent     │     │
│  │  Session TTL eviction (default 3600s)                   │     │
│  └─────────────────────────────────────────────────────────┘     │
└────┬─────────────────────────────────────────────────────────────┘
     │ JSON-RPC 2.0  (WrapperClient singleton, circuit breaker)
     ▼
┌──────────────────────────────────────────────────────────────────┐
│             MCP WRAPPER  :8002  (mcp_wrapper.py)                 │
│                                                                  │
│  GET  /tools  — tool catalogue + session handshake guide         │
│  GET  /health — wrapper + circuit breaker status                 │
│  POST /mcp    — initialize / tools/list / tools/call             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │ 18 tools: cognify · search · save_interaction · memify  │     │
│  │           improve · list_data · delete · prune          │     │
│  │           create_dataset · switch_dataset · delete_dataset│    │
│  │           cognify_status · visualize_graph · ...        │     │
│  │ Session layer      mcp-session-id UUID per caller       │     │
│  │ Circuit breaker    opens after 3 failures / 60s timeout │     │
│  │ Background cognify ThreadPoolExecutor (4 workers)       │     │
│  └─────────────────────────────────────────────────────────┘     │
└────┬───────────────────────────┬─────────────────────────────────┘
     │ JSON-RPC proxy            │ docker exec (dataset cognify + tools)
     ▼                           ▼
┌─────────────────┐    ┌──────────────────────────────────────────┐
│  COGNEE MCP     │    │  cognee_mcp_server container             │
│  :8001 internal │    │  cognify_dataset  — cognee.add/cognify   │
│  :8000 in Docker│    │  memify           — temporal enrichment  │
│                 │    │  visualize_graph  — Neo4j → HTML render  │
│  MCP tools      │    └──────────────────────────────────────────┘
│  HTTP/SSE :8000 │
└────┬────────────┘
     ▼
┌──────────────────────────────────────────────────────────────────┐
│                      COGNEE CORE ENGINE                          │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  Postgres 15 │   │   Neo4j 5    │   │  OpenAI / Azure OAI  │ │
│  │  cognee_db   │   │  bolt::7687  │   │  LLM + embeddings    │ │
│  │  (metadata,  │   │  (entity     │   │  (extraction,        │ │
│  │   principals)│   │   graph,     │   │   synthesis,         │ │
│  │              │   │   summaries, │   │   Cypher generation) │ │
│  │              │   │   Event nodes│   │                      │ │
│  └──────────────┘   └──────────────┘   └──────────────────────┘ │
│                                                                  │
│  ┌──────────────┐                                                │
│  │  LanceDB     │                                                │
│  │  vector store│                                                │
│  │  (file-backed│                                                │
│  └──────────────┘                                                │
└──────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  FASTAPI WEB  :8000  (app.py)                                  │
│  REST API /api/v1/memory/*  · graph UI · ontology endpoints    │
│  Reads Neo4j directly for stats/visualization                  │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  AGENT SERVER  :8080  (agent_server.py)  — local only          │
│  Chat UI · file upload panel (PDF/CSV/XLSX/images)             │
│  Connects to orchestrator :8003 via MCP                        │
└────────────────────────────────────────────────────────────────┘
```

Boot order: `db` → `neo4j` → `cognee-mcp` → `mcp-wrapper` → `orchestrator` → `web`

---

## 1. Session Handshake

Both orchestrator (:8003) and mcp-wrapper (:8002) require the same MCP handshake.

```
Agent                        Orchestrator (:8003)
  │                                  │
  │  POST /mcp                       │
  │  { method: "initialize",         │
  │    clientInfo: { name: "..." },  │
  │    protocolVersion: "2024-11-05"}│
  │ ────────────────────────────────►│  creates session UUID
  │                                  │  _sessions[uuid] = { ... }
  │  200  mcp-session-id: <uuid>     │
  │ ◄────────────────────────────────│
  │                                  │
  │  POST /mcp                       │
  │  mcp-session-id: <uuid>          │
  │  { method: "notifications/       │
  │    initialized" }                │
  │ ────────────────────────────────►│  (required handshake step)
  │                                  │
  │  POST /mcp                       │
  │  mcp-session-id: <uuid>          │
  │  { method: "tools/call",         │
  │    params: { name: "save",       │
  │      arguments: { data: "..." }  │
  │  } }                             │
  │ ────────────────────────────────►│  dispatch → LangGraph pipeline
  │  SSE: data: { result: { content: │
  │    [{ type: "text", text: "..." }│
  │  ] } }                           │
  │ ◄────────────────────────────────│
```

Sessions are evicted after `SESSION_TTL` (default 3600s) by a background loop.
The orchestrator's `WrapperClient` singleton reconnects automatically on 400 (stale session).

---

## 2. Typed Datasets

The document classifier assigns every document to one of 7 typed datasets:

| Dataset | Content type | Keyword signals |
|---------|-------------|-----------------|
| `ds_policy` | Policies, procedures, compliance, guidelines | policy, procedure, guideline, rule |
| `ds_regulation` | Laws, GDPR, ISO standards, legal directives | law, regulation, decree, GDPR, ISO |
| `ds_employees` | HR, staff, salaries, hiring, performance | employee, HR, staff, salary, hiring |
| `ds_market` | Revenue, sales, competitors, Q1–Q4 forecasts | revenue, market, sales, competitor |
| `ds_workflow` | BPMN, processes, task sequences | process, workflow, step, task, BPMN |
| `ds_contracts` | Supplier/vendor contracts, SLAs | contract, agreement, supplier, SLA |
| `ds_reports` | Analysis reports, KPI dashboards, audit findings | report, analysis, findings, KPI |

Classification is a two-pass pipeline:
1. **Keyword score** — if top score ≥ 2 hits, use it (confidence = min(hits/5, 0.9))
2. **LLM fallback** — if confidence < 0.6, ask LLM for JSON `{type_label, confidence}`

---

## 3. Write Path — `save` (orchestrator)

```
Agent POST /mcp  tools/call  name=save  arguments={data: "...", hint?: "..."}
  │
  ▼
Orchestrator: LangGraph save_graph
  │
  ├─ classify_node
  │    keyword pass → LLM fallback → DocumentType{dataset_name, type_label, confidence}
  │    temporal detection (regex: years, months, Q1-Q4, ISO dates)
  │
  ├─ override_hint_node  [only if hint= provided]
  │    matches hint against TAXONOMY, overrides classifier result
  │
  ├─ ensure_dataset_node
  │    WrapperClient.call("create_dataset", {dataset_name})  [idempotent]
  │
  ├─ check_version_node
  │    WrapperClient.call("search", {query: text[:200], search_type: "CHUNKS_LEXICAL"})
  │    if empty → action="create", version=1
  │    if found  → scan for [VERSION:N] tags → version=max+1
  │    appends: \n\n[VERSION:N][DATASET:ds_X][UPDATED:iso_timestamp]
  │
  └─ cognify_node
       WrapperClient.call("cognify", {data, temporal, dataset_name}, timeout=30)
         │
         ▼  mcp-wrapper receives cognify call
         │  detects dataset_name param → background path
         │
         ├─ returns immediately  {"status": "queued"}  (< 2ms)
         │
         └─ _fire_cognify_bg() → ThreadPoolExecutor.submit(_cognify_bg_thread)
              │  (OS thread — survives request scope)
              ├─ tar cognify_dataset_run.py → container.put_archive /tmp
              └─ container.exec_run("python3 /tmp/cognify_dataset_run.py")
                   │  (inside cognee_mcp_server container)
                   ├─ await cognee.run_migrations()   ← recreates schema after prune
                   ├─ await cognee.add(data, dataset_name=ds_X)
                   └─ await cognee.cognify(datasets=[ds_X], temporal_cognify=T)
                          │
                          ├─ chunk text
                          ├─ LLM entity extraction
                          ├─ generate embeddings → LanceDB
                          ├─ write entity/relationship nodes → Neo4j
                          └─ write metadata → Postgres

Orchestrator returns to caller immediately:
  { status: "processing", dataset: "ds_X", type: "...", version: N,
    action: "create|update", is_temporal: bool, classifier_confidence: 0.XX }

Background thread logs (mcp-wrapper):
  "background cognify done dataset=ds_X"    ← success (30–420s later)
  "background cognify failed (exit 1) ..."  ← failure
```

---

## 4. Read Path — `search` (orchestrator)

```
Agent POST /mcp  tools/call  name=search  arguments={query: "...", hint?: "..."}
  │
  ▼
Orchestrator: LangGraph search_graph
  │
  ├─ analyze_node
  │    temporal detection (regex pass, no LLM)
  │    LLM classification → QueryIntent{intent, search_type, is_temporal, confidence}
  │
  ├─ route_node  (memory_router)
  │    is_temporal → TEMPORAL
  │    intent==summary → GRAPH_SUMMARY_COMPLETION
  │    intent==cypher  → CYPHER
  │    default         → GRAPH_COMPLETION
  │
  └─ search_node
       WrapperClient.call("search",
         {search_query: query, search_type: route.search_type, dataset_name?: ...})
         │
         ▼
       MCP Wrapper → proxy.call() → Cognee MCP search pipeline
         ├─ embed query → LanceDB (vector similarity, top-K candidates)
         ├─ graph traversal → Neo4j (Bolt)
         │    MATCH (a)-[r]->(b) WHERE a.id IN [candidates]
         └─ LLM synthesis → natural-language answer

Typical latency: 5–15s
```

### Search Type Routing Rules

| Query pattern | Routed to | Backends |
|--------------|-----------|---------|
| Time references, dates, "when", "history" | `TEMPORAL` | Neo4j Event nodes |
| "Overview", "summary", "what is X" | `GRAPH_SUMMARY_COMPLETION` | LanceDB + Neo4j + summary nodes |
| Deep reasoning, "how", "why" | `GRAPH_COMPLETION_COT` | LanceDB + Neo4j (chain-of-thought) |
| "Find exactly", specific fact | `CYPHER` | Neo4j direct (LLM-generated Cypher) |
| Default | `GRAPH_COMPLETION` | LanceDB + Neo4j |

---

## 5. Improve Path — `improve` (orchestrator)

```
Agent POST /mcp  tools/call  name=improve  arguments={dataset_name?: "..."}
  │
  ▼
Orchestrator: dispatch
  │
  ├─ dataset_name == "all" (default)
  │    WrapperClient.call("improve", {})
  │
  └─ specific dataset
       WrapperClient.call("switch_dataset", {dataset_name})
       WrapperClient.call("improve", {dataset_name})
         │
         ▼
       MCP Wrapper → Cognee MCP improve pipeline
         ├─ re-weight graph edges from feedback scores
         ├─ persist session learnings to permanent graph
         └─ rebuild session index

Typical latency: 60s
```

---

## 6. Delete Path — `delete` (orchestrator)

```
Agent POST /mcp  tools/call  name=delete
  arguments={ target: "all|<dataset>|<uuid>", mode?: "soft|hard" }
  │
  ▼
Orchestrator dispatch
  │
  ├─ target == "all"
  │    WrapperClient.call("prune", {})
  │    ⚠ wipes entire cognee DB schema — next save auto-recreates via run_migrations()
  │
  ├─ target matches UUID regex [0-9a-f-]{36}
  │    WrapperClient.call("delete", {data_id: target, mode})
  │
  └─ target is dataset name
       WrapperClient.call("delete_dataset", {dataset_name: target})
         (retries once on 400 — stale session auto-refreshed)
```

---

## 7. Temporal Enrichment Path — `memify`

```
Agent POST /mcp  tools/call  name=memify  arguments={dataset?: "..."}
  │
  ▼
MCP Wrapper dispatch → run_subprocess_tool()
  │
  ├─ asyncio.run_in_executor → thread pool
  │   docker exec cognee_mcp_server python3 scripts/run_memify.py
  │
  └─ LLM reads DocumentChunk nodes → extracts Event{timestamp, description}
     writes Event nodes to Neo4j
     enables TEMPORAL search type for date-range queries

Typical latency: 60–180s
```

---

## 8. Graph Visualization Path — `visualize_graph`

```
Agent POST /mcp  tools/call  name=visualize_graph
  │
  ▼
MCP Wrapper dispatch → run_subprocess_tool()
  │
  ├─ docker exec cognee_mcp_server python3 scripts/visualize_neo4j_graph.py
  │   Cypher: MATCH (a)-[r]->(b) RETURN a,r,b  (hides orphan nodes)
  │   Renders interactive HTML with legend panel
  │
  └─ Output → /graph/cognee_graph.html
     (graph_output volume shared between mcp-wrapper and web)

Typical latency: 15s
```

---

## 9. Circuit Breaker — Two Levels

### mcp-wrapper → cognee-mcp

```
CLOSED ──(3 consecutive failures)──► OPEN
  ▲                                    │
  └────(success after 60s timeout)─────┘
```

Config: `CIRCUIT_BREAKER_THRESHOLD=3`, `CIRCUIT_BREAKER_TIMEOUT=60`

### orchestrator → mcp-wrapper (WrapperClient)

```
call()
  │
  ├─ circuit open? → raise "circuit_breaker_open"
  │
  ├─ attempt 0 → POST /mcp
  │    400 response (stale session) → clear _session_id, retry once
  │    other error → record_failure(), raise
  │
  └─ attempt 1 → fresh ensure_session() → POST /mcp
```

---

## 10. Background Cognify Thread Model

```
HTTP request scope (anyio)
  │
  ├─ cognify_node await wrapper.call("cognify", ..., timeout=30)
  │    returns in < 2ms  ← mcp-wrapper fires background thread and responds
  │
  └─ request scope closes ← asyncio tasks would be CANCELLED here

mcp-wrapper process (OS level)
  │
  └─ _bg_executor = ThreadPoolExecutor(max_workers=4)
       _bg_executor.submit(_cognify_bg_thread, payload)
         │  runs as OS thread — not subject to anyio cancellation
         ├─ tar script → put_archive → exec_run (blocking, up to ~420s)
         └─ logs: "background cognify done|failed dataset=X"
```

---

## 11. FastAPI Web Layer — `/api/v1/memory/*`

| API Endpoint | Service Method | MCP Tool (via mcp-wrapper) |
|---|---|---|
| `POST /api/v1/memory/record` | `record_agent_memory()` | `save_interaction` + `cognify` |
| `POST /api/v1/memory/recall` | `recall_context()` | `search` (3-tier fallback) |
| `GET /api/v1/memory/data` | `get_data_inventory()` | `list_data` |
| `DELETE /api/v1/memory/data` | `delete_memory_data()` | `delete` |
| `DELETE /api/v1/memory/prune` | `prune_all_memory()` | `prune` |
| `GET /api/v1/memory/cognify/status` | `get_cognify_status()` | `cognify_status` |

### Recall 3-tier degradation

```
recall_context()
  ├─ try GRAPH_COMPLETION  → ContextEnvelope(is_degraded=False)
  │    failure ↓
  ├─ try CHUNKS            → ContextEnvelope(is_degraded=True, method="CHUNKS_FALLBACK")
  │    failure ↓
  └─ return                  ContextEnvelope(is_degraded=True, method="NONE_FAILED")
```

---

## 12. Environment Variables Affecting Data Flow

```env
# Orchestrator
MCP_WRAPPER_URL=http://mcp-wrapper:8002
ORCHESTRATOR_SESSION_TTL=3600
LLM_API_KEY=<key>
LLM_MODEL=openai/gpt-4o-mini
LLM_ENDPOINT=https://api.openai.com/v1

# MCP Wrapper → Cognee MCP connection
MCP_SERVER_URL=http://cognee-mcp:8000
MCP_HOST_HEADER=localhost:8001
MCP_TIMEOUT=300

# Session management
MCP_WRAPPER_SESSION_TTL=3600
MCP_WRAPPER_STRICT_SESSIONS=false

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD=3
CIRCUIT_BREAKER_TIMEOUT=60

# Subprocess / docker exec tools
COGNEE_CONTAINER_NAME=cognee_mcp_server
GRAPH_OUTPUT_PATH=/graph/cognee_graph.html

# Cognee MCP backends
LLM_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
DB_PROVIDER=postgres    DB_HOST=db    DB_NAME=cognee_db
GRAPH_DATABASE_PROVIDER=neo4j    GRAPH_DATABASE_URL=bolt://neo4j:7687
VECTOR_DB_PROVIDER=lancedb
ENABLE_BACKEND_ACCESS_CONTROL=False
ACCEPT_LOCAL_FILE_PATH=True
```

---

## 13. Complete Request/Response Table

| # | Component | In | Out |
|---|-----------|---|-----|
| 1 | External Agent | HTTP POST /mcp | JSON-RPC 2.0 request |
| 2 | Orchestrator session layer | request | validated session UUID |
| 3 | Orchestrator LangGraph | tool_name + args | StateGraph execution |
| 4 | Document classifier | text | dataset_name + type_label |
| 5 | Version manager | text + dataset | versioned_text |
| 6 | WrapperClient | tool + args | SSE text response |
| 7 | MCP Wrapper session layer | request | validated session UUID |
| 8 | MCP Wrapper dispatch | tool_name + args | proxy or docker exec |
| 9a | Proxy path | JSON-RPC to Cognee MCP :8000 | JSON-RPC response |
| 9b | Background thread | docker exec + tar | stdout (30–420s) |
| 10 | Cognee MCP | tool call | DB ops + LLM calls |
| 11 | Neo4j / Postgres / LanceDB | Bolt/SQL/file | query results |
| 12 | Orchestrator SSE | result text | `data: {...}\n\n` |
| 13 | External Agent | SSE event | content[0].text |

---

## 14. Migration Notes

### v3.0 → v4.0 (2026-05-14)

| Area | v3.0 | v4.0 |
|------|------|------|
| Agent entry point | MCP Wrapper :8002 (18 tools) | Orchestrator :8003 (4 tools) |
| Document routing | All data → `main_dataset` | Auto-classified → 7 typed datasets |
| Cognify latency (caller) | 30–420s blocking | < 2ms (background thread) |
| Versioning | None | search-before-save, `[VERSION:N]` tags |
| Temporal detection | Manual `search_type=TEMPORAL` | Auto-detected from query/content |
| Search routing | Caller chooses search_type | Memory router selects best type |
| Post-prune recovery | Manual restart required | `run_migrations()` auto-recreates schema |
| LangGraph | Not used | save_graph, search_graph StateGraphs |
| Session retry | Single attempt | Retry-on-400 with fresh session |

### v2.0 → v3.0 (2026-04-19)

| Area | v2.0 | v3.0 |
|------|------|------|
| Agent entry point | Direct Cognee MCP :8001 | MCP Wrapper :8002 |
| Tool count | 7 (Cognee native) | 18 (+ memify, improve, dataset mgmt) |
| Session management | None | `mcp-session-id` UUID, TTL eviction |
| Resilience | App-level circuit breaker | Wrapper circuit breaker + backoff |
| Graph visualization | `docker cp` to host | Shared `graph_output` volume |
