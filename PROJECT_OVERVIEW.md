# Cognee-Layer — Complete Technical Reference

> **Version:** 4.0.0 &nbsp;|&nbsp; **Status:** Production Ready &nbsp;|&nbsp; **Date:** 2026-05-14

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [System Architecture](#2-system-architecture)
3. [Docker Services](#3-docker-services)
4. [File & Folder Map](#4-file--folder-map)
5. [Core Application Layer](#5-core-application-layer)
6. [Orchestrator Agent](#6-orchestrator-agent)
7. [API Layer](#7-api-layer)
8. [Service Layer](#8-service-layer)
9. [Data Layer](#9-data-layer)
10. [Configuration](#10-configuration)
11. [Scripts & Tooling](#11-scripts--tooling)
12. [Knowledge Graph & Visualization](#12-knowledge-graph--visualization)
13. [Orchestrator Tools Reference](#13-orchestrator-tools-reference)
14. [MCP Gateway Tools Reference](#14-mcp-gateway-tools-reference)
15. [REST API Reference](#15-rest-api-reference)
16. [LLM Backends](#16-llm-backends)
17. [Data Flow — End-to-End](#17-data-flow--end-to-end)
18. [Tech Stack Summary](#18-tech-stack-summary)

---

## 1. What is this project?

**Cognee-Layer** is an enterprise-grade, graph-backed **temporal memory microservice** for AI agents.

It gives any AI agent (LangChain, AutoGen, CrewAI, custom agents…) a persistent memory layer that:

- Stores raw text and **extracts entities, relationships, and temporal events** into a knowledge graph
- **Auto-classifies documents** into typed datasets (policies, regulations, contracts, market data…)
- Answers semantic queries using **14 search strategies** (graph completion, vector RAG, Cypher, temporal, etc.)
- Tracks feedback, agent traces, and improves retrieval quality in a continuous loop
- Exposes everything through **MCP (Model Context Protocol)** — the standard AI agent tool protocol

The system is self-contained: spin it up with `docker compose up -d` and point any MCP-compatible agent at:
- **`http://localhost:8003`** — Orchestrator (4 smart tools, recommended)
- **`http://localhost:8002`** — Raw MCP Gateway (18 tools, direct control)

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│          AI Agent / Web UI / CLI / LangChain / AutoGen        │
└───────────────────────────┬──────────────────────────────────┘
                            │  MCP  (JSON-RPC 2.0 + SSE)
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              Orchestrator Agent   :8003                      │
│                                                              │
│   save · search · improve · delete  (4 smart tools)          │
│                                                              │
│    ┌─────────────────────────────────┐                       │
│    │ Orchestrator Modules            │                       │
│    │  document_classifier            │                       │
│    │  query_analyzer (temporal)      │                       │
│    │  version_manager                │                       │
│    │  memory_router                  │                       │
│    └─────────────────────────────────┘                       │
└───────────────────────────┬──────────────────────────────────┘
                            │  JSON-RPC 2.0
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              MCP Gateway   :8002                             │
│   18 tools · session management · circuit breaker            │
│   docker-exec dataset cognify · retry logic · SSE proxy      │
└───────────────────────────┬──────────────────────────────────┘
                            │  proxy + docker exec
                            ▼
┌────────────────────────────┐
│   Cognee MCP    :8001      │  ← core memory engine (ECL pipeline)
│   (cognee/cognee-mcp)      │    chunk → embed → extract → graph
└──────┬──────────┬──────────┘
       │          │
       ▼          ▼
  Neo4j 5     Postgres 15         LanceDB (file-backed vectors)
  :7687/7474  (cognee_db)         1536D or 768D embeddings
  Knowledge   Metadata / datasets
  Graph

┌────────────────────────────┐
│   FastAPI Web   :8000      │  ← REST memory API + graph viewer UI
│   Swagger at /api/docs     │    ontology upload, graph build, proxies
└────────────────────────────┘

┌────────────────────────────┐
│   Agent Server  :8080      │  ← Web chat UI + file upload panel
│   (local, not Docker)      │    SSE streaming, orchestrator routing
└────────────────────────────┘
```

**Boot order:** `db` → `neo4j` → `cognee-mcp` → `mcp-wrapper` → `orchestrator` → `web`

---

## 3. Docker Services

| Service | Image | Port(s) | Role |
|---|---|---|---|
| **db** | `postgres:15` | `5432` | Relational store — workspaces, metadata, Cognee datasets |
| **neo4j** | `neo4j:5` | `7687` (Bolt) `7474` (UI) | Knowledge graph — entities, relationships, temporal events |
| **cognee-mcp** | `Dockerfile.cognee` | `8001` | Core AI memory engine — ECL pipeline, MCP protocol |
| **mcp-wrapper** | `Dockerfile.wrapper` | `8002` | Gateway — 18 tools, session management, circuit breaker, docker-exec dataset routing |
| **orchestrator** | `Dockerfile.orchestrator` | `8003` | Orchestrator — 4 smart tools, LangGraph pipelines, auto-classification |
| **web** | `Dockerfile` | `8000` | FastAPI REST API, graph viewer, ontology management |

### Dockerfile breakdown

| File | Base | What it adds |
|---|---|---|
| `Dockerfile` | `python:3.11-slim` | FastAPI app, non-root user, uvicorn entrypoint |
| `Dockerfile.cognee` | `cognee/cognee-mcp:main` | graphiti-core, tiktoken/litellm Azure patches, sitecustomize fix |
| `Dockerfile.wrapper` | `python:3.11-slim` | MCP gateway, docker SDK, minimal footprint |
| `Dockerfile.orchestrator` | `python:3.11-slim` | Orchestrator agent, LangGraph, openai SDK |

---

## 4. File & Folder Map

```
cognee-layer/
│
├── app.py                        # FastAPI application factory (web service :8000)
├── mcp_wrapper.py                # MCP gateway service (port 8002) — 18 tools
├── orchestrator.py               # Orchestrator FastAPI MCP server (port 8003) — 4 tools
├── graph_builder.py              # Neo4j → NetworkX layout → vis-network HTML
│
├── orchestrator/                 # Orchestrator agent modules
│   ├── __init__.py
│   ├── mcp_client.py             # httpx client → mcp-wrapper, SSE parsing, circuit breaker
│   ├── query_analyzer.py         # Intent detection + temporal regex → QueryIntent
│   ├── document_classifier.py    # 7-type keyword + LLM taxonomy → DocumentType
│   ├── version_manager.py        # Search-before-save, version tagging → VersionResult
│   ├── memory_router.py          # QueryIntent → best search_type → RouteResult
│   └── graphs/
│       ├── save_graph.py         # LangGraph: classify → override_hint → ensure_dataset → check_version → cognify
│       ├── search_graph.py       # LangGraph: analyze → route → search → format
│       ├── improve_graph.py      # LangGraph: improve across all/one dataset
│       └── delete_graph.py       # LangGraph: prune / delete dataset / delete item
│
├── api/
│   ├── routers/
│   │   ├── v1/
│   │   │   ├── memory_router.py  # /api/v1/memory/* endpoints
│   │   │   └── workspace_router.py # /api/v1/workspaces/* CRUD
│   ├── schemas/
│   │   ├── memory_schema.py      # Pydantic request/response models
│   │   └── workspace_schema.py
│   ├── middlewares/
│   │   ├── logging_middleware.py
│   │   ├── error_handling_middleware.py
│   │   └── auth_middleware.py
│   ├── dependencies.py
│   └── custom_api_exceptions.py
│
├── service/
│   ├── memory_service.py         # 3-tier recall degradation
│   ├── workspace_service.py
│   ├── business_models.py
│   └── custom_service_exceptions.py
│
├── data/
│   ├── repositories/
│   │   ├── cognee_repository.py  # HTTP client → Cognee MCP (JSON-RPC 2.0)
│   │   └── workspace_repository.py
│   ├── models/
│   │   └── workspace_model.py
│   ├── bpi_ingest_registry.json
│   └── data_bpi/
│
├── configuration/
│   ├── settings.py
│   ├── database.py
│   └── logging_setup.py
│
├── scripts/
│   ├── run_memify.py             # Temporal event extraction (inside container)
│   ├── test_orchestrator.py      # Orchestrator unit + integration tests
│   ├── test_streaming.py         # SSE streaming smoke test
│   ├── test_graphiti_e2e.py
│   ├── test_azure_endpoint.py
│   ├── mcp_probe.py
│   └── postgres-init/
│       └── 01-create-cognee-db.sh
│
├── ontologies/
│   └── ontology.ttl              # RDF/Turtle domain ontology (financial)
│
├── lib/
│   ├── vis-9.1.2/                # Bundled vis-network (offline-capable)
│   └── tom-select/
│
├── tests/
│   ├── test_stack_integration.py
│   └── unit_tests/
│
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.cognee
├── Dockerfile.wrapper
├── Dockerfile.orchestrator
├── requirements.txt
├── requirements-wrapper.txt
├── requirements-orchestrator.txt
├── .env.example
└── README.md
```

---

## 5. Core Application Layer

### `app.py` — FastAPI Entrypoint (port 8000)

- Mounts all API routers (`/api/v1/memory`, `/api/v1/workspaces`)
- Serves the interactive graph viewer at `/graph`
- Proxies `/add` and `/search` to the MCP Wrapper
- Exposes ontology management endpoints (`/ontology/upload`, `/ontology/commit`)
- Queries Neo4j directly for stats (`/graph/stats`, `/graph/data`)

---

### `mcp_wrapper.py` — MCP Gateway (port 8002)

The central routing layer between the orchestrator and Cognee MCP. Exposes 18 tools.

**Key features:**

| Feature | Detail |
|---|---|
| **JSON-RPC 2.0** | Full `initialize` / `tools/call` / `tools/list` protocol |
| **Session management** | UUID4 session IDs via `mcp-session-id` header, TTL eviction |
| **SSE streaming** | Server-Sent Events forwarded to the caller |
| **Circuit breaker** | Opens after N failures to cognee-mcp, auto-recovers after timeout |
| **Dataset cognify** | Bypasses cognee-mcp MCP schema (which has no `dataset_name`) via `docker exec` — runs `cognee.add() + cognee.cognify()` directly in the container |
| **docker-exec tools** | `memify`, `visualize_graph`, dataset-scoped cognify |
| **Admin endpoint** | `POST /admin/reload-cognee` hot-reloads the ontology |

**Dataset routing fix:** The cognee-mcp `cognify` MCP tool accepts only `data`, `graph_model_file`, `graph_model_name`, `custom_prompt` — no `dataset_name`. To route documents into typed datasets, `mcp_wrapper.py` detects a `dataset_name` arg and runs `cognee.add(data, dataset_name=...) + cognee.cognify(datasets=[...])` via `docker exec` instead of the MCP tool.

---

### `orchestrator.py` — Orchestrator MCP Server (port 8003)

FastAPI MCP server exposing 4 simplified tools. Acts as the intelligent front-end to `mcp-wrapper:8002`.

- Implements full MCP session management (UUID sessions, TTL eviction)
- Dispatches to `LangGraph` pipelines in `orchestrator/graphs/`
- SSE responses identical to mcp-wrapper format
- `GET /health`, `POST /mcp` endpoints

---

### `graph_builder.py` — Knowledge Graph Renderer

1. Fetches nodes + edges from Neo4j via Bolt
2. Runs **NetworkX spring layout** (k=4.0, 120 iterations, scale=5000)
3. Renders interactive vis-network HTML with zoom-adaptive labels, hover animations, search, detail panel

**Node type colors:** Entity `#10b981` · EntityType `#f59e0b` · DocumentChunk `#3b82f6` · TextDocument `#a855f7` · Event `#ef4444` · Timestamp `#f97316`

---

### `agent_server.py` — Web UI Agent (port 8080, local)

Full-stack web interface:
- **Chat panel** — SSE-streamed conversation with Groq LLaMA-4
- **Upload tab** — drag-and-drop files (PDF, CSV, Excel, images, text, JSON, Markdown) → text extraction → orchestrator `save`
- **Paste tab** — paste raw text with optional dataset hint
- All write paths route through `orchestrator.call("save", ...)` for auto-classification
- Falls back to mcp-wrapper direct call if orchestrator unavailable

**File extraction:** `pypdf` (PDF), stdlib `csv` (CSV), `openpyxl` (Excel), plain decode (text/JSON/MD), filename note (images)

---


## 6. Orchestrator Agent

The orchestrator package provides intelligent routing on top of the raw MCP gateway.

### `orchestrator/document_classifier.py`

Classifies document text into one of 7 typed datasets using a two-pass approach:

1. **Keyword pass** — score each type by keyword hits. If top score ≥ 2 → use it (confidence = min(hits/5, 0.9))
2. **LLM pass** — if keyword confidence < 0.6, ask LLM with JSON schema `{"type_label": "...", "confidence": 0.0}`
3. **Fallback** — `ds_reports`

| Label | Dataset | Keywords |
|-------|---------|---------|
| `policy` | `ds_policy` | policy, procedure, guideline, compliance |
| `regulation` | `ds_regulation` | law, regulation, GDPR, ISO, decree |
| `employees` | `ds_employees` | employee, HR, staff, salary, hiring |
| `market` | `ds_market` | revenue, sales, competitor, Q1/Q2/Q3/Q4 |
| `workflow` | `ds_workflow` | process, BPMN, workflow, step, task |
| `contracts` | `ds_contracts` | contract, supplier, vendor, SLA, agreement |
| `reports` | `ds_reports` | report, analysis, KPI, dashboard, findings |

`ensure_dataset_exists(name)` creates the dataset in mcp-wrapper if it doesn't exist (idempotent).

### `orchestrator/query_analyzer.py`

Detects temporal references via regex (years 2020–2030, ISO dates, Q1–Q4, yesterday/today/last week) then classifies intent and best search type via LLM.

Returns `QueryIntent(intent, search_type, is_temporal, confidence)`.

Fallback on LLM failure: `search_type="GRAPH_COMPLETION"`.

### `orchestrator/version_manager.py`

Search-before-save: runs `CHUNKS_LEXICAL` search on first 200 chars. If existing content found, extracts highest `[VERSION:N]` tag and increments. Appends version metadata to text:

```
\n\n[VERSION:2][DATASET:ds_market][UPDATED:2026-05-14T10:00:00Z]
```

Returns `VersionResult(action, version_number, dataset_name, versioned_text)`.

### `orchestrator/memory_router.py`

Maps `QueryIntent` → single best `search_type`. Overrides to `TEMPORAL` for any query with `is_temporal=True`.

### `orchestrator/mcp_client.py`

Async httpx client to `mcp-wrapper:8002`. Lazy MCP initialize on first call, stores `mcp-session-id`. SSE response parsing. Simple circuit breaker (3 failures → open, 60s reset). Module-level singleton `wrapper`.

### `orchestrator/graphs/`

Four LangGraph `StateGraph` pipelines compiled once at import time:

**`save_graph`** — `classify` → (hint?) `override_hint` → `ensure_dataset` → `check_version` → `cognify`

**`search_graph`** — `analyze` → `route` → `search` → `format`

**`improve_graph`** — `improve` (all datasets or named one)

**`delete_graph`** — branch on target: `prune` / `delete_dataset` / `delete_item`

---

## 7. API Layer

### Memory Router — `/api/v1/memory/`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/record` | Store a fact → fast STM save + background cognify |
| `POST` | `/recall` | Retrieve context — 3-tier graceful degradation |
| `POST` | `/feedback` | Rate a recall result (score 1–5) |
| `POST` | `/trace` | Log an agent action step |
| `GET` | `/data` | List all datasets |
| `DELETE` | `/data` | Delete a data item by UUID |
| `DELETE` | `/datasets/{id}` | Delete all items in a dataset |
| `DELETE` | `/prune` | Wipe all stored knowledge |
| `GET` | `/cognify/status` | Check pipeline status |

### Workspace Router — `/api/v1/workspaces/`

Standard CRUD for tenant workspaces (UUID-keyed, status: ACTIVE / INACTIVE / ARCHIVED).

### Pydantic Schemas

| Schema | Used for |
|---|---|
| `RecordMemoryRequest` | `POST /record` body |
| `RecallMemoryRequest` | `POST /recall` body |
| `ContextResponseSchema` | Recall response — includes `is_degraded` flag |
| `FeedbackRequest` | Score + qa_id + user_id |
| `TraceRequest` | Function name, params, status, error |
| `WorkspaceCreate / Response / UpdatePartial` | Workspace CRUD |

---

## 8. Service Layer

### `memory_service.py`

**3-tier graceful degradation** for recall:

```
Tier 1: GRAPH_COMPLETION via Cognee MCP
    ↓ (if fails / empty)
Tier 2: CHUNKS search (vector similarity)
    ↓ (if fails / empty)
Tier 3: Fallback response with is_degraded=True
```

For `record_agent_memory()`:
- Immediately calls `save_interaction` (fast, ~2s) → acknowledges 202 Accepted
- Fires `cognify` in the background (~30s) for full graph processing

---

## 9. Data Layer

### `cognee_repository.py`

HTTP client speaking JSON-RPC 2.0 to Cognee MCP at `:8001`.

**Session flow:**
1. `_initialize_session()` → POST `initialize` → receive `mcp-session-id` header
2. All subsequent calls include that header
3. Responses are SSE — `_parse_response_payload()` handles `data: {...}` lines

### `workspace_repository.py`

SQLAlchemy async repository. All methods use `AsyncSession`.

### `workspace_model.py`

```python
class Workspace(Base):
    id:          UUID (PK)
    name:        String (unique)
    description: Text
    status:      Enum(ACTIVE, INACTIVE, ARCHIVED)
```

---

## 10. Configuration

### `configuration/settings.py`

Three Pydantic `BaseSettings` classes:

**`AppConfig`** — `APP_HOST`, `APP_PORT`, `APP_ENV`, `CORS_ORIGINS`

**`DatabaseSettings`** — PostgreSQL async URI, pool size

**`CogneeSettings`** — `MCP_SERVER_URL` (default: `http://mcp-wrapper:8002`), `MCP_TIMEOUT`

### `.env.example` — Required Variables

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

---

## 11. Scripts & Tooling

### `scripts/run_memify.py` — Temporal Enrichment

Runs inside the cognee container via docker exec. Reads `DocumentChunk` nodes from Neo4j, calls LLM to extract temporal events, writes `Event` nodes with timestamps back to Neo4j.

### `scripts/test_orchestrator.py` — Orchestrator Tests

Full orchestrator test suite:
- `--unit-only` flag runs unit tests without Docker
- Integration tests: save → classify → search round-trip
- Tests all 4 tool schemas

### `scripts/test_streaming.py`

SSE streaming smoke test verifying the full `initialize` → `tools/call` → SSE response chain.

### `scripts/postgres-init/01-create-cognee-db.sh`

Creates `cognee_db` on first Postgres container boot.

---

## 12. Knowledge Graph & Visualization

### `ontologies/ontology.ttl`

RDF/Turtle domain ontology for the financial/BPI domain. 15+ entity types: `AccountingCriterion` · `BusinessGroup` · `FinancialFlow` · `FundingProgramme` · `PublicEstablishment` …

### Neo4j Browser

```
http://localhost:7474
Login: neo4j / neo4j_pass
```

```cypher
-- All entities and relationships
MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 100

-- Temporal events (after memify)
MATCH (e:Event) RETURN e ORDER BY e.timestamp DESC LIMIT 20

-- Data by typed dataset
MATCH (n:TextDocument) RETURN n.dataset_name, count(n) ORDER BY count(n) DESC
```

---

## 13. Orchestrator Tools Reference

Connect to `http://localhost:8003` via MCP JSON-RPC 2.0. Four tools replace all 18:

| Tool | Latency | Description |
|---|---|---|
| `save` | ~35 s | Store any text or document. Auto-classifies into typed dataset, detects temporal content, versions duplicates. Optional `hint` overrides classifier. |
| `search` | ~7 s | Query memory. Auto-detects temporal queries, picks best search strategy. Optional `hint` for type override. |
| `improve` | ~60 s | Run quality improvement loop. `dataset_name="all"` (default) or a specific dataset. |
| `delete` | ~3 s | Delete a dataset by name, a data item by UUID, or `"all"` to wipe everything. |

### Typed datasets

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

---

## 14. MCP Gateway Tools Reference

All tools called via `POST /mcp` on the wrapper (`:8002`) using JSON-RPC 2.0.

### Write Tools

| Tool | Latency | Description |
|---|---|---|
| `cognify` | ~30 s | Deep-process text. Pass `dataset_name` to route to a typed dataset (uses docker exec to call `cognee.add() + cognee.cognify()` directly). |
| `save_interaction` | ~2 s | Fast-append a short note to Short-Term Memory. |
| `memify` | ~120 s | Extract temporal Event nodes from stored chunks. |
| `improve` | ~60 s | Feedback weights → graph persistence → trace sync → session index. |

### Read Tools

| Tool | Latency | Description |
|---|---|---|
| `search` | ~5 s | 14 search strategies (see below). Accepts `dataset_name` to scope. |
| `list_data` | ~2 s | List all datasets and data items. |
| `cognify_status` | ~1 s | Pipeline idle/running check. |
| `visualize_graph` | ~15 s | Render Neo4j graph as interactive HTML. |

### Delete Tools

| Tool | Latency | Description |
|---|---|---|
| `delete` | ~3 s | Delete item by UUID or dataset by name (soft/hard). |
| `prune` | ~5 s | Wipe ALL knowledge — irreversible. |

### Dataset Management

| Tool | Description |
|---|---|
| `create_dataset` | Register a dataset name for the session. |
| `switch_dataset` | Switch active dataset for subsequent calls. |
| `delete_dataset` | Remove a dataset and all its data. |

### 14 Search Types

| Type | Strategy |
|---|---|
| `GRAPH_COMPLETION` | Graph traversal + LLM completion |
| `GRAPH_COMPLETION_COT` | Chain-of-thought graph completion |
| `GRAPH_COMPLETION_CONTEXT_EXTENSION` | Extended context window |
| `GRAPH_COMPLETION_DECOMPOSITION` | Query decomposition |
| `GRAPH_SUMMARY_COMPLETION` | Summary node completion |
| `TEMPORAL` | Time-ordered event retrieval |
| `TRIPLET_COMPLETION` | Subject-predicate-object triples |
| `CYPHER` | Direct Cypher query generation |
| `NATURAL_LANGUAGE` | NL to graph pattern |
| `RAG_COMPLETION` | Vector similarity + LLM |
| `SUMMARIES` | Return summary nodes |
| `CHUNKS` | Return raw document chunks |
| `CHUNKS_LEXICAL` | BM25-style keyword search |
| `FEELING_LUCKY` | Best-guess auto-select |

---

## 15. REST API Reference

### Graph Endpoints (`/graph/`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/graph/stats` | Neo4j entity and relation counts |
| `GET` | `/graph/data` | Raw JSON (nodes + edges) |
| `POST` | `/graph/build` | Trigger HTML graph generation |
| `GET` | `/graph` | Serve the rendered graph viewer |

### Ontology Endpoints (`/ontology/`)

| Method | Path | Description |
|---|---|---|
| `GET` | `/ontology/info` | Current ontology file path and status |
| `POST` | `/ontology/upload` | Upload `.ttl` / `.owl` file (multipart) |
| `POST` | `/ontology/commit` | Commit raw TTL string — hot-reloads cognee-mcp |

### Proxy Endpoints

| Method | Path | Proxied to |
|---|---|---|
| `POST` | `/add` | `mcp-wrapper cognify` |
| `POST` | `/search` | `mcp-wrapper search` |

### Utility

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/api/docs` | Swagger UI |

---

## 16. LLM Backends

### Azure OpenAI (cloud)
```env
LLM_PROVIDER=openai
LLM_MODEL=openai/gpt-4o-mini
LLM_ENDPOINT=https://<resource>.services.ai.azure.com/models
LLM_API_KEY=<key>
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
```

### OpenAI (cloud)
```env
LLM_ENDPOINT=https://api.openai.com/v1
```

### Ollama (local, offline)
```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:latest
LLM_ENDPOINT=http://host.docker.internal:11434/v1
LLM_API_KEY=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_ENDPOINT=http://host.docker.internal:11434/api/embeddings
EMBEDDING_DIMENSIONS=768
```

> Switching embedding dimensions requires wiping the vector store:
> `docker volume rm cognee-layer_cognee_data cognee-layer_cognee_storage`

---

## 17. Data Flow — End-to-End

### Ingestion via Orchestrator (`save`)

```
Agent text / file upload
    → POST /mcp  {name: "save", data: "...", hint: ""}
    → orchestrator/graphs/save_graph.py:
        ├─ classify_node    — keyword + LLM → ds_market / ds_policy / …
        ├─ ensure_dataset   — create_dataset in mcp-wrapper if new
        ├─ check_version    — CHUNKS_LEXICAL search → version tag appended
        └─ cognify_node     — calls mcp-wrapper cognify {dataset_name: "ds_market"}
    → mcp_wrapper.py  dispatch("cognify"):
        ├─ detects dataset_name != "main_dataset"
        └─ docker exec cognee-mcp python3 /tmp/cognify_dataset_run.py
            ├─ cognee.add(data, dataset_name="ds_market")
            └─ cognee.cognify(datasets=["ds_market"])
    → Neo4j / LanceDB / Postgres  (scoped to ds_market)
```

### Retrieval via Orchestrator (`search`)

```
Agent query
    → POST /mcp  {name: "search", query: "Q3 2024 revenue?"}
    → orchestrator/graphs/search_graph.py:
        ├─ analyze_node  — temporal regex + LLM → QueryIntent(search_type="TEMPORAL")
        ├─ route_node    — memory_router → RouteResult(search_type="TEMPORAL")
        └─ search_node   — mcp-wrapper search {search_type: "TEMPORAL"}
    → cognee-mcp:
        ├─ Neo4j Event nodes (temporal)
        ├─ LanceDB vector similarity
        └─ LLM completion → SSE response
```

### Temporal Enrichment (`memify`)

```
POST /mcp {name: "memify"}
    → mcp_wrapper.py  docker exec cognee-mcp python scripts/run_memify.py
    → run_memify.py:
        ├─ MATCH (c:DocumentChunk) in Neo4j
        ├─ LLM extracts Event nodes with timestamps
        └─ MERGE (Event) -[HAS_TIMESTAMP]-> (Timestamp) in Neo4j
```

### File Upload (agent_server.py)

```
Browser drag-and-drop / file picker
    → POST /api/upload  (multipart, UploadFile + hint)
    → agent_server.py extract_text():
        ├─ .pdf  → pypdf PdfReader → page text concat
        ├─ .csv  → csv.reader → row strings
        ├─ .xlsx → openpyxl → cell values
        ├─ .png/.jpg → filename + note
        └─ .txt/.json/.md → plain decode
    → orchestrator.call("save", {data, hint})
    → save_graph pipeline (see above)
```

### Feedback Loop (`improve`)

```
Agent rates recall: submit_feedback {qa_id, score: 5}
    → Stored in FS session cache (/app/cognee_data/cache)

Agent triggers: improve {}
    → Stage 1: Apply feedback weights to graph nodes
    → Stage 2: Persist session Q&A pairs as graph episodes
    → Stage 3: Sync agent traces to graph
    → Stage 4: Rebuild session index for future retrieval
```

---

## 18. Tech Stack Summary

| Category | Technology |
|---|---|
| **Web framework** | FastAPI 0.115 + Uvicorn 0.30 |
| **Protocol** | Model Context Protocol (MCP) — JSON-RPC 2.0 over HTTP + SSE |
| **Agent pipelines** | LangGraph (StateGraph) |
| **Knowledge graph** | Neo4j 5 with APOC plugin |
| **Relational DB** | PostgreSQL 15 + asyncpg 0.29 + SQLAlchemy 2.0 |
| **Vector store** | LanceDB (file-backed, embedded in cognee container) |
| **LLM routing** | LiteLLM (Azure OpenAI / OpenAI / Ollama) |
| **Agent SDK** | Groq (LLaMA-4-Scout), LangChain |
| **Graph layout** | NetworkX spring layout (pre-computed server-side) |
| **Graph visualization** | vis-network 9.1.x |
| **Ontology** | RDF/Turtle (OWL-compatible) — domain: BPI finance |
| **Validation** | Pydantic v2 |
| **Async HTTP** | httpx |
| **PDF parsing** | pypdf |
| **Excel parsing** | openpyxl |
| **Containerization** | Docker Compose v2 |
| **Testing** | pytest + httpx |
| **Logging** | Structured JSON (dictConfig + JsonFormatter) |
