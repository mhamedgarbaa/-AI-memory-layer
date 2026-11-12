# Container Architecture

Detailed breakdown of all 5 Docker services defined in `docker-compose.yml`.

---

## 1. `db` — PostgreSQL 15 (`workspace_db`)

**Role:** Shared relational store for both the FastAPI app and Cognee.

**Startup:** On first boot, `scripts/postgres-init/01-create-cognee-db.sh` runs inside `docker-entrypoint-initdb.d` and provisions a second database (`cognee_db`) alongside the app database (`workspace_db`).

**Two logical databases:**

| Database | Owner | Purpose |
|---|---|---|
| `workspace_db` | `workspace_user` | FastAPI app: workspaces, sessions, traces, agent state |
| `cognee_db` | `workspace_user` | Cognee internals: dataset metadata, pipeline state, chunk registry |

**How connected:**
- `web` → `workspace_db` via asyncpg (`POSTGRES_URI`)
- `cognee-mcp` → `cognee_db` via `DB_HOST=db`, `DB_NAME=cognee_db`

**Volume:** `pgdata` (named, local) — survives container restarts.

**Port exposed:** `5432` (host-accessible for local psql/migrations).

---

## 2. `neo4j` — Neo4j 5 (`workspace_neo4j`)

**Role:** Primary graph database. Every entity, relationship, summary, and Event node extracted by Cognee lives here.

**What's stored:**

| Node type | Created by | Purpose |
|---|---|---|
| `__Entity__`, `__Community__` | `cognify` pipeline | Extracted entities and clusters |
| `__Chunk__` / `DocumentChunk` | `cognify` pipeline | Raw text segments with embeddings |
| `DocumentSummary` | `cognify` pipeline | Per-dataset summaries |
| `Event` | `memify` | Temporal events with timestamps |
| Graphiti episode nodes | `graphiti_add` tool | Time-ordered fact evolution episodes |

**Plugin:** APOC loaded automatically via `NEO4J_PLUGINS: '["apoc"]'` — used by Cognee's graph traversal queries.

**Memory:** 2 GB initial heap / 4 GB max.

**Ports exposed:**
- `7474` — Browser UI (HTTP)
- `7687` — Bolt protocol (used by all other containers)

**Volumes:** `neo4j_data`, `neo4j_logs`, `neo4j_plugins`.

**Who connects:**
- `cognee-mcp` → via `GRAPH_DATABASE_URL=bolt://neo4j:7687`
- `mcp-wrapper` → via `GRAPH_DATABASE_URL=bolt://neo4j:7687` (Graphiti + graph_builder)
- `web` → direct Neo4j async driver in `app.py` for stats and `/graph/data`

---

## 3. `cognee-mcp` — Cognee MCP Server (`cognee_mcp_server`)

**Image:** Custom — built from `Dockerfile.cognee` on top of `cognee/cognee-mcp:main`, with `graphiti-core` pip-installed and two patches applied:
- `patches/sitecustomize.py` — tiktoken aliases + Azure embedding fix at Python startup
- `patches/fix_litellm_engine.py` — patches `LiteLLMEmbeddingEngine` to bypass broken Azure header path

**Role:** The core knowledge graph engine. Runs the full Cognee pipeline (chunk → embed → extract entities → build graph → summarize). Speaks JSON-RPC over HTTP (MCP protocol).

**Storage layout:**

| Path (inside container) | Volume | Contents |
|---|---|---|
| `/app/cognee_storage/data` | `cognee_storage` | Raw document files (pre-cognify input) |
| `/app/cognee_data/system` | `cognee_data` | Cognee system state (pipeline locks, metadata) |
| `/app/cognee_data/cache` | `cognee_data` | LanceDB vector files + session FS cache |
| `/ontologies/ontology.ttl` | `./ontologies` (bind, ro) | RDF/OWL ontology for entity typing |
| `/data/bpi` | `./data/data_bpi` (bind, ro) | BPI process documents |
| `/graph` | `graph_output` | Shared graph HTML output |

**LLM stack:**
- **LLM:** configurable via `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY` (defaults: OpenAI `gpt-5-nano`)
- **Embeddings:** configurable via `EMBEDDING_PROVIDER` (defaults: Ollama `nomic-embed-text` at `host.docker.internal:11434`)
- **Vector store:** LanceDB (file-backed inside `cognee_data`)
- **Graph store:** Neo4j at `bolt://neo4j:7687`
- **Relational store:** Postgres `cognee_db`

**Key flags:**

| Flag | Value | Why |
|---|---|---|
| `ENABLE_BACKEND_ACCESS_CONTROL` | `False` | Neo4j backend doesn't support Cognee's ACL handler |
| `ACCEPT_LOCAL_FILE_PATH` | `True` | Cognee writes documents to local files during cognify, then reads them back |
| `ONTOLOGY_RESOLVER` | `rdflib` | Entity type matching uses RDF lookup |
| `MATCHING_STRATEGY` | `fuzzy` | Fuzzy match against ontology classes |
| `COGNEE_LOG_SEARCH_HISTORY` | `true` | All searches are logged |
| `LLM_RATE_LIMIT_REQUESTS` | `60` | 60 LLM req/min |
| `EMBEDDING_RATE_LIMIT_REQUESTS` | `60` | 60 embedding req/min |

**Port:** `8001` (host) → `8000` (container). Not meant for direct external use — accessed via `mcp-wrapper`.

**extra_hosts:** `host.docker.internal:host-gateway` — allows reaching Ollama on the Docker host from Linux.

---

## 4. `mcp-wrapper` — Unified Tool Gateway (`mcp_wrapper`)

**Image:** Built from `Dockerfile.wrapper` — Python 3.11-slim, installs `requirements-wrapper.txt`, copies `mcp_wrapper.py`, `graph_builder.py`, and `configuration/`.

**Role:** The single MCP endpoint for all external agents. Proxies base tools to `cognee-mcp` and adds custom tools implemented locally. Every AI agent (LangChain, Claude, etc.) connects here — never directly to `cognee-mcp`.

**Internal architecture:**

```
External agent  ──►  POST :8002/mcp  (JSON-RPC 2.0)
                          │
                ┌─────────┴──────────┐
           base tools           custom tools
         (forwarded via         (local dispatch)
        CogneeMCPProxy)
                │                    │
         cognee-mcp:8000          local logic:
         (JSON-RPC + SSE)          memify subprocess
                                   graphiti_add/search
                                   graph visualization
                                   session management
                                   improve / persist
```

**CogneeMCPProxy:**
- Maintains a persistent HTTP connection to `cognee-mcp:8000`
- Circuit breaker: opens after 3 consecutive failures, resets after 60s
- Retry with exponential backoff (max 4 attempts, 5s–60s) for 429 rate-limit responses
- Reconnect: up to 10 attempts with 5s delay

**Session management (`_sessions` dict):**
- Sessions keyed by `session_id` (auto-generated UUID if not supplied)
- TTL: 3600s default (`MCP_WRAPPER_SESSION_TTL`)
- Stores: user, dataset, `last_active`, session-local Q&A cache, trace log, feedback scores

**Full tool catalogue (18 tools):**

| Tool | Handled by | What it does |
|---|---|---|
| `cognify` | proxy → cognee-mcp | Deep-process text into graph; passes `temporal_cognify` flag |
| `save_interaction` | proxy → cognee-mcp (routed to cognify) | Fast text ingest (Cognee's native save_interaction has a file-path bug — routed to cognify) |
| `search` | proxy → cognee-mcp | 14 search strategies (GRAPH_COMPLETION, TEMPORAL, CYPHER, etc.) |
| `list_data` | proxy → cognee-mcp | List all datasets |
| `cognify_status` | proxy → cognee-mcp | Check pipeline idle/running state |
| `prune` | proxy → cognee-mcp | Wipe all knowledge |
| `forget_memory` | proxy → cognee-mcp | Delete dataset or all user memory |
| `delete` | proxy → cognee-mcp | Soft/hard delete by data UUID |
| `memify` | subprocess via Docker socket | Extract `Event` nodes for TEMPORAL search; `docker exec cognee_mcp_server python3 scripts/run_memify.py` |
| `improve` | local | 4-stage: apply feedback weights → persist Q&A → persist traces → sync to session cache |
| `persist_sessions` | local → cognify | Legacy: store raw transcript as graph node |
| `improve_answer` | local | Re-query with COT, store corrected answer |
| `submit_feedback` | local | Score a recall result (1–5), stored in session for `improve` |
| `record_trace` | local | Log a tool-call step to session trace |
| `create_dataset` | proxy → cognee-mcp | Create a new named dataset |
| `switch_dataset` | local (session) | Change active dataset for current session |
| `delete_dataset` | proxy → cognee-mcp | Delete a named dataset |
| `graphiti_add` / `graphiti_search` | local subprocess | Add/search temporal episodes via Graphiti directly to Neo4j |
| `visualize_graph` | local (`graph_builder.py`) | Fetch Neo4j → NetworkX layout → write HTML to `/graph` volume |

**Docker socket:** Mounted at `/var/run/docker.sock:ro` — `memify` uses it to `docker exec` into `cognee_mcp_server` and run `scripts/run_memify.py` directly inside the Cognee Python environment.

**Scripts mount:** `./scripts:/scripts:ro` — passed into the container so they can be `exec`'d by the memify dispatcher.

**Port:** `8002`.

---

## 5. `web` — FastAPI Agent Service (`workspace_api`)

**Image:** Built from `Dockerfile` — multi-stage Python 3.11-slim build; runs as non-root `appuser`; full `./` bind-mounted with `--reload` in dev.

**Role:** User-facing API and web UI layer. Orchestrates the agent, proxies tool calls to `mcp-wrapper`, reads graph state directly from Neo4j, and manages workspace/session data in Postgres.

**Routes:**

| Endpoint | File | Purpose |
|---|---|---|
| `GET /health` | `app.py` | Liveness + Neo4j stats (entities, relations, memified nodes) |
| `GET /graph/stats` | `app.py` | Same stats, JSON only |
| `GET /graph/data` | `app.py` | Live Neo4j nodes+edges as JSON (up to 5000 nodes) |
| `POST /graph/build` | `app.py` | Background: fetch Neo4j → layout → write HTML to `/graph` volume |
| `GET /graph` | `app.py` | Serve graph viewer HTML |
| `POST /add` | `app.py` | UI proxy → `mcp-wrapper` `cognify` |
| `POST /search` | `app.py` | UI proxy → `mcp-wrapper` `search` |
| `GET /ontology/info` | `app.py` | Ontology file metadata |
| `POST /ontology/upload` | `app.py` | Upload TTL/OWL → write to shared volume → trigger cognee-mcp restart |
| `POST /ontology/commit` | `app.py` | Commit generated TTL → shared volume → trigger cognee-mcp restart |
| `POST /api/v1/memory/record` | `api/routers/v1/memory_router.py` | Record fact → STM + async LTM cognify |
| `POST /api/v1/memory/recall` | `api/routers/v1/memory_router.py` | Retrieve context with graceful degradation |
| `GET /api/v1/memory/data` | `api/routers/v1/memory_router.py` | List datasets |
| `DELETE /api/v1/memory/data` | `api/routers/v1/memory_router.py` | Delete data item by UUID |
| `DELETE /api/v1/memory/datasets/{id}` | `api/routers/v1/memory_router.py` | Delete all items in a dataset |
| `DELETE /api/v1/memory/prune` | `api/routers/v1/memory_router.py` | Wipe all knowledge |
| `POST /api/v1/memory/feedback` | `api/routers/v1/memory_router.py` | Submit recall quality score (1–5) |
| `POST /api/v1/memory/trace` | `api/routers/v1/memory_router.py` | Record agent trace step |
| `/api/v1/workspaces/*` | `api/routers/v1/workspace_router.py` | Workspace CRUD (Postgres) |

**Service layer:**
`MemoryService` (`service/memory_service.py`) → `CogneeRepository` (`data/repositories/cognee_repository.py`) → `mcp-wrapper` via JSON-RPC. `trigger_cognify` always sets `temporal_cognify=True`.

**Direct Neo4j access:** `app.py` uses `AsyncGraphDatabase` driver for read-only stats and graph export — bypasses `mcp-wrapper` for low-latency queries.

**Volumes:** `./:/app` (bind, full codebase — enables `--reload`) + `graph_output:/graph` (shared with `mcp-wrapper` for the graph HTML).

**Port:** `8000`.

---

## Network & Dependency Order

```
cognee_net (bridge)
│
├── db          ← starts first   (healthcheck: pg_isready)
├── neo4j       ← starts second  (healthcheck: wget :7474)
├── cognee-mcp  ← waits for db + neo4j healthy   (healthcheck: GET /health)
├── mcp-wrapper ← waits for cognee-mcp healthy   (healthcheck: GET /health)
└── web         ← waits for db + neo4j healthy + cognee-mcp started
```

---

## Shared Volumes

| Volume | Used by | Contents |
|---|---|---|
| `graph_output` | `web`, `mcp-wrapper` | Built graph HTML (`cognee_graph.html`) |
| `cognee_data` | `cognee-mcp` | LanceDB vector files, FS cache, system state |
| `cognee_storage` | `cognee-mcp` | Raw document files (pre-cognify input) |
| `pgdata` | `db` | Postgres data directory |
| `neo4j_data` | `neo4j` | Graph data |
| `neo4j_logs` | `neo4j` | Query and server logs |
| `neo4j_plugins` | `neo4j` | APOC and other plugins |
