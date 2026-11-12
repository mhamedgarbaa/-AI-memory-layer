# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run the full stack
```bash
docker compose up -d
# Wait ~60s for cognee-mcp to become healthy
docker compose ps
```

### Rebuild a single service after code changes
```bash
docker compose build <service>   # web | mcp-wrapper | cognee-mcp
docker compose up -d <service>
```

### View logs
```bash
docker compose logs -f web
docker compose logs -f mcp-wrapper
docker compose logs -f cognee-mcp
```

### Run integration tests
```bash
pytest tests/test_stack_integration.py -v
# Against a remote host:
BASE_URL=http://<host>:8000 MCP_URL=http://<host>:8002 pytest tests/test_stack_integration.py -v
```

### Streaming smoke test
```bash
python scripts/test_streaming.py
```

### Wipe vector store (required when embedding model or dimensions change)
```bash
docker compose down
docker volume rm cognee-layer_cognee_data cognee-layer_cognee_storage
docker compose up -d
```

### Local dev (web service only, without Docker)
```bash
pip install -r requirements.txt
# Set DB_HOST=localhost, MCP_SERVER_URL=http://localhost:8001 in .env
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Architecture

### Five-service Docker stack

Services start in strict dependency order: `db` → `neo4j` → `cognee-mcp` → `mcp-wrapper` → `web`

| Service | Port | Container | Role |
|---------|------|-----------|------|
| `db` | 5432 | `workspace_db` | Postgres 15 — two databases: `workspace_db` (FastAPI app state) + `cognee_db` (Cognee internals) |
| `neo4j` | 7474/7687 | `workspace_neo4j` | Graph database for all entities, communities, chunks, Event nodes |
| `cognee-mcp` | 8001→8000 | `cognee_mcp_server` | Core knowledge graph engine — runs the full cognify pipeline (chunk → embed → extract → graph) |
| `mcp-wrapper` | 8002 | `mcp_wrapper` | Unified MCP gateway exposing 18 tools; proxies to cognee-mcp, adds session management, circuit breaker, docker-exec tools |
| `web` | 8000 | `workspace_api` | FastAPI REST API and UI; proxies to mcp-wrapper; reads Neo4j directly for stats/graph |

### Request routing

External agents and clients connect **only to `mcp-wrapper:8002`** via JSON-RPC 2.0. Never bypass to `cognee-mcp:8001` directly (causes host-header 421 errors and loses session management).

The `web` service at `:8000` provides the higher-level REST API and talks to `mcp-wrapper` for all memory operations, but reads Neo4j directly for graph stats and visualization.

### Internal code structure (`web` service)

```
app.py                          # FastAPI entry point, direct Neo4j routes, ontology endpoints
api/
  routers/v1/memory_router.py  # REST: /record, /recall, /feedback, /trace, /data, /prune
  routers/v1/workspace_router.py
  schemas/memory_schema.py
  dependencies.py              # Dependency injection wiring (httpx client, CogneeRepository)
service/
  memory_service.py            # Business logic: 3-tier recall with circuit-breaker fallback
  business_models.py           # ContextEnvelope model
data/
  repositories/cognee_repository.py  # MCP JSON-RPC client (session handshake, tool calls, response parsing)
  models/workspace_model.py
configuration/
  settings.py     # AppConfig, DatabaseSettings, CogneeSettings (pydantic-settings)
  database.py     # SQLAlchemy async engine (asyncpg)
  logging_setup.py
```

### MCP session protocol

Every tool call requires a prior `initialize` handshake to get a `mcp-session-id` header. `CogneeRepository` handles this automatically (lazy init on first call). All subsequent requests must include `mcp-session-id`. Responses are Server-Sent Events (`text/event-stream`) with `data: {...}` lines.

### Key design decisions

- **`temporal_cognify=True`** is hardcoded in `CogneeRepository.trigger_cognify()` — every LTM write is temporal. The `memify` tool additionally extracts `Event` nodes via `docker exec` into the cognee container.
- **Circuit breaker** in `mcp-wrapper` opens after 3 consecutive failures to `cognee-mcp`, resets after 60s. The `web` service's `memory_service.py` has its own 3-tier fallback: `GRAPH_COMPLETION` → `CHUNKS` → empty.
- **`ENABLE_BACKEND_ACCESS_CONTROL=False`** is required because the Neo4j provider doesn't support Cognee's ACL handler. Without it, `search` fails.
- **`ACCEPT_LOCAL_FILE_PATH=True`** is required — Cognee writes files to its own data directory during cognify and reads them back.
- **Ontology** (`.ttl` file) is bind-mounted read-only into `cognee-mcp`. Uploading via `/ontology/upload` or `/ontology/commit` on the web service writes to the shared `./ontologies/` directory and triggers an automatic `cognee-mcp` restart.
- **`memify`** runs via `docker exec cognee_mcp_server python3 scripts/run_memify.py` — requires the Docker socket mounted in `mcp-wrapper`.
- **Graph HTML** is written to the `graph_output` shared volume; both `web` and `mcp-wrapper` mount it at `/graph`.

### 14 search types

`GRAPH_COMPLETION` · `GRAPH_COMPLETION_COT` · `GRAPH_COMPLETION_CONTEXT_EXTENSION` · `GRAPH_COMPLETION_DECOMPOSITION` · `GRAPH_SUMMARY_COMPLETION` · `TEMPORAL` · `TRIPLET_COMPLETION` · `CYPHER` · `NATURAL_LANGUAGE` · `RAG_COMPLETION` · `SUMMARIES` · `CHUNKS` · `CHUNKS_LEXICAL` · `FEELING_LUCKY`

### Environment configuration

Minimum required `.env` variables:
```
LLM_PROVIDER=openai
LLM_MODEL=openai/gpt-4o-mini
LLM_API_KEY=<key>
LLM_ENDPOINT=https://api.openai.com/v1   # or Azure endpoint

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_API_KEY=<key>
EMBEDDING_ENDPOINT=https://api.openai.com/v1
EMBEDDING_DIMENSIONS=1536
```

For local Ollama, set `EMBEDDING_DIMENSIONS=768` and use `host.docker.internal:11434` endpoints.

### Active development branch

`feature/orchestrator-agent` — planned new container at `:8003` adding an Orchestrator Agent layer that exposes 4 simplified tools (`save`, `search`, `improve`, `delete`) and internally dispatches to all 18 mcp-wrapper tools with automatic document classification, temporal detection, and memory routing.
