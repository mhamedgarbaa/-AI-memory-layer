# IA Memory Layer — Comprehensive Presentation Generation Prompt

> Use this prompt with any LLM (Claude, GPT, Gemini, etc.) to generate complete presentations.

---

## Master Prompt Template

```
You are an expert presentation architect specializing in enterprise AI systems and knowledge graphs.

Generate a complete, production-ready presentation from the context and parameters below.

---

## CONTEXT: IA Memory Layer — Full Technical Overview

### What is IA Memory Layer?

**IA Memory Layer** is an enterprise-grade, graph-backed **temporal memory microservice** for AI agents.

It gives any AI agent (LangChain, AutoGen, CrewAI, custom agents) a persistent memory layer that:
- Stores raw text and **extracts entities, relationships, and temporal events** into a knowledge graph
- **Auto-classifies documents** into typed datasets (policies, regulations, contracts, market data, etc.)
- Answers semantic queries using **14 search strategies** (graph completion, vector RAG, Cypher, temporal, etc.)
- Tracks feedback, agent traces, and improves retrieval quality in a continuous loop
- Exposes everything through **MCP (Model Context Protocol)** — the standard AI agent tool protocol

The system is self-contained: spin it up with `docker compose up -d` and point any MCP-compatible agent at:
- **`http://localhost:8003`** — Orchestrator (4 smart tools, recommended)
- **`http://localhost:8002`** — Raw MCP Gateway (18 tools, direct control)

---

### System Architecture

Six Docker services boot in dependency order: `db` → `neo4j` → `cognee-mcp` → `mcp-wrapper` → `orchestrator` → `web`

**Service Stack:**
| Service | Port | Role |
|---------|------|------|
| **db** (PostgreSQL 15) | 5432 | Relational store — workspaces, metadata, Cognee datasets |
| **neo4j** | 7687/7474 | Knowledge graph — entities, relationships, temporal events |
| **cognee-mcp** | 8001 | Core AI memory engine — ECL pipeline (chunk → embed → extract → graph) |
| **mcp-wrapper** | 8002 | Gateway — 18 tools, session management, circuit breaker, docker-exec routing |
| **orchestrator** | 8003 | Orchestrator — 4 smart tools, LangGraph pipelines, auto-classification |
| **web** | 8000 | FastAPI REST API, graph viewer, ontology management |
| **agent_server** | 8080 | Web chat UI + file upload panel (local, not Docker) |

---

### Core Components

#### 1. Orchestrator (Port 8003)
- **4 smart tools**: `save`, `search`, `improve`, `delete`
- Auto-classifies documents into 7 typed datasets
- Detects temporal content (years, dates, Q1–Q4, relative time)
- Versions duplicates automatically
- Routes queries to optimal search strategies
- Built on LangGraph StateGraphs

**Document Classification:**
- `ds_policy` — Policies, procedures, compliance, guidelines
- `ds_regulation` — Laws, GDPR, ISO standards, legal directives
- `ds_employees` — HR, staff, salaries, hiring, performance
- `ds_market` — Revenue, sales, competitors, Q1–Q4 forecasts
- `ds_workflow` — BPMN, processes, task sequences
- `ds_contracts` — Supplier/vendor contracts, SLAs
- `ds_reports` — Analysis reports, KPI dashboards, audit findings

#### 2. MCP Wrapper (Port 8002)
- **18 raw tools** for direct control
- JSON-RPC 2.0 protocol with SSE streaming
- Session management (UUID4 session IDs, TTL eviction)
- Circuit breaker (opens after 3 failures, auto-recovers 60s)
- Docker-exec integration (dataset-scoped cognify, memify, graph visualization)

**Write Tools**: `cognify` (30s), `save_interaction` (2s), `memify` (120s), `improve` (60s)
**Read Tools**: `search` (5s), `list_data` (2s), `cognify_status` (1s), `visualize_graph` (15s)
**Delete Tools**: `delete` (3s), `prune` (5s)

#### 3. Cognee MCP (Port 8001)
- **Core memory engine** — ECL pipeline (Extract, Cognify, Load)
- Processes up to 30 file formats
- Extracts entities and relationships via LLM
- Persists to hybrid backend: Neo4j (graph), PostgreSQL (metadata), LanceDB (vectors)
- Every edge carries temporal metadata — tracks fact evolution over time

#### 4. Temporal Enrichment (`memify`)
- Runs inside cognee container via docker exec
- Reads DocumentChunk nodes from Neo4j
- LLM extracts temporal Event nodes with timestamps
- Writes Event nodes back to Neo4j
- Enables time-series reasoning and reconstruction of knowledge state at any past point

#### 5. Context Adapter (Built into Orchestrator)
- Authenticates caller and resolves permissions
- Selects appropriate retrieval strategies (runs in parallel)
- Compresses and ranks results within token budget
- Delivers focused, role-appropriate context to agent
- Prevents context window flooding with unfocused information

---

### 14 Search Strategies

| Strategy | Use Case |
|----------|----------|
| `GRAPH_COMPLETION` | Graph traversal + LLM completion (default) |
| `GRAPH_COMPLETION_COT` | Chain-of-thought graph completion |
| `GRAPH_COMPLETION_CONTEXT_EXTENSION` | Extended context window reasoning |
| `GRAPH_COMPLETION_DECOMPOSITION` | Query decomposition across subgraphs |
| `GRAPH_SUMMARY_COMPLETION` | Summary node completion |
| `TEMPORAL` | Time-ordered event retrieval (auto-selected for temporal queries) |
| `TRIPLET_COMPLETION` | Subject-predicate-object triple extraction |
| `CYPHER` | Direct Cypher query generation against Neo4j |
| `NATURAL_LANGUAGE` | Natural language to graph pattern conversion |
| `RAG_COMPLETION` | Vector similarity + LLM (hybrid retrieval) |
| `SUMMARIES` | Return summary nodes (community-level) |
| `CHUNKS` | Return raw document chunks |
| `CHUNKS_LEXICAL` | BM25-style keyword search |
| `FEELING_LUCKY` | Best-guess auto-select |

---

### API Endpoints

**Memory Router (`/api/v1/memory/`)**
- `POST /record` — Store a fact (fast STM save + background cognify)
- `POST /recall` — Retrieve context (3-tier graceful degradation)
- `POST /feedback` — Rate a recall result (score 1–5)
- `POST /trace` — Log an agent action step
- `GET /data` — List all datasets
- `DELETE /data` — Delete a data item by UUID
- `DELETE /datasets/{id}` — Delete all items in a dataset
- `DELETE /prune` — Wipe all stored knowledge
- `GET /cognify/status` — Check pipeline status

**Graph Endpoints (`/graph/`)**
- `GET /graph/stats` — Neo4j entity and relation counts
- `GET /graph/data` — Raw JSON (nodes + edges)
- `POST /graph/build` — Trigger HTML graph generation
- `GET /graph` — Serve the rendered graph viewer

**Ontology Endpoints (`/ontology/`)**
- `GET /ontology/info` — Current ontology file path and status
- `POST /ontology/upload` — Upload `.ttl` / `.owl` file
- `POST /ontology/commit` — Commit raw TTL string (hot-reloads cognee)

---

### Data Flow: End-to-End

#### Ingestion Pipeline (Save)
```
Agent text / file upload
    → POST /mcp {name: "save", data: "...", hint: ""}
    → Orchestrator save_graph:
        1. classify_node — keyword + LLM → ds_market / ds_policy / etc.
        2. ensure_dataset — create_dataset in mcp-wrapper if new
        3. check_version — CHUNKS_LEXICAL search → version tag appended
        4. cognify_node — calls mcp-wrapper cognify {dataset_name}
    → MCP Wrapper dispatch("cognify"):
        • detects dataset_name
        • docker exec cognee-mcp to run cognee.add() + cognee.cognify()
    → Neo4j / LanceDB / PostgreSQL (scoped to dataset)
```

#### Retrieval Pipeline (Search)
```
Agent query: "Q3 2024 revenue?"
    → POST /mcp {name: "search", query: "..."}
    → Orchestrator search_graph:
        1. analyze_node — temporal regex + LLM → QueryIntent
        2. route_node — memory_router → optimal search_type
        3. search_node — mcp-wrapper search
    → Cognee MCP:
        • Neo4j Event nodes (if temporal)
        • LanceDB vector similarity (hybrid)
        • LLM completion
    → SSE response with results
```

#### Temporal Enrichment (Memify)
```
POST /mcp {name: "memify"}
    → MCP Wrapper docker exec
    → run_memify.py:
        1. MATCH (c:DocumentChunk) in Neo4j
        2. LLM extracts Event nodes with timestamps
        3. MERGE (Event) -[HAS_TIMESTAMP]-> (Timestamp) in Neo4j
```

#### File Upload
```
Browser drag-and-drop / file picker
    → POST /api/upload (multipart)
    → Extract text (PDF→pypdf, CSV→csv reader, XLSX→openpyxl, etc.)
    → orchestrator.call("save", {data, hint})
    → save_graph pipeline (as above)
```

---

### Key Design Decisions

- **`temporal_cognify=True`** is hardcoded — every LTM write is temporal
- **Circuit breaker** in mcp-wrapper: opens after 3 failures, resets after 60s
- **3-tier graceful degradation** in recall: GRAPH_COMPLETION → CHUNKS → fallback
- **Ontology layer** preprocesses documents, prevents entity ambiguity
- **MCP protocol** for standard tool interface (LangChain, AutoGen, CrewAI compatible)
- **Docker-based deployment** — 6 services boot in order, all via compose
- **Hybrid storage**: Neo4j (graph), PostgreSQL (metadata), LanceDB (vectors, file-backed)
- **LLM backends**: OpenAI, Azure OpenAI, or local Ollama

---

### Tech Stack Summary

| Category | Technology |
|----------|-----------|
| **Web framework** | FastAPI 0.115 + Uvicorn 0.30 |
| **Protocol** | MCP (JSON-RPC 2.0) over HTTP + SSE |
| **Agent pipelines** | LangGraph (StateGraph) |
| **Knowledge graph** | Neo4j 5 with APOC plugin |
| **Relational DB** | PostgreSQL 15 + asyncpg 0.29 + SQLAlchemy 2.0 |
| **Vector store** | LanceDB (file-backed, embedded) |
| **LLM routing** | LiteLLM (Azure / OpenAI / Ollama) |
| **Graph visualization** | vis-network 9.1.x + NetworkX spring layout |
| **Ontology** | RDF/Turtle (OWL-compatible) |
| **Validation** | Pydantic v2 |
| **Async HTTP** | httpx |
| **Containerization** | Docker Compose v2 |
| **Testing** | pytest + httpx |

---

## YOUR PRESENTATION PARAMETERS

Fill in the fields below:

### 1. Presentation Scope
- **title**: (e.g., "IA Memory Layer: Enterprise Temporal Memory for AI Agents")
- **audience**: (e.g., C-suite, engineering, product, academics, students)
- **duration_minutes**: (total time in minutes: 7, 15, 25, 40)
- **depth**: (short | standard | deep)
- **tone**: (formal | conversational | persuasive | technical)

### 2. Presentation Focus
- **key_messages**: (list 3–5 one-sentence bullets; e.g., ["LLMs hallucinate without enterprise memory", "Cognee unifies temporal + community memory", "14 search strategies adapt to any query"])
- **must_include**: (list as: diagram, demo_steps, metrics, ontology, temporal-example, data-flow, api-reference, cost-analysis, etc.)
- **max_slides**: (optional; default 12 for standard)

### 3. Output Preferences
- **format**: (markdown | html | pptx-outline | speaker-script | all)
- **include_handout**: (yes | no)
- **include_qa_cheat_sheet**: (yes | no)
- **include_live_demo_steps**: (yes | no)

---

## REQUIRED OUTPUT FORMAT

### 1. Slide Deck (Ordered)

For each slide, provide:

**Slide N: [Title]**
- **Slide Type**: {title, agenda, problem, solution, architecture, diagram, code, demo, data, timeline, conclusion, appendix}
- **Content**: 3–6 bullet points (concise, 1 sentence each)
- **Speaker Notes**: 1 paragraph (40–120 words; what the presenter says)
- **Time**: suggested duration in seconds
- **Visual**: short instruction (e.g., "Diagram: 6-service Docker stack" or "Screenshot: Neo4j graph with colored nodes")
- **Key Talking Points**: 2–3 advanced details for Q&A preparation

---

### 2. Exportable Artifacts

#### A. Markdown Slide Outline
(One heading per slide; bullet points; ready for export to PowerPoint, Google Slides, or PDF)

#### B. Full Speaker Script
(Complete monologue; ~duration length; paragraph format; natural rhythm)

#### C. 1-Sentence TL;DR
(Executive summary; one sentence; capture the essence)

#### D. 3 Live Demo Steps (if applicable)
(Actionable commands or UI steps; ~2–5 minutes; runnable)

#### E. Handout (1-page summary, if requested)
(Compressed deck outline; key diagrams as ASCII or reference; takeaways)

#### F. Q&A Cheat Sheet (if requested)
(Likely questions + concise answers; 5–8 pairs)

---

### 3. Constraints

- **Total slide time** ≈ requested duration (allow ±10%)
- **Slide count** ≤ max_slides (or suggest if exceeding)
- **Tone** matches audience and depth
- **Context grounding** — cite specific file names or section numbers when pulling from the context provided
- **Visuals** — describe in language (ASCII diagrams acceptable; reference assets in repo if known)
- **Token budget** — if provided, keep entire output under limit

---

## EXAMPLE INVOCATION (Fill & Send)

```json
{
  "title": "IA Memory Layer: Temporal Memory for Enterprise AI",
  "audience": "Engineering + Product (mixed technical/non-technical)",
  "duration_minutes": 18,
  "depth": "standard",
  "tone": "technical",
  "key_messages": [
    "AI agents need persistent, structured memory to avoid hallucinations",
    "IA Memory Layer unifies temporal tracking + community reasoning in one system",
    "Auto-classification + ontology grounding ensures semantic consistency",
    "MCP protocol enables deployment with any agent framework"
  ],
  "must_include": ["architecture_diagram", "temporal_example", "demo_steps", "14_search_types"],
  "max_slides": 14,
  "format": "all",
  "include_handout": true,
  "include_qa_cheat_sheet": true,
  "include_live_demo_steps": true
}
```

---

## END OF PROMPT

Send the completed JSON (or fill fields directly) to your LLM model.
```

---

## VARIANT PROMPTS

### Variant 1: Quick 5-Minute Lightning Talk
```
Adjust parameters:
- duration_minutes: 5
- max_slides: 5
- depth: short
- tone: conversational
- key_messages: (2–3 most critical)
- must_include: ["problem", "solution", "call_to_action"]
```

### Variant 2: Deep Technical Masterclass (45 min)
```
Adjust parameters:
- duration_minutes: 45
- max_slides: 30
- depth: deep
- tone: technical
- key_messages: (5–7, comprehensive)
- must_include: ["full_architecture", "all_14_search_types", "data_flow_diagram", "api_reference", "deployment_guide", "Neo4j_queries"]
- include_handout: true
- include_qa_cheat_sheet: true
```

### Variant 3: Executive Pitch (10 min)
```
Adjust parameters:
- duration_minutes: 10
- max_slides: 6
- depth: short
- tone: persuasive
- key_messages: (business impact focus)
- must_include: ["problem", "solution", "benefits", "cost_savings", "roi_metrics"]
- include_handout: false
```

---

## USAGE INSTRUCTIONS

### Step 1: Choose Your Variant
Pick from: **Quick**, **Standard**, **Deep**, or **Executive**.

### Step 2: Customize Parameters
Edit the JSON block under "EXAMPLE INVOCATION" with your specifics.

### Step 3: Provide Context (Optional)
- Paste this entire prompt to your LLM
- Include the JSON with your custom parameters
- Add: `"context": "Use only the IA Memory Layer technical overview provided above"`

### Step 4: Send to Model
Example:
```
Claude / ChatGPT / Gemini / etc.

[Paste the full prompt from this file]

[Paste your filled JSON parameters]

Generate the presentation following the output format specified.
```

### Step 5: Export
- **Markdown** → PowerPoint, Google Slides, or PDF via Markdown converter
- **HTML** → Save as `.html` for standalone viewing
- **Speaker script** → Copy into teleprompter or printed notes
- **Demo steps** → Save as runnable terminal commands

---

## QUICK START EXAMPLES

### Example A: "Standard 18-Minute Engineering Talk"
```json
{
  "title": "IA Memory Layer: Enterprise Temporal Memory",
  "audience": "engineering",
  "duration_minutes": 18,
  "depth": "standard",
  "tone": "technical",
  "key_messages": [
    "Cognee unifies temporal + community memory in a single ECL engine",
    "14 adaptive search strategies enable precise, context-aware retrieval",
    "Auto-classification into 7 datasets + ontology grounding ensures consistency",
    "MCP protocol makes it deployable with LangChain, AutoGen, CrewAI"
  ],
  "must_include": ["architecture_diagram", "temporal_example", "demo_steps"],
  "max_slides": 14,
  "format": "markdown",
  "include_handout": true
}
```

### Example B: "5-Minute Lightning Round"
```json
{
  "title": "IA Memory Layer: Why AI Agents Need Memory",
  "audience": "general",
  "duration_minutes": 5,
  "depth": "short",
  "tone": "conversational",
  "key_messages": [
    "LLMs hallucinate on proprietary data",
    "IA Memory Layer = structured, temporal, graph-backed memory",
    "Deploy in 60 seconds with docker compose"
  ],
  "must_include": ["problem", "solution", "demo"],
  "max_slides": 5,
  "format": "markdown"
}
```

### Example C: "Deep 40-Minute Technical Workshop"
```json
{
  "title": "Building Enterprise AI Memory Systems: IA Memory Layer Internals",
  "audience": "engineers + architects",
  "duration_minutes": 40,
  "depth": "deep",
  "tone": "technical",
  "key_messages": [
    "ECL pipeline: chunk → embed → extract → graph (per dataset)",
    "Temporal metadata on every edge enables fact versioning",
    "14 search strategies + community summarization = hybrid reasoning",
    "Ontology layer prevents entity ambiguity",
    "Context Adapter prevents token window flooding"
  ],
  "must_include": ["full_architecture", "ecl_pipeline", "14_search_types", "data_flows", "deployment", "neo4j_queries", "demo"],
  "max_slides": 28,
  "format": "all",
  "include_handout": true,
  "include_qa_cheat_sheet": true
}
```

---

End of presentation prompt.
