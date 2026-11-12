import asyncio
import os
import time

from fastapi import FastAPI, Request
from fastapi.background import BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.middlewares.logging_middleware import LoggingMiddleware, RequestIDMiddleware
from api.routers.v1.workspace_router import router as workspaces_v1_router
from api.routers.v1.memory_router import router as memory_router
from api.routers.v1.ingest_router import router as ingest_router
from configuration.database import init_db
from configuration.logging_setup import logger
from configuration.settings import settings

_NEO4J_URI       = os.getenv("GRAPH_DATABASE_URL",      "bolt://neo4j:7687")
_NEO4J_USER      = os.getenv("GRAPH_DATABASE_USERNAME",  "neo4j")
_NEO4J_PASS      = os.getenv("GRAPH_DATABASE_PASSWORD",  "neo4j_pass")
_MCP_WRAPPER_URL = os.getenv("MCP_WRAPPER_URL",         "http://mcp-wrapper:8002")
_GRAPH_FILE      = os.getenv("GRAPH_OUTPUT_PATH",        "/graph/cognee_graph.html")


async def _neo4j_stats() -> dict:
    """Return entity / relation / memified-node counts from Neo4j."""
    try:
        from neo4j import AsyncGraphDatabase
        driver = AsyncGraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))
        async with driver.session() as s:
            e  = await (await s.run("MATCH (n) RETURN count(n) AS c")).single()
            r  = await (await s.run("MATCH ()-[r]->() RETURN count(r) AS c")).single()
            m  = await (await s.run(
                "MATCH (n) WHERE 'Event' IN labels(n) OR n.created_at IS NOT NULL "
                "RETURN count(n) AS c"
            )).single()
        await driver.close()
        return {
            "entities":      e["c"] if e else 0,
            "relations":     r["c"] if r else 0,
            "memified_nodes": m["c"] if m else 0,
        }
    except Exception as exc:
        logger.warning("Neo4j stats query failed: %s", exc)
        return {"entities": 0, "relations": 0, "memified_nodes": 0}


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.TITLE,
        description=settings.DESCRIPTION,
        version=settings.VERSION,
        docs_url=settings.DOCS_URL,
        redoc_url=settings.REDOC_URL,
        openapi_url=settings.OPENAPI_URL,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health")
    async def health_check():
        t0 = time.monotonic()
        stats = await _neo4j_stats()
        ping_ms = round((time.monotonic() - t0) * 1000)
        return {
            "status":  "connected",
            "pingMs":  ping_ms,
            "details": stats,          # {entities, relations, memified_nodes}
        }

    # ── Graph stats ───────────────────────────────────────────────────────────
    @app.get("/graph/stats")
    async def graph_stats():
        s = await _neo4j_stats()
        return {
            "entities":     s["entities"],
            "relations":    s["relations"],
            "memifiedNodes": s["memified_nodes"],
        }

    # ── Knowledge graph: live JSON data ──────────────────────────────────────
    @app.get("/graph/data")
    async def graph_data(limit: int = 5000):
        """Return nodes + edges from Neo4j as JSON for the graph viewer."""
        try:
            from neo4j import AsyncGraphDatabase
            driver = AsyncGraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASS))
            nodes: dict = {}
            edges: list = []
            async with driver.session() as s:
                res = await s.run(
                    "MATCH (n) RETURN elementId(n) AS id, labels(n) AS lbls, "
                    "properties(n) AS props LIMIT $lim",
                    lim=limit,
                )
                async for rec in res:
                    nid = str(rec["id"])
                    lbls = rec["lbls"] or []
                    props = {k: str(v)[:300] for k, v in (rec["props"] or {}).items()}
                    _internal = {"__Node__", "__Entity__", "__Community__", "__Chunk__"}
                    node_type = next((l for l in lbls if l not in _internal), lbls[0] if lbls else "Node")
                    name = (
                        props.get("name") or props.get("label") or
                        props.get("title") or node_type
                    )
                    nodes[nid] = {
                        "id": nid, "label": str(name)[:60],
                        "type": node_type,
                        "labels": [l for l in lbls if l not in _internal] or lbls,
                        "properties": props,
                    }
                node_ids = set(nodes)
                res = await s.run(
                    "MATCH (n)-[r]->(m) RETURN elementId(r) AS eid, elementId(n) AS src, "
                    "elementId(m) AS tgt, type(r) AS rel LIMIT $lim",
                    lim=limit * 3,
                )
                seen: set = set()
                async for rec in res:
                    src, tgt = str(rec["src"]), str(rec["tgt"])
                    if src in node_ids and tgt in node_ids:
                        eid = str(rec["eid"])
                        if eid not in seen:
                            seen.add(eid)
                            edges.append({
                                "id": eid, "source": src,
                                "target": tgt, "label": rec["rel"] or "",
                            })
            await driver.close()
            return {"nodes": list(nodes.values()), "edges": edges}
        except Exception as exc:
            logger.error("graph_data error: %s", exc)
            return JSONResponse({"error": str(exc), "nodes": [], "edges": []}, status_code=500)

    # ── Knowledge graph: build (fetch Neo4j → NetworkX layout → HTML) ────────
    @app.post("/graph/build")
    async def graph_build(background_tasks: BackgroundTasks, limit: int = 5000):
        """Trigger graph_builder: fetch Neo4j, compute layout, write HTML."""
        async def _run():
            try:
                from graph_builder import build as _build
                await _build(_GRAPH_FILE, limit)
                logger.info("graph_builder finished → %s", _GRAPH_FILE)
            except Exception as exc:
                logger.error("graph_builder error: %s", exc)
        background_tasks.add_task(_run)
        return {
            "status":   "building",
            "message":  "Graph is being built. Refresh /graph in a few seconds.",
            "graph_url": "http://localhost:8000/graph",
        }

    # ── Serve the graph viewer HTML ───────────────────────────────────────────
    @app.get("/graph", response_class=HTMLResponse)
    @app.get("/graph/cognee_graph.html", response_class=FileResponse)
    async def serve_graph():
        if not os.path.isfile(_GRAPH_FILE):
            return HTMLResponse(
                "<html><body style='background:#0f172a;color:#94a3b8;"
                "font-family:sans-serif;display:flex;align-items:center;"
                "justify-content:center;height:100vh;margin:0'>"
                "<div style='text-align:center'>"
                "<h2 style='color:#38bdf8'>No graph yet</h2>"
                "<p>Call the <code>visualize_graph</code> MCP tool first,<br>"
                "or POST to <code>/graph/build</code>.</p>"
                "</div></body></html>",
                status_code=404,
            )
        return FileResponse(
            _GRAPH_FILE,
            media_type="text/html",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    # ── Ontology helpers ──────────────────────────────────────────────────────
    import httpx as _httpx

    async def _reload_cognee_ontology() -> None:
        """Tell mcp-wrapper to restart cognee-mcp so it picks up the new ontology file."""
        try:
            async with _httpx.AsyncClient(base_url=_MCP_WRAPPER_URL, timeout=90) as c:
                resp = await c.post("/admin/reload-cognee")
                if resp.status_code != 200:
                    logger.warning("Cognee reload returned %s: %s", resp.status_code, resp.text[:200])
                else:
                    logger.info("Cognee MCP restarted — new ontology active")
        except Exception as exc:
            logger.warning("Failed to reload cognee-mcp after ontology update: %s", exc)

    # ── Ontology info ─────────────────────────────────────────────────────────
    @app.get("/ontology/info")
    async def ontology_info():
        import pathlib
        # The file is written here by /ontology/upload (web container path)
        local_path = pathlib.Path("/app/ontologies/ontology.ttl")
        exists = local_path.is_file()
        return {
            "path":   "/ontologies/ontology.ttl",  # path as seen by cognee-mcp
            "format": "text/turtle",
            "exists": exists,
            "size":   local_path.stat().st_size if exists else 0,
        }

    # ── Ontology upload (TTL / OWL only) ─────────────────────────────────────
    from fastapi import UploadFile, File as FastAPIFile
    ONTOLOGY_DIR = "/app/ontologies"
    ALLOWED_ONTOLOGY_EXTS = {".ttl", ".owl"}

    @app.post("/ontology/upload")
    async def upload_ontology(file: UploadFile = FastAPIFile(...)):
        import pathlib
        ext = pathlib.Path(file.filename).suffix.lower()
        if ext not in ALLOWED_ONTOLOGY_EXTS:
            return JSONResponse(
                {"error": f"Unsupported format '{ext}'. Use .ttl or .owl"},
                status_code=400,
            )
        raw = await file.read()
        dest = pathlib.Path(ONTOLOGY_DIR) / "ontology.ttl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        await _reload_cognee_ontology()
        return {
            "path":     f"/ontologies/{dest.name}",
            "filename": file.filename,
            "size":     len(raw),
        }

    # ── Ontology commit (generated or edited TTL) → shared volume ────────────
    @app.post("/ontology/commit")
    async def commit_ontology(request: Request):
        import pathlib
        body = await request.json()
        content = body.get("content", "").strip()
        if not content:
            return JSONResponse({"error": "No content provided"}, status_code=400)
        dest = pathlib.Path(ONTOLOGY_DIR) / "ontology.ttl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        await _reload_cognee_ontology()
        return {
            "path": "/ontologies/ontology.ttl",
            "size": len(content.encode("utf-8")),
            "injected": True,
        }

    # ── Cognify proxy (for UI /add) — calls mcp-wrapper via JSON-RPC ─────────
    @app.post("/add")
    async def add_data(request: Request):
        import httpx, json as _json
        body = await request.json()
        data = body.get("data") or _json.dumps(body)
        try:
            async with httpx.AsyncClient(base_url=_MCP_WRAPPER_URL, timeout=300) as c:
                resp = await c.post("/mcp", json={
                    "jsonrpc": "2.0", "id": 1,
                    "method": "tools/call",
                    "params": {"name": "cognify", "arguments": {"data": data}},
                })
            # MCP wrapper returns SSE: "data: {...}\n\n"
            result_text = ""
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    payload = _json.loads(line[5:].strip())
                    if "error" in payload:
                        return JSONResponse(
                            {"status": "error", "error": payload["error"].get("message", "MCP error")},
                            status_code=502,
                        )
                    content = payload.get("result", {}).get("content", [])
                    result_text = " ".join(
                        item.get("text", "") for item in content if item.get("type") == "text"
                    )
            return {"status": "success", "response": result_text[:500]}
        except Exception as exc:
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=502)

    # ── Search proxy (for UI /search) — calls mcp-wrapper via JSON-RPC ───────
    @app.post("/search")
    async def search_data(request: Request):
        import httpx, json as _json
        body = await request.json()
        query = body.get("query", "")
        try:
            async with httpx.AsyncClient(base_url=_MCP_WRAPPER_URL, timeout=120) as c:
                resp = await c.post("/mcp", json={
                    "jsonrpc": "2.0", "id": 1,
                    "method": "tools/call",
                    "params": {"name": "search", "arguments": {
                        "search_query": query,
                        "search_type":  body.get("search_type", "GRAPH_COMPLETION"),
                    }},
                })
            result_text = ""
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    payload = _json.loads(line[5:].strip())
                    content = payload.get("result", {}).get("content", [])
                    result_text = " ".join(
                        item.get("text", "") for item in content if item.get("type") == "text"
                    )
            return {"status": "success", "results": result_text[:2000]}
        except Exception as exc:
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=502)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(workspaces_v1_router, prefix=f"{settings.API_PREFIX}/v1")
    app.include_router(memory_router,        prefix=f"{settings.API_PREFIX}/v1")
    app.include_router(ingest_router,        prefix=f"{settings.API_PREFIX}/v1")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail":     exc.errors(),
                "body":       exc.body,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    return app


app = create_app()


@app.on_event("startup")
async def on_startup():
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database ready.")
