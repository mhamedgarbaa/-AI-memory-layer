"""
Bulk-ingest all PDFs from data/data_bpi/ into the Cognee knowledge graph.
Passes container file paths directly to cognify — Cognee MCP reads and extracts
the PDFs natively. wait_idle() between files prevents QueuePool exhaustion.

Requires: docker compose up -d cognee-mcp  (to mount /data/bpi)

Run:
    python bulk_ingest_bpi.py            # skip already-ingested PDFs
    python bulk_ingest_bpi.py --force    # re-ingest everything (clears registry)

NOTE: Cognee has NO built-in deduplication — calling cognify on the same file
twice will create duplicate nodes/edges in the graph. The registry below is the
guard that prevents that.
"""

import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

MCP_URL            = os.getenv("MCP_URL", "http://localhost:8002/mcp")
DATA_DIR           = Path(__file__).parent / "data" / "data_bpi"
CONTAINER_DATA_DIR = "/data/bpi"
MCP_HEADERS        = {"Accept": "application/json, text/event-stream"}

# Registry file — tracks which PDFs have been successfully ingested.
# Lives next to the data so it survives script moves.
REGISTRY_FILE = Path(__file__).parent / "data" / "bpi_ingest_registry.json"


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_registry(registry: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def parse_sse(body: str) -> dict:
    for line in body.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise RuntimeError(f"No SSE data: {body[:200]}")


async def mcp_call(http: httpx.AsyncClient, session_id: str, tool: str, args: dict, timeout: float = 600) -> str:
    resp = await http.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
              "params": {"name": tool, "arguments": args}},
        headers={**MCP_HEADERS, "mcp-session-id": session_id},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = parse_sse(resp.text)
    content = data.get("result", {}).get("content", [])
    return next((c["text"] for c in content if c.get("type") == "text"), "done")


async def wait_idle(
    http: httpx.AsyncClient,
    session_id: str,
    poll_interval: int = 10,
    max_wait: int = 900,   # 15 min ceiling — prevents infinite loops on stuck pipelines
) -> None:
    """Poll cognify_status until the pipeline reaches a terminal state."""
    waited = 0
    while True:
        status = await mcp_call(http, session_id, "cognify_status", {}, timeout=30)
        s = status.strip()
        # Terminal: idle, empty dict, or any finished/errored PipelineRunStatus
        if s in ("{}", "", "idle") or any(
            kw in s for kw in ("ERRORED", "COMPLETED", "FINISHED")
        ):
            return
        if waited >= max_wait:
            print(f"    wait_idle timeout after {max_wait}s — continuing anyway")
            return
        print(f"    pipeline busy ({s[:80]}) — waiting {poll_interval}s…")
        await asyncio.sleep(poll_interval)
        waited += poll_interval


async def main():
    force = "--force" in sys.argv

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DATA_DIR}")
        return

    # ── ingest registry ────────────────────────────────────────────────────────
    # IMPORTANT: Cognee has NO deduplication. Every cognify() call on the same
    # content creates new duplicate nodes in the graph. This registry prevents
    # re-running the bulk ingest from bloating the graph.
    registry = _load_registry()

    if force:
        print("--force flag set: re-ingesting ALL PDFs (ignoring registry)\n")
        to_ingest = pdfs
    else:
        to_ingest = [p for p in pdfs if p.name not in registry]
        skipped = len(pdfs) - len(to_ingest)
        if skipped:
            print(f"Skipping {skipped} already-ingested PDF(s).")
            print(f"  → Registry: {REGISTRY_FILE}")
            print(f"  → Use --force to re-ingest everything.\n")

    if not to_ingest:
        print("Nothing new to ingest. All PDFs are already in the graph.")
        return

    print(f"Ingesting {len(to_ingest)} new PDF(s)...\n")

    base_url = MCP_URL.rsplit("/mcp", 1)[0]
    async with httpx.AsyncClient(base_url=base_url) as http:
        resp = await http.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 0, "method": "initialize",
                  "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                             "clientInfo": {"name": "bulk-ingest", "version": "1.0"}}},
            headers=MCP_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        session_id = resp.headers.get("mcp-session-id")
        print(f"MCP session: {session_id}\n")

        for i, pdf in enumerate(to_ingest, 1):
            print(f"[{i}/{len(to_ingest)}] {pdf.name}")

            # Cognee parses the path as a URL internally — '#' is treated as a
            # fragment separator and truncates the filename. Create a sanitized
            # temporary copy for any file whose name contains URL-special chars.
            safe_name = re.sub(r"[#?&=+%]", "_", pdf.name)
            tmp_pdf = pdf.parent / safe_name if safe_name != pdf.name else None
            if tmp_pdf:
                shutil.copy2(pdf, tmp_pdf)

            container_path = f"{CONTAINER_DATA_DIR}/{safe_name}"

            print(f"  Cognifying ...", end=" ", flush=True)
            try:
                result = await mcp_call(http, session_id, "cognify", {"data": container_path})
                await wait_idle(http, session_id)
                print(f"OK — {result[:80]}")
                # ✅ Mark as ingested ONLY on success — prevents partial ingest
                # entries from being skipped on the next run.
                registry[pdf.name] = {
                    "ingested": True,
                    "container_path": container_path,
                }
                _save_registry(registry)
            except Exception as e:
                print(f"ERROR — {e}")
            finally:
                if tmp_pdf and tmp_pdf.exists():
                    tmp_pdf.unlink()
            print()

    print("All PDFs processed.")
    print(f"Registry saved → {REGISTRY_FILE}")
    print("View graph: http://localhost:8000/graph/cognee_graph.html")


if __name__ == "__main__":
    asyncio.run(main())
