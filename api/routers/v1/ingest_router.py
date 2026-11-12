"""
File / text ingestion endpoint for the web container.

Two paths:
  - dataset_name provided  → mcp-wrapper cognify directly
                             (add → cognify on the named dataset, background)
  - no dataset_name        → orchestrator save
                             (auto-classify → typed dataset, background)

Supported formats: PDF, CSV, XLSX, TXT, JSON, MD, PNG, JPG
"""
import csv
import io
import json
import os

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/ingest", tags=["INGEST"])

_ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8003")
_MCP_WRAPPER_URL  = os.getenv("MCP_WRAPPER_URL",  "http://mcp-wrapper:8002")

_SUPPORTED_EXT = {".pdf", ".csv", ".xlsx", ".xls", ".txt", ".json", ".md",
                  ".png", ".jpg", ".jpeg", ".gif", ".webp"}


# ── Text extractors ───────────────────────────────────────────────────────────

def _extract_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except ImportError:
        return "[PDF extraction requires pypdf]"


def _extract_csv(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return ""
    header = " | ".join(rows[0])
    body   = [" | ".join(r) for r in rows[1:200]]
    suffix = [f"[... {len(rows) - 201} more rows truncated]"] if len(rows) > 201 else []
    return "\n".join([header, "-" * len(header)] + body + suffix)


def _extract_xlsx(data: bytes) -> str:
    try:
        import openpyxl
        wb  = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        out = []
        for sheet in wb.worksheets:
            out.append(f"=== Sheet: {sheet.title} ===")
            for i, row in enumerate(sheet.iter_rows(values_only=True)):
                if i > 500:
                    out.append("[... truncated]")
                    break
                out.append(" | ".join("" if v is None else str(v) for v in row))
        return "\n".join(out)
    except ImportError:
        return "[XLSX extraction requires openpyxl]"


def extract_text(filename: str, content_type: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf") or "pdf" in content_type:
        return _extract_pdf(data)
    if name.endswith(".csv") or "csv" in content_type:
        return _extract_csv(data)
    if name.endswith((".xlsx", ".xls")) or "spreadsheet" in content_type or "excel" in content_type:
        return _extract_xlsx(data)
    if any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")) or "image" in content_type:
        return f"[Image file: {filename}]\nNote: image was uploaded; text content not extracted."
    try:
        return data.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


# ── MCP helpers ───────────────────────────────────────────────────────────────

def _parse_sse(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        candidate = line[5:].strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if "error" in payload:
            raise RuntimeError(payload["error"].get("message", str(payload["error"])))
        content = payload.get("result", {}).get("content", [])
        return " ".join(item.get("text", "") for item in content if item.get("type") == "text")
    return ""


async def _mcp_call(client: httpx.AsyncClient, base_url: str, tool: str,
                    args: dict, timeout: float = 30.0) -> str:
    """Single-shot MCP call: initialize → notifications/initialized → tools/call."""
    init = await client.post(
        f"{base_url}/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                         "clientInfo": {"name": "web-ingest", "version": "1.0"}}},
        headers={"Content-Type": "application/json"},
    )
    init.raise_for_status()
    sid = init.headers.get("mcp-session-id", "")
    h   = {"Content-Type": "application/json", "mcp-session-id": sid}

    await client.post(
        f"{base_url}/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        headers=h,
    )

    resp = await client.post(
        f"{base_url}/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": tool, "arguments": args}},
        headers=h,
        timeout=timeout,
    )
    resp.raise_for_status()
    return _parse_sse(resp.text)


# ── Upload endpoint ───────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(
    file:         UploadFile = File(...),
    dataset_name: str        = Form("", description="Target dataset: ds_policy | ds_regulation | ds_employees | ds_market | ds_workflow | ds_contracts | ds_reports. Leave empty for auto-classification."),
    hint:         str        = Form("", description="Auto-classify hint (used only when dataset_name is empty): policy | regulation | employees | market | workflow | contracts | reports"),
    temporal:     bool       = Form(False, description="Set True if document contains time-sensitive data"),
):
    """
    Upload a file and ingest it into the knowledge graph.

    **With dataset_name** → calls mcp-wrapper `cognify` directly on that dataset
    (follows `cognee.add → cognee.cognify` pattern, returns immediately).

    **Without dataset_name** → calls orchestrator `save` (LLM auto-classifies
    the document type and routes to the right typed dataset).

    Processing always runs in the background. Returns immediately with status=processing.
    """
    filename     = file.filename or "upload"
    content_type = file.content_type or ""
    data         = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext and ext not in _SUPPORTED_EXT:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

    text = extract_text(filename, content_type, data)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from file.")

    document_text = f"[File: {filename}]\n\n{text}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(35.0)) as client:

        ds = dataset_name.strip()

        if ds:
            # ── Direct path: mcp-wrapper cognify with explicit dataset_name ──
            # Mirrors: cognee.add(data, dataset_name=ds) → cognee.cognify(datasets=ds)
            try:
                raw = await _mcp_call(
                    client, _MCP_WRAPPER_URL, "cognify",
                    {"data": document_text, "dataset_name": ds, "temporal": temporal},
                )
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"message": raw[:300]}

                return {
                    "status":       "processing",
                    "filename":     filename,
                    "dataset_name": ds,
                    "temporal":     temporal,
                    "backend":      "mcp-wrapper",
                    **parsed,
                }
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"mcp-wrapper cognify failed: {exc}") from exc

        else:
            # ── Orchestrator path: auto-classify → typed dataset ──
            try:
                args: dict = {"data": document_text}
                if hint.strip():
                    args["hint"] = hint.strip().lower()

                raw = await _mcp_call(client, _ORCHESTRATOR_URL, "save", args)
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"message": raw[:300]}

                return {
                    "status":   "processing",
                    "filename": filename,
                    "backend":  "orchestrator",
                    **parsed,
                }
            except Exception as orch_exc:
                # Fallback: mcp-wrapper cognify without dataset (main_dataset)
                try:
                    raw = await _mcp_call(
                        client, _MCP_WRAPPER_URL, "cognify",
                        {"data": document_text},
                    )
                    return {
                        "status":   "processing",
                        "filename": filename,
                        "backend":  "mcp-wrapper-fallback",
                        "dataset":  "main_dataset",
                        "warning":  f"Orchestrator unavailable ({orch_exc})",
                        "message":  raw[:300],
                    }
                except Exception as fallback_exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Both backends failed. Orchestrator: {orch_exc}. MCP-wrapper: {fallback_exc}",
                    ) from fallback_exc


# ── Raw text endpoint ─────────────────────────────────────────────────────────

@router.post("/cognify")
async def cognify_text(
    data:         str  = Form(...,  description="Text content to store"),
    dataset_name: str  = Form("",   description="Target dataset name. Leave empty for auto-classification."),
    hint:         str  = Form("",   description="Auto-classify hint (used when dataset_name is empty)"),
    temporal:     bool = Form(False, description="True if content contains time-sensitive data"),
):
    """
    Submit raw text to the knowledge graph.

    Follows the same two-path logic as /upload:
    dataset_name → mcp-wrapper direct | no dataset_name → orchestrator auto-classify.
    """
    if not data.strip():
        raise HTTPException(status_code=400, detail="data must not be empty.")

    async with httpx.AsyncClient(timeout=httpx.Timeout(35.0)) as client:
        ds = dataset_name.strip()

        if ds:
            try:
                raw = await _mcp_call(
                    client, _MCP_WRAPPER_URL, "cognify",
                    {"data": data, "dataset_name": ds, "temporal": temporal},
                )
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"message": raw[:300]}
                return {"status": "processing", "dataset_name": ds, "backend": "mcp-wrapper", **parsed}
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        else:
            try:
                args: dict = {"data": data}
                if hint.strip():
                    args["hint"] = hint.strip().lower()
                raw = await _mcp_call(client, _ORCHESTRATOR_URL, "save", args)
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"message": raw[:300]}
                return {"status": "processing", "backend": "orchestrator", **parsed}
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
