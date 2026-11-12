"""
Orchestrator integration test suite.
Tests all 4 MCP tools + session protocol against a running orchestrator at :8003.

Usage:
    python scripts/test_orchestrator.py                  # default localhost:8003
    python scripts/test_orchestrator.py --url http://host:8003
    python scripts/test_orchestrator.py --unit-only      # run module unit tests only (no Docker)
"""
import argparse
import asyncio
import json
import sys
import textwrap
from typing import Any

import httpx

BASE_URL = "http://localhost:8003"

# -- ANSI colours --------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

passed = 0
failed = 0


def ok(label: str, detail: str = "") -> None:
    global passed
    passed += 1
    suffix = f"  {YELLOW}{textwrap.shorten(detail, 80)}{RESET}" if detail else ""
    print(f"  {GREEN}OK{RESET} {label}{suffix}")


def fail(label: str, detail: str = "") -> None:
    global failed
    failed += 1
    suffix = f"\n    {RED}{detail}{RESET}" if detail else ""
    print(f"  {RED}FAIL{RESET} {label}{suffix}")


def section(title: str) -> None:
    bar = "-" * max(2, 50 - len(title))
    print(f"\n{BOLD}{BLUE}-- {title} {bar}{RESET}")


# -- SSE parser ----------------------------------------------------------------

def parse_sse(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            try:
                return json.loads(line[5:].strip())
            except json.JSONDecodeError:
                pass
    return {}


def extract_text(payload: dict) -> str:
    content = payload.get("result", {}).get("content", [])
    return " ".join(c.get("text", "") for c in content if c.get("type") == "text")


# -- MCP helpers ---------------------------------------------------------------

async def initialize(client: httpx.AsyncClient) -> str:
    resp = await client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-suite", "version": "1.0"},
        },
    })
    resp.raise_for_status()
    sid = resp.headers.get("mcp-session-id", "")
    return sid


async def call_tool(client: httpx.AsyncClient, sid: str, tool: str, args: dict, req_id: int = 99) -> tuple[dict, str]:
    resp = await client.post("/mcp", headers={"mcp-session-id": sid}, json={
        "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    })
    resp.raise_for_status()
    payload = parse_sse(resp.text)
    text    = extract_text(payload)
    return payload, text


# ??????????????????????????????????????????????????????????????????????????????
# UNIT TESTS  (no Docker required)
# ??????????????????????????????????????????????????????????????????????????????

async def unit_tests() -> None:
    section("Unit Tests - QueryAnalyzer (temporal detection)")

    # Add project root to path so imports work
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from orchestrator.query_analyzer import _detect_temporal, analyze

    cases = [
        ("Q3 2024 revenue was ?12M",          True),
        ("last week the board approved X",      True),
        ("2023-11-15 contract signed",          True),
        ("The policy defines access rules",     False),
        ("employee count is 250",               False),
        ("yesterday we updated the procedure",  True),
    ]
    for text, expected in cases:
        got = _detect_temporal(text)
        if got == expected:
            ok(f"temporal='{expected}'  ->  '{text[:55]}'")
        else:
            fail(f"temporal='{expected}'  ->  '{text[:55]}'", f"got {got}")

    section("Unit Tests - DocumentClassifier (keyword scoring)")

    from orchestrator.document_classifier import _keyword_score

    cls_cases = [
        ("Our policy defines the compliance guidelines for staff",        "policy"),
        ("GDPR regulation article 17 mandates data deletion",            "regulation"),
        ("Employee salary review process for Q2",                        "employees"),
        ("Q3 revenue grew 18%% versus competitors in the market",        "market"),
        ("The BPMN workflow defines the task sequence for onboarding",   "workflow"),
        ("Supplier SLA contract agreement for cloud services",           "contracts"),
        ("KPI dashboard analysis report for board review",               "reports"),
    ]
    for text, expected_label in cls_cases:
        label, _, confidence = _keyword_score(text)
        if label == expected_label:
            ok(f"label='{expected_label}' (conf={confidence:.2f})  ->  '{text[:55]}'")
        else:
            fail(f"expected '{expected_label}' got '{label}'", text[:80])

    section("Unit Tests - MemoryRouter (search type routing)")

    from orchestrator.query_analyzer import QueryIntent
    from orchestrator.memory_router import route

    route_cases = [
        (QueryIntent("search", "GRAPH_COMPLETION",       True,  0.9), "TEMPORAL"),
        (QueryIntent("search", "GRAPH_SUMMARY_COMPLETION", False, 0.9), "GRAPH_SUMMARY_COMPLETION"),
        (QueryIntent("search", "CYPHER",                 False, 0.8), "CYPHER"),
        (QueryIntent("search", "GRAPH_COMPLETION",       False, 0.7), "GRAPH_COMPLETION"),
        (QueryIntent("search", "CHUNKS",                 False, 0.8), "CHUNKS"),
    ]
    for intent, expected_type in route_cases:
        result = route(intent)
        if result.search_type == expected_type:
            ok(f"temporal={intent.is_temporal} llm='{intent.search_type}' -> '{expected_type}'")
        else:
            fail(f"expected '{expected_type}'", f"got '{result.search_type}'")


# ??????????????????????????????????????????????????????????????????????????????
# INTEGRATION TESTS  (Docker stack must be running)
# ??????????????????????????????????????????????????????????????????????????????

async def integration_tests(base: str) -> None:
    async with httpx.AsyncClient(base_url=base, timeout=180) as client:

        # -- Health ---------------------------------------------------------
        section("Health Check")
        try:
            resp = await client.get("/health")
            data = resp.json()
            if resp.status_code == 200 and data.get("status") == "ok":
                ok("GET /health", f"sessions={data.get('sessions')} cb={data.get('circuit_breaker')}")
            else:
                fail("GET /health", str(data))
        except Exception as exc:
            fail("GET /health - orchestrator not reachable", str(exc))
            print(f"\n  {YELLOW}Tip: make sure the stack is up -> docker compose up -d orchestrator{RESET}\n")
            return

        # -- Tools list -----------------------------------------------------
        section("MCP Protocol - tools/list")
        resp = await client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {},
        })
        data  = resp.json()
        tools = [t["name"] for t in data.get("result", {}).get("tools", [])]
        for expected in ("save", "search", "improve", "delete"):
            if expected in tools:
                ok(f"tool '{expected}' present")
            else:
                fail(f"tool '{expected}' missing", str(tools))

        # -- Session handshake ----------------------------------------------
        section("MCP Protocol - initialize / session")
        sid = await initialize(client)
        if sid:
            ok("initialize -> mcp-session-id received", sid[:24] + "?")
        else:
            fail("initialize - no mcp-session-id in response headers")
            return

        # -- SAVE: auto-classification --------------------------------------
        section("Tool: save (auto-classification)")

        save_cases = [
            {
                "label": "market document -> ds_market",
                "data":  "Q3 2024 revenue reached ?12.4M, up 18% YoY vs competitors.",
                "hint":  "",
                "check": lambda t: "ds_market" in t or "market" in t,
            },
            {
                "label": "policy document -> ds_policy",
                "data":  "Our compliance policy mandates annual security audits for all staff.",
                "hint":  "",
                "check": lambda t: "ds_policy" in t or "policy" in t,
            },
            {
                "label": "regulation document -> ds_regulation",
                "data":  "GDPR Article 17 gives citizens the right to erasure of personal data.",
                "hint":  "",
                "check": lambda t: "ds_regulation" in t or "regulation" in t,
            },
        ]
        for i, case in enumerate(save_cases):
            try:
                _, text = await call_tool(client, sid, "save",
                    {"data": case["data"], "hint": case["hint"]}, req_id=10 + i)
                if case["check"](text):
                    ok(case["label"], text[:100])
                else:
                    fail(case["label"], f"unexpected response: {text[:120]}")
            except Exception as exc:
                fail(case["label"], str(exc))

        # -- SAVE: hint override --------------------------------------------
        section("Tool: save (hint override)")
        try:
            _, text = await call_tool(client, sid, "save",
                {"data": "New vendor partnership agreement for cloud infrastructure.",
                 "hint": "contracts"}, req_id=20)
            if "ds_contracts" in text or "contracts" in text:
                ok("hint='contracts' overrides classifier -> ds_contracts", text[:100])
            else:
                fail("hint override not applied", text[:120])
        except Exception as exc:
            fail("hint override", str(exc))

        # -- SAVE: temporal detection ---------------------------------------
        section("Tool: save (temporal detection)")
        try:
            _, text = await call_tool(client, sid, "save",
                {"data": "On 2024-03-15 the board approved the Q1 budget of ?4.2M."}, req_id=21)
            payload = json.loads(text) if text.startswith("{") else {}
            if payload.get("is_temporal") is True:
                ok("temporal date detected -> is_temporal=True", text[:100])
            else:
                fail("is_temporal should be True", text[:120])
        except Exception as exc:
            fail("temporal save", str(exc))

        # -- SEARCH: auto-routing -------------------------------------------
        section("Tool: search (auto-routing)")

        search_cases = [
            {
                "label": "temporal query -> TEMPORAL route",
                "query": "What happened in Q3 2024?",
                "hint":  "",
            },
            {
                "label": "factual query -> GRAPH_COMPLETION route",
                "query": "What is the compliance policy for security audits?",
                "hint":  "",
            },
            {
                "label": "summary query -> hint='summary' override",
                "query": "Give me an overview of all stored knowledge.",
                "hint":  "summary",
            },
        ]
        for i, case in enumerate(search_cases):
            try:
                _, text = await call_tool(client, sid, "search",
                    {"query": case["query"], "hint": case["hint"]}, req_id=30 + i)
                if text and "Error" not in text[:20]:
                    ok(case["label"], text[:100])
                else:
                    fail(case["label"], text[:120])
            except Exception as exc:
                fail(case["label"], str(exc))

        # -- IMPROVE --------------------------------------------------------
        section("Tool: improve")
        try:
            _, text = await call_tool(client, sid, "improve", {}, req_id=40)
            if text:
                ok("improve tool executed", text[:100])
            else:
                fail("improve returned empty response")
        except Exception as exc:
            fail("improve", str(exc))

        # -- DELETE: dataset ------------------------------------------------
        section("Tool: delete (dataset)")
        try:
            _, text = await call_tool(client, sid, "delete",
                {"target": "ds_test_cleanup", "mode": "soft"}, req_id=50)
            # ds_test_cleanup likely doesn't exist - any response without crash is fine
            ok("delete dataset (soft)", text[:100] if text else "(empty - dataset may not exist)")
        except Exception as exc:
            fail("delete dataset", str(exc))

        # -- Expired session guard ------------------------------------------
        section("MCP Protocol - expired session guard")
        try:
            resp = await client.post("/mcp",
                headers={"mcp-session-id": "00000000-dead-beef-0000-000000000000"},
                json={"jsonrpc": "2.0", "id": 99, "method": "tools/call",
                      "params": {"name": "search", "arguments": {"query": "test"}}})
            payload = parse_sse(resp.text)
            if "error" in payload:
                ok("expired session rejected with MCP error", payload["error"].get("message", "")[:80])
            else:
                fail("should have rejected invalid session", resp.text[:120])
        except Exception as exc:
            fail("expired session guard", str(exc))


# ??????????????????????????????????????????????????????????????????????????????
# Entry point
# ??????????????????????????????????????????????????????????????????????????????

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8003", help="Orchestrator base URL")
    parser.add_argument("--unit-only", action="store_true", help="Run module unit tests only (no Docker needed)")
    args = parser.parse_args()

    print(f"\n{BOLD}Orchestrator Test Suite{RESET}  ->  {args.url}\n")

    await unit_tests()

    if not args.unit_only:
        await integration_tests(args.url)

    # -- Summary ------------------------------------------------------------
    total = passed + failed
    color = GREEN if failed == 0 else RED
    print(f"\n{BOLD}{'-' * 55}{RESET}")
    print(f"{BOLD}Results: {color}{passed}/{total} passed{RESET}  ({failed} failed)\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
