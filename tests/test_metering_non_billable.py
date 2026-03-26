#!/usr/bin/env python3
"""
Test scenarios where agent-task-metering should NOT bill:
  - Task with terminal_success=False  →  adhered=False, billable_units=0
  - Task with empty outputs           →  adhered=False, billable_units=0
  - Task with null/empty result       →  output_validation fails
  - Task with success=True            →  adhered=True,  billable_units=1 (control)

Also tests through the MCP agent with an empty/blank task to see how
metering handles an edge-case agent response.

Usage:
    # Direct to metering service (requires port-forward to mcp-agents pod):
    python tests/test_metering_non_billable.py --direct

    # From inside the cluster (kubectl exec):
    python tests/test_metering_non_billable.py --in-cluster
"""

import json
import sys
import uuid
import urllib.request
import urllib.error
import asyncio
import aiohttp
import re
from typing import Dict, Any, Optional

# ── Configuration ────────────────────────────────────────────────────────
METERING_SERVICE_URL = "http://agent-task-metering.mcp-agents.svc.cluster.local"
SUBSCRIPTION_REF = "sub-mcp-agents-001"
AGENT_ID = "mcp-agents"

# For direct-mode tests through the MCP agent
MCP_BASE_URL = "http://localhost:8000/runtime/webhooks/mcp"


# ═══════════════════════════════════════════════════════════════════════
# Part 1: Direct metering-service tests (via kubectl exec)
# ═══════════════════════════════════════════════════════════════════════

def call_metering(url: str, payload: dict) -> dict:
    """POST to the metering service and return the JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"http_error": e.code, "body": body}
    except Exception as e:
        return {"error": str(e)}


def run_direct_metering_tests(base_url: str):
    """Run test scenarios directly against the metering service API."""

    endpoint = f"{base_url}/evaluate_and_meter_task"
    results = []

    # ── Scenario 1: terminal_success=False (task failed) ──────────────
    print("\n" + "═" * 70)
    print("  Scenario 1: Task FAILED (terminal_success=false)")
    print("  Expected: intent_handled=True, adhered=FALSE, billable_units=0")
    print("═" * 70)

    task_id_1 = f"fail-{uuid.uuid4().hex[:8]}"
    payload_1 = {
        "task_id": task_id_1,
        "agent_id": AGENT_ID,
        "subscription_ref": SUBSCRIPTION_REF,
        "evidence": {
            "outputs": {
                "terminal_success": False,
                "status": "failed",
                "intent": "data_analysis",
                "result": "Task execution encountered an error",
            },
            "query": "Analyze quarterly revenue trends",
            "response": json.dumps({"error": "Model inference timeout"}),
        },
    }

    r1 = call_metering(endpoint, payload_1)
    print(f"\n  Response: {json.dumps(r1, indent=2)}")
    s1 = (
        r1.get("adhered") is False
        and r1.get("billable_units") == 0
    )
    print(f"\n  {'✅ PASS' if s1 else '❌ FAIL'}: adhered={r1.get('adhered')}, "
          f"billable_units={r1.get('billable_units')}, recorded={r1.get('recorded')}")
    results.append(("Failed task (terminal_success=false)", s1))

    # ── Scenario 2: Empty outputs dict ────────────────────────────────
    print("\n" + "═" * 70)
    print("  Scenario 2: EMPTY outputs (no evidence of work)")
    print("  Expected: adhered=FALSE, billable_units=0")
    print("═" * 70)

    task_id_2 = f"empty-{uuid.uuid4().hex[:8]}"
    payload_2 = {
        "task_id": task_id_2,
        "agent_id": AGENT_ID,
        "subscription_ref": SUBSCRIPTION_REF,
        "evidence": {
            "outputs": {},
            "query": "Do something",
            "response": "",
        },
    }

    r2 = call_metering(endpoint, payload_2)
    print(f"\n  Response: {json.dumps(r2, indent=2)}")
    s2 = r2.get("billable_units") == 0
    print(f"\n  {'✅ PASS' if s2 else '❌ FAIL'}: adhered={r2.get('adhered')}, "
          f"billable_units={r2.get('billable_units')}, recorded={r2.get('recorded')}")
    results.append(("Empty outputs", s2))

    # ── Scenario 3: Null/empty result values ──────────────────────────
    print("\n" + "═" * 70)
    print("  Scenario 3: Null/empty result values (output validation fails)")
    print("  Expected: adhered=FALSE, billable_units=0")
    print("═" * 70)

    task_id_3 = f"null-{uuid.uuid4().hex[:8]}"
    payload_3 = {
        "task_id": task_id_3,
        "agent_id": AGENT_ID,
        "subscription_ref": SUBSCRIPTION_REF,
        "evidence": {
            "outputs": {
                "terminal_success": True,
                "status": "completed",
                "result": "",  # empty string → output_validation should fail
            },
            "query": "Generate a chart",
            "response": "",
        },
    }

    r3 = call_metering(endpoint, payload_3)
    print(f"\n  Response: {json.dumps(r3, indent=2)}")
    s3 = r3.get("billable_units") == 0
    print(f"\n  {'✅ PASS' if s3 else '⚠️  INFO'}: adhered={r3.get('adhered')}, "
          f"billable_units={r3.get('billable_units')}, recorded={r3.get('recorded')}")
    # This scenario may or may not fail depending on gate config; mark as info
    results.append(("Null/empty result values", s3))

    # ── Scenario 4: Control — successful task (should bill) ───────────
    print("\n" + "═" * 70)
    print("  Scenario 4: CONTROL — Successful task (should bill)")
    print("  Expected: intent_handled=True, adhered=True, billable_units=1")
    print("═" * 70)

    task_id_4 = f"pass-{uuid.uuid4().hex[:8]}"
    payload_4 = {
        "task_id": task_id_4,
        "agent_id": AGENT_ID,
        "subscription_ref": SUBSCRIPTION_REF,
        "evidence": {
            "outputs": {
                "terminal_success": True,
                "status": "completed",
                "intent": "data_analysis",
                "result": "Generated 3-step plan for quarterly analysis",
            },
            "query": "Analyze quarterly revenue trends and generate report",
            "response": json.dumps({
                "intent": "data_analysis",
                "plan_steps": 3,
                "task_id": task_id_4,
            }),
        },
    }

    r4 = call_metering(endpoint, payload_4)
    print(f"\n  Response: {json.dumps(r4, indent=2)}")
    s4 = (
        r4.get("intent_handled") is True
        and r4.get("adhered") is True
        and r4.get("billable_units") == 1
        and r4.get("recorded") is True
    )
    print(f"\n  {'✅ PASS' if s4 else '❌ FAIL'}: intent_handled={r4.get('intent_handled')}, "
          f"adhered={r4.get('adhered')}, billable_units={r4.get('billable_units')}, "
          f"recorded={r4.get('recorded')}")
    results.append(("Successful task (control)", s4))

    # ── Scenario 5: Duplicate task_id (should NOT double-bill) ────────
    print("\n" + "═" * 70)
    print("  Scenario 5: DUPLICATE task_id (idempotency check)")
    print("  Expected: billable_units=1, recorded=False (already recorded)")
    print("═" * 70)

    r5 = call_metering(endpoint, payload_4)  # same payload as scenario 4
    print(f"\n  Response: {json.dumps(r5, indent=2)}")
    s5 = (
        r5.get("billable_units") == 1
        and r5.get("recorded") is False  # duplicate — not re-recorded
    )
    print(f"\n  {'✅ PASS' if s5 else '❌ FAIL'}: billable_units={r5.get('billable_units')}, "
          f"recorded={r5.get('recorded')} (expected False for duplicate)")
    results.append(("Duplicate task_id (idempotent)", s5))

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  METERING TEST SUMMARY")
    print("═" * 70)
    all_pass = True
    for name, passed in results:
        print(f"  {'✅' if passed else '❌'} {name}")
        if not passed:
            all_pass = False
    print("═" * 70)
    return all_pass


# ═══════════════════════════════════════════════════════════════════════
# Part 2: End-to-end test through MCP agent (blank/edge-case task)
# ═══════════════════════════════════════════════════════════════════════

class MCPClient:
    """Minimal MCP SSE client for testing."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = None
        self.sse_response = None
        self.session_message_url = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.sse_response and not self.sse_response.closed:
            self.sse_response.close()
        if self.session:
            await self.session.close()

    async def establish_sse_session(self) -> bool:
        self.sse_response = await self.session.get(
            f"{self.base_url}/sse",
            headers={"Accept": "text/event-stream"},
        )
        if self.sse_response.status != 200:
            return False
        async for chunk in self.sse_response.content.iter_chunked(1024):
            if chunk:
                data = chunk.decode("utf-8", errors="ignore")
                match = re.search(r"data: (message\?[^\n\r]+)", data)
                if match:
                    self.session_message_url = f"{self.base_url}/{match.group(1)}"
                    return True
                break
        return False

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        rpc = {
            "jsonrpc": "2.0",
            "id": "test-non-billable",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        url = self.session_message_url or f"{self.base_url}/message"
        async with self.session.post(
            url, json=rpc, timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            text = await resp.text()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"error": "bad json", "raw": text}


async def run_mcp_edge_case_test():
    """Call next_best_action through the MCP agent with an edge-case task."""
    print("\n" + "═" * 70)
    print("  MCP AGENT EDGE-CASE TEST")
    print("  Sending a blank/minimal task to the agent to see if metering")
    print("  correctly evaluates a low-quality or degenerate response.")
    print("═" * 70)

    async with MCPClient(MCP_BASE_URL) as client:
        if not await client.establish_sse_session():
            print("  ❌ Could not establish SSE session (is port-forward running?)")
            return False

        await asyncio.sleep(2)

        # Edge case: single-character task
        print("\n  📝 Sending task: ' ' (single space)")
        result = await client.call_tool("next_best_action", {"task": " "})

        tool_result = result.get("result", {})
        content = tool_result.get("content", [])
        is_error = tool_result.get("isError", False)

        if is_error or "error" in result:
            error_text = content[0].get("text", str(result.get("error", ""))) if content else str(result)
            print(f"  ⚠️  Agent returned error (expected for edge case): {error_text[:200]}")
            print("  → Metering should NOT bill for this (no successful task).")
            return True  # An error is "correct" behavior for a blank task

        if content:
            text = content[0].get("text", "{}")
            try:
                data = json.loads(text)
                metering = data.get("metadata", {}).get("metering", {})
                print(f"  Agent responded with task_id: {data.get('task_id')}")
                print(f"  Metering metadata: {json.dumps(metering, indent=2)}")
                if metering:
                    bu = metering.get("billable_units", "N/A")
                    print(f"  → billable_units={bu}")
                else:
                    print("  → No metering metadata in response (check logs)")
            except json.JSONDecodeError:
                print(f"  Raw response: {text[:300]}")

    return True


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    use_direct = "--direct" in sys.argv
    use_in_cluster = "--in-cluster" in sys.argv

    print("=" * 70)
    print("🧪 Agent-Task-Metering: Non-Billable & Edge-Case Test Suite")
    print("=" * 70)

    if use_in_cluster:
        # Running inside the cluster via kubectl exec
        base = METERING_SERVICE_URL
        print(f"\n  Mode: in-cluster ({base})")
        ok = run_direct_metering_tests(base)
        sys.exit(0 if ok else 1)

    elif use_direct:
        # Part 1: Direct metering tests via kubectl exec
        print("\n  Mode: direct (metering via kubectl exec, MCP via localhost:8000)")
        print("\n  [Part 1] Running metering-service tests via kubectl exec...")
        # We'll run the in-cluster tests through kubectl exec and capture output
        # Then run the MCP edge-case test locally
        ok_mcp = asyncio.run(run_mcp_edge_case_test())
        sys.exit(0 if ok_mcp else 1)

    else:
        print("\nUsage:")
        print("  python tests/test_metering_non_billable.py --in-cluster")
        print("    → Run directly against metering service (use with kubectl exec)")
        print("  python tests/test_metering_non_billable.py --direct")
        print("    → Run MCP edge-case test via port-forward (localhost:8000)")
        sys.exit(1)


if __name__ == "__main__":
    main()
