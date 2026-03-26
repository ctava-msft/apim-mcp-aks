#!/usr/bin/env python3
"""Run metering non-billable test scenarios (runs INSIDE the cluster)."""
import json
import os
import time
import urllib.request

url = os.environ.get(
    "METERING_URL",
    "http://agent-task-metering.mcp-agents.svc.cluster.local/evaluate_and_meter_task",
)

# Use a unique suffix per run to avoid in-memory deduplication across runs
_run = str(int(time.time()))

scenarios = [
    {
        "name": "Scenario 1: FAILED TASK (terminal_success=false)",
        "expect_bu": 0,
        "expect_recorded": None,
        "payload": {
            "task_id": f"fail-test-{_run}-001",
            "agent_id": "mcp-agents",
            "subscription_ref": "sub-mcp-agents-001",
            "evidence": {
                "outputs": {
                    "terminal_success": False,
                    "status": "failed",
                    "result": "Error occurred during processing",
                },
                "query": "Analyze quarterly revenue trends",
                "response": '{"error": "Model timeout"}',
            },
        },
    },
    {
        "name": "Scenario 2: EMPTY OUTPUTS (no evidence of work)",
        "expect_bu": 0,
        "expect_recorded": None,
        "payload": {
            "task_id": f"empty-test-{_run}-002",
            "agent_id": "mcp-agents",
            "subscription_ref": "sub-mcp-agents-001",
            "evidence": {
                "outputs": {},
                "query": "Do something",
                "response": "",
            },
        },
    },
    {
        "name": "Scenario 3: EMPTY RESULT VALUE (output_validation gate)",
        "expect_bu": 0,
        "expect_recorded": None,
        "payload": {
            "task_id": f"null-test-{_run}-003",
            "agent_id": "mcp-agents",
            "subscription_ref": "sub-mcp-agents-001",
            "evidence": {
                "outputs": {
                    "terminal_success": True,
                    "status": "completed",
                    "result": "",
                },
                "query": "Generate a chart",
                "response": "",
            },
        },
    },
    {
        "name": "Scenario 4: CONTROL - SUCCESSFUL TASK (should bill)",
        "expect_bu": 1,
        "expect_recorded": True,
        "payload": {
            "task_id": f"pass-test-{_run}-004",
            "agent_id": "mcp-agents",
            "subscription_ref": "sub-mcp-agents-001",
            "evidence": {
                "outputs": {
                    "terminal_success": True,
                    "status": "completed",
                    "result": "Generated 3-step plan for analysis",
                },
                "query": "Analyze quarterly revenue trends and generate report",
                "response": '{"intent": "data_analysis", "plan_steps": 3}',
            },
        },
    },
    {
        "name": "Scenario 5: DUPLICATE task_id (idempotency - no double-bill)",
        "expect_bu": 1,
        "expect_recorded": False,
        "payload": {
            "task_id": f"pass-test-{_run}-004",
            "agent_id": "mcp-agents",
            "subscription_ref": "sub-mcp-agents-001",
            "evidence": {
                "outputs": {
                    "terminal_success": True,
                    "status": "completed",
                    "result": "Generated 3-step plan for analysis",
                },
                "query": "Analyze quarterly revenue trends and generate report",
                "response": '{"intent": "data_analysis", "plan_steps": 3}',
            },
        },
    },
]

results = []
for s in scenarios:
    print()
    print("=" * 65)
    print("  " + s["name"])
    print("=" * 65)
    data = json.dumps(s["payload"]).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            r = json.loads(resp.read().decode("utf-8"))
            print("  intent_handled: " + str(r.get("intent_handled")))
            print("  adhered:        " + str(r.get("adhered")))
            print("  billable_units: " + str(r.get("billable_units")))
            print("  recorded:       " + str(r.get("recorded")))
            print("  reason_codes:   " + str(r.get("reason_codes")))
            cid = r.get("correlation_id", "")
            print("  correlation_id: " + cid[:16] + "...")

            bu = r.get("billable_units", -1)
            rec = r.get("recorded")
            ok = bu == s["expect_bu"]
            if s["expect_recorded"] is not None:
                ok = ok and rec == s["expect_recorded"]

            tag = "PASS" if ok else "FAIL"
            print("  Result: " + tag)
            results.append((s["name"], ok))
    except Exception as e:
        print("  ERROR: " + str(e))
        results.append((s["name"], False))

print()
print("=" * 65)
print("  SUMMARY")
print("=" * 65)
all_ok = True
for name, passed in results:
    mark = "PASS" if passed else "FAIL"
    print("  [" + mark + "] " + name)
    if not passed:
        all_ok = False
print("=" * 65)
if all_ok:
    print("  All scenarios passed!")
else:
    print("  Some scenarios FAILED.")
