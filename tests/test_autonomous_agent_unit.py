"""
Unit tests for autonomous_agent.py (Customer Churn Analysis Agent – CCA-001)

Tests cover:
- Health endpoint
- MCP initialize
- tools/list
- tools/call for each tool (get_customer_facts, get_churn_prediction,
  get_customer_segment, search_similar_churned, recommend_retention_action,
  create_retention_case, log_analysis_result)
- Tool validation (missing required args)
- Unknown tool handling
- MCPTool / MCPToolResult dataclasses
- execute_tool dispatcher
- Root endpoint
- SSE endpoint basics
- Invalid JSON-RPC version handling
- Unknown method handling
"""

import json
import sys
import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import asdict

# ---------------------------------------------------------------------------
# Patch heavy third-party imports BEFORE importing the module under test.
# ---------------------------------------------------------------------------

for _mod in ["dotenv"]:
    sys.modules.setdefault(_mod, MagicMock())
sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None

# Environment variables so the module doesn't try to connect
os.environ.setdefault("FOUNDRY_PROJECT_ENDPOINT", "")
os.environ.setdefault("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-test")

# Add src/ to sys.path
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))

import autonomous_agent as agent


# ===========================================================================
#  Helper
# ===========================================================================

def _run(coro):
    """Run an async coroutine in a new event loop (test helper)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
#  Tests – Health endpoint
# ===========================================================================

class TestHealthEndpoint(unittest.TestCase):
    def test_health_returns_healthy(self):
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)


# ===========================================================================
#  Tests – Root endpoint
# ===========================================================================

class TestRootEndpoint(unittest.TestCase):
    def test_root_returns_server_info(self):
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "customer-churn-agent")
        self.assertEqual(data["spec_id"], "CCA-001")
        self.assertIn("tools", data)
        self.assertIn("get_customer_facts", data["tools"])


# ===========================================================================
#  Tests – MCP Initialize
# ===========================================================================

class TestMCPInitialize(unittest.TestCase):
    def test_initialize_returns_capabilities(self):
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        resp = client.post(
            "/runtime/webhooks/mcp/message",
            json={
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {},
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["jsonrpc"], "2.0")
        self.assertEqual(data["id"], "init-1")
        result = data["result"]
        self.assertEqual(result["protocolVersion"], "2024-11-05")
        self.assertIn("tools", result["capabilities"])
        self.assertEqual(result["serverInfo"]["name"], "customer-churn-agent")

    def test_invalid_jsonrpc_version(self):
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        resp = client.post(
            "/runtime/webhooks/mcp/message",
            json={
                "jsonrpc": "1.0",
                "id": "bad-1",
                "method": "initialize",
            },
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data["error"]["code"], -32600)

    def test_unknown_method(self):
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        resp = client.post(
            "/runtime/webhooks/mcp/message",
            json={
                "jsonrpc": "2.0",
                "id": "unk-1",
                "method": "nonexistent/method",
            },
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertEqual(data["error"]["code"], -32601)


# ===========================================================================
#  Tests – tools/list
# ===========================================================================

class TestToolsList(unittest.TestCase):
    def test_tools_list_returns_all_tools(self):
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        resp = client.post(
            "/runtime/webhooks/mcp/message",
            json={
                "jsonrpc": "2.0",
                "id": "list-1",
                "method": "tools/list",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        tools = data["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        expected = [
            "get_customer_facts",
            "get_churn_prediction",
            "get_customer_segment",
            "search_similar_churned",
            "recommend_retention_action",
            "create_retention_case",
            "log_analysis_result",
        ]
        for name in expected:
            self.assertIn(name, tool_names, f"Tool '{name}' missing from tools/list")

    def test_tools_have_input_schema(self):
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        resp = client.post(
            "/runtime/webhooks/mcp/message",
            json={"jsonrpc": "2.0", "id": "list-2", "method": "tools/list"},
        )
        tools = resp.json()["result"]["tools"]
        for tool in tools:
            self.assertIn("inputSchema", tool, f"Tool '{tool['name']}' missing inputSchema")
            self.assertEqual(tool["inputSchema"]["type"], "object")


# ===========================================================================
#  Tests – tools/call
# ===========================================================================

class TestToolsCall(unittest.TestCase):
    """Test tools/call for each tool through the MCP message endpoint."""

    def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        resp = client.post(
            "/runtime/webhooks/mcp/message",
            json={
                "jsonrpc": "2.0",
                "id": f"call-{tool_name}",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    # -- get_customer_facts --------------------------------------------------
    def test_get_customer_facts(self):
        data = self._call_tool("get_customer_facts", {"customer_id": "C-100"})
        result = data["result"]
        self.assertFalse(result["isError"])
        content_text = result["content"][0]["text"]
        payload = json.loads(content_text)
        self.assertEqual(payload["customer_id"], "C-100")
        self.assertIn("facts", payload)

    def test_get_customer_facts_missing_id(self):
        data = self._call_tool("get_customer_facts", {})
        result = data["result"]
        self.assertTrue(result["isError"])

    # -- get_churn_prediction ------------------------------------------------
    def test_get_churn_prediction(self):
        data = self._call_tool("get_churn_prediction", {"customer_id": "C-200"})
        result = data["result"]
        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertEqual(payload["customer_id"], "C-200")
        self.assertIn("churn_score", payload)
        self.assertIn("drivers", payload)

    def test_get_churn_prediction_missing_id(self):
        data = self._call_tool("get_churn_prediction", {})
        self.assertTrue(data["result"]["isError"])

    # -- get_customer_segment ------------------------------------------------
    def test_get_customer_segment(self):
        data = self._call_tool("get_customer_segment", {"customer_id": "C-300"})
        result = data["result"]
        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertEqual(payload["customer_id"], "C-300")
        self.assertIn("segment", payload)

    def test_get_customer_segment_missing_id(self):
        data = self._call_tool("get_customer_segment", {})
        self.assertTrue(data["result"]["isError"])

    # -- search_similar_churned ----------------------------------------------
    def test_search_similar_churned(self):
        profile = {"segment": "business", "tenure_months": 18}
        data = self._call_tool("search_similar_churned", {"customer_profile": profile, "limit": 2})
        result = data["result"]
        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertIn("similar_churned", payload)

    def test_search_similar_churned_missing_profile(self):
        data = self._call_tool("search_similar_churned", {})
        self.assertTrue(data["result"]["isError"])

    # -- recommend_retention_action ------------------------------------------
    def test_recommend_retention_action_critical(self):
        data = self._call_tool("recommend_retention_action", {"customer_id": "C-400", "risk_level": "critical"})
        result = data["result"]
        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertEqual(payload["risk_level"], "critical")
        self.assertEqual(payload["recommendation"]["priority"], "immediate")

    def test_recommend_retention_action_elevated(self):
        data = self._call_tool("recommend_retention_action", {"customer_id": "C-401", "risk_level": "elevated"})
        payload = json.loads(data["result"]["content"][0]["text"])
        self.assertEqual(payload["recommendation"]["priority"], "this_week")

    def test_recommend_retention_action_normal(self):
        data = self._call_tool("recommend_retention_action", {"customer_id": "C-402", "risk_level": "normal"})
        payload = json.loads(data["result"]["content"][0]["text"])
        self.assertEqual(payload["recommendation"]["priority"], "standard")

    def test_recommend_retention_action_missing_args(self):
        data = self._call_tool("recommend_retention_action", {"customer_id": "C-400"})
        self.assertTrue(data["result"]["isError"])

    # -- create_retention_case -----------------------------------------------
    def test_create_retention_case(self):
        action = {"type": "personal_outreach", "description": "Call customer"}
        data = self._call_tool("create_retention_case", {"customer_id": "C-500", "action": action})
        result = data["result"]
        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertIn("case_id", payload)
        self.assertEqual(payload["status"], "open")

    def test_create_retention_case_missing_action(self):
        data = self._call_tool("create_retention_case", {"customer_id": "C-500"})
        self.assertTrue(data["result"]["isError"])

    # -- log_analysis_result -------------------------------------------------
    def test_log_analysis_result(self):
        result_obj = {"risk": "critical", "score": 0.82}
        data = self._call_tool("log_analysis_result", {"customer_id": "C-600", "result": result_obj})
        result = data["result"]
        self.assertFalse(result["isError"])
        payload = json.loads(result["content"][0]["text"])
        self.assertIn("log_id", payload)

    def test_log_analysis_result_missing_result(self):
        data = self._call_tool("log_analysis_result", {"customer_id": "C-600"})
        self.assertTrue(data["result"]["isError"])

    # -- unknown tool --------------------------------------------------------
    def test_unknown_tool(self):
        data = self._call_tool("nonexistent_tool", {})
        self.assertTrue(data["result"]["isError"])
        self.assertIn("Unknown tool", data["result"]["content"][0]["text"])


# ===========================================================================
#  Tests – execute_tool directly
# ===========================================================================

class TestExecuteToolDirect(unittest.TestCase):
    def test_execute_tool_returns_result(self):
        result = _run(agent.execute_tool("get_customer_facts", {"customer_id": "C-1"}))
        self.assertIsInstance(result, agent.MCPToolResult)
        self.assertFalse(result.isError)

    def test_execute_tool_unknown(self):
        result = _run(agent.execute_tool("does_not_exist", {}))
        self.assertTrue(result.isError)


# ===========================================================================
#  Tests – MCPTool / MCPToolResult dataclasses
# ===========================================================================

class TestDataclasses(unittest.TestCase):
    def test_mcp_tool_fields(self):
        tool = agent.MCPTool(name="t", description="d", inputSchema={"type": "object"})
        d = asdict(tool)
        self.assertEqual(d["name"], "t")
        self.assertEqual(d["description"], "d")

    def test_mcp_tool_result_defaults(self):
        r = agent.MCPToolResult(content=[{"type": "text", "text": "ok"}])
        self.assertFalse(r.isError)

    def test_mcp_tool_result_error(self):
        r = agent.MCPToolResult(content=[], isError=True)
        self.assertTrue(r.isError)


# ===========================================================================
#  Tests – SSE endpoint
# ===========================================================================

class TestSSEEndpoint(unittest.TestCase):
    def test_sse_returns_event_stream(self):
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        with client.stream("GET", "/runtime/webhooks/mcp/sse") as resp:
            self.assertEqual(resp.status_code, 200)
            self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
            # Read first chunk – should contain the message URL
            for chunk in resp.iter_text():
                self.assertIn("message?sessionId=", chunk)
                break


# ===========================================================================
#  Tests – Tool function unit tests
# ===========================================================================

class TestToolFunctions(unittest.TestCase):
    """Direct unit tests for the underlying tool functions."""

    def test_get_customer_facts_returns_dict(self):
        result = agent.get_customer_facts("C-10")
        self.assertEqual(result["customer_id"], "C-10")
        self.assertIn("facts", result)
        self.assertIn("tenure_months", result["facts"])

    def test_get_churn_prediction_returns_score(self):
        result = agent.get_churn_prediction("C-20")
        self.assertIn("churn_score", result)
        self.assertGreater(result["churn_score"], 0)
        self.assertLessEqual(result["churn_score"], 1)

    def test_get_customer_segment_returns_segment(self):
        result = agent.get_customer_segment("C-30")
        self.assertIn("segment", result)

    def test_search_similar_churned_respects_limit(self):
        result = agent.search_similar_churned({"segment": "business"}, limit=1)
        self.assertLessEqual(len(result["similar_churned"]), 1)

    def test_recommend_retention_action_unknown_risk(self):
        result = agent.recommend_retention_action("C-40", "unknown_level")
        # Falls back to "normal" actions
        self.assertEqual(result["recommendation"]["priority"], "standard")

    def test_create_retention_case_generates_id(self):
        result = agent.create_retention_case("C-50", {"type": "discount"})
        self.assertTrue(result["case_id"].startswith("CASE-"))

    def test_log_analysis_result_generates_id(self):
        result = agent.log_analysis_result("C-60", {"risk": "low"})
        self.assertTrue(result["log_id"].startswith("LOG-"))


if __name__ == "__main__":
    unittest.main()
