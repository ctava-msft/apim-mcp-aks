"""
Unit tests for autonomous_agent.py (Utilization Management Agent)

Tests cover:
- Health endpoint
- MCP initialize
- tools/list
- tools/call (UM domain tools + Lightning tools)
- MCPTool / MCPToolResult dataclasses
- get_model_deployment
- _decompose_question
- _detect_conflicts
- _structured_summary
- execute_tool dispatcher (Lightning unavailable paths)
"""

import json
import sys
import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import asdict
from enum import Enum as _Enum

# ---------------------------------------------------------------------------
# Patch heavy third-party imports BEFORE importing the module under test.
# ---------------------------------------------------------------------------

# ---- memory stubs ----------------------------------------------------------
_memory_mod = MagicMock()


class _MemoryType(_Enum):
    CONTEXT = "context"
    CONVERSATION = "conversation"
    TASK = "task"
    PLAN = "plan"


_memory_mod.MemoryType = _MemoryType
_memory_mod.AISEARCH_CONTEXT_PROVIDER_AVAILABLE = False

for _name in [
    "ShortTermMemory", "MemoryEntry", "CompositeMemory", "LongTermMemory",
    "MemoryProvider", "MemorySearchResult",
]:
    if not hasattr(_memory_mod, _name):
        setattr(_memory_mod, _name, MagicMock())

sys.modules.setdefault("memory", _memory_mod)

# ---- lightning stub --------------------------------------------------------
sys.modules.setdefault("lightning", MagicMock())

# ---- azure SDK stubs -------------------------------------------------------
for _mod in [
    "azure.identity",
    "azure.cosmos",
    "azure.cosmos.exceptions",
    "openai",
    "dotenv",
]:
    sys.modules.setdefault(_mod, MagicMock())

sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None

# ---- environment variables -------------------------------------------------
os.environ.setdefault("COSMOSDB_ENDPOINT", "")
os.environ.setdefault("FOUNDRY_PROJECT_ENDPOINT", "")
os.environ.setdefault("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-test")
os.environ.setdefault("EMBEDDING_MODEL_DEPLOYMENT_NAME", "embed-test")
os.environ.setdefault("AZURE_SEARCH_ENDPOINT", "")
os.environ.setdefault("ENABLE_LIGHTNING_CAPTURE", "false")
os.environ.setdefault("USE_TUNED_MODEL", "false")

# Add src/ to sys.path
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))

import autonomous_agent as agent


# ===========================================================================
#  Helper
# ===========================================================================

def _run(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
#  Tests – Dataclasses
# ===========================================================================

class TestMCPToolDataclass(unittest.TestCase):
    def test_mcp_tool_fields(self):
        tool = agent.MCPTool(name="test", description="desc", inputSchema={"type": "object"})
        self.assertEqual(tool.name, "test")
        self.assertEqual(tool.description, "desc")
        self.assertEqual(tool.inputSchema, {"type": "object"})

    def test_mcp_tool_result_default(self):
        result = agent.MCPToolResult(content=[{"type": "text", "text": "ok"}])
        self.assertFalse(result.isError)

    def test_mcp_tool_result_error(self):
        result = agent.MCPToolResult(content=[{"type": "text", "text": "err"}], isError=True)
        self.assertTrue(result.isError)

    def test_mcp_tool_result_asdict(self):
        result = agent.MCPToolResult(content=[{"type": "text", "text": "hello"}])
        d = asdict(result)
        self.assertIn("content", d)
        self.assertIn("isError", d)
        self.assertFalse(d["isError"])


# ===========================================================================
#  Tests – get_model_deployment
# ===========================================================================

class TestGetModelDeployment(unittest.TestCase):
    def test_returns_base_model_when_tuned_disabled(self):
        with patch.object(agent, "USE_TUNED_MODEL", False):
            result = agent.get_model_deployment()
            self.assertEqual(result, agent.FOUNDRY_MODEL_DEPLOYMENT_NAME)

    def test_returns_base_model_when_no_registry(self):
        with patch.object(agent, "USE_TUNED_MODEL", True), \
             patch.object(agent, "deployment_registry", None), \
             patch.object(agent, "TUNED_MODEL_DEPLOYMENT_NAME", ""):
            result = agent.get_model_deployment()
            self.assertEqual(result, agent.FOUNDRY_MODEL_DEPLOYMENT_NAME)

    def test_returns_tuned_from_env(self):
        with patch.object(agent, "USE_TUNED_MODEL", True), \
             patch.object(agent, "deployment_registry", None), \
             patch.object(agent, "TUNED_MODEL_DEPLOYMENT_NAME", "ft:gpt-4o-mini:custom"):
            result = agent.get_model_deployment()
            self.assertEqual(result, "ft:gpt-4o-mini:custom")


# ===========================================================================
#  Tests – Question Decomposition
# ===========================================================================

class TestDecomposeQuestion(unittest.TestCase):
    def test_prior_auth_detected(self):
        result = agent._decompose_question("Is prior authorization required for this biologic?")
        sources = [sq["source_type"] for sq in result]
        self.assertIn("coverage_policy", sources)

    def test_step_therapy_detected(self):
        result = agent._decompose_question("What are the step therapy requirements?")
        sources = [sq["source_type"] for sq in result]
        self.assertIn("step_therapy", sources)

    def test_turnaround_time_detected(self):
        result = agent._decompose_question("What is the urgent turnaround time?")
        sources = [sq["source_type"] for sq in result]
        self.assertIn("operations", sources)

    def test_continuity_of_care_detected(self):
        result = agent._decompose_question("Does continuity of care apply for out-of-network provider?")
        sources = [sq["source_type"] for sq in result]
        self.assertIn("regulatory", sources)

    def test_medicare_regulatory_detected(self):
        result = agent._decompose_question("What does CMS regulation say about this?")
        sources = [sq["source_type"] for sq in result]
        self.assertIn("regulatory", sources)

    def test_fallback_for_generic_query(self):
        result = agent._decompose_question("Tell me about this policy")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_type"], "coverage_policy")

    def test_multiple_topics_detected(self):
        result = agent._decompose_question(
            "Is prior authorization required? What is the step therapy process? "
            "What is the urgent turnaround time?"
        )
        sources = [sq["source_type"] for sq in result]
        self.assertIn("coverage_policy", sources)
        self.assertIn("step_therapy", sources)
        self.assertIn("operations", sources)


# ===========================================================================
#  Tests – Conflict Detection
# ===========================================================================

class TestDetectConflicts(unittest.TestCase):
    def test_policy_exception_conflict(self):
        evidence = {
            "q1": {"source_type": "coverage_policy", "result_count": 1},
            "q2": {"source_type": "exception_logic", "result_count": 1},
        }
        conflicts = agent._detect_conflicts(evidence)
        self.assertTrue(any(c["type"] == "policy_exception_overlap" for c in conflicts))

    def test_ops_vs_policy_conflict(self):
        evidence = {
            "q1": {"source_type": "operations", "result_count": 1},
            "q2": {"source_type": "coverage_policy", "result_count": 1},
        }
        conflicts = agent._detect_conflicts(evidence)
        self.assertTrue(any(c["type"] == "ops_vs_policy" for c in conflicts))

    def test_no_conflict(self):
        evidence = {
            "q1": {"source_type": "regulatory", "result_count": 1},
        }
        conflicts = agent._detect_conflicts(evidence)
        self.assertEqual(len(conflicts), 0)


# ===========================================================================
#  Tests – Structured Summary
# ===========================================================================

class TestStructuredSummary(unittest.TestCase):
    def test_basic_summary(self):
        retrieval = {
            "evidence": {
                "PA required?": {
                    "source_type": "coverage_policy",
                    "results": [{"title": "PA Policy v2", "content": "PA is required for biologics", "score": 0.9}],
                },
            },
            "conflicts": [],
        }
        summary = agent._structured_summary("Is PA required?", retrieval)
        self.assertIn("PA Policy v2", summary)
        self.assertIn("Coverage Policy", summary)

    def test_summary_with_conflicts(self):
        retrieval = {
            "evidence": {},
            "conflicts": [{"type": "policy_exception_overlap", "description": "Conflicting", "resolution": "Check exception"}],
        }
        summary = agent._structured_summary("test", retrieval)
        self.assertIn("Conflicts Detected", summary)
        self.assertIn("policy_exception_overlap", summary)


# ===========================================================================
#  Tests – TOOLS list
# ===========================================================================

class TestToolsList(unittest.TestCase):
    def test_tools_list_not_empty(self):
        self.assertGreater(len(agent.TOOLS), 0)

    def test_all_tools_have_required_fields(self):
        for tool in agent.TOOLS:
            self.assertIsInstance(tool.name, str)
            self.assertIsInstance(tool.description, str)
            self.assertIsInstance(tool.inputSchema, dict)
            self.assertIn("type", tool.inputSchema)

    def test_um_domain_tools_present(self):
        names = [t.name for t in agent.TOOLS]
        for expected in [
            "search_coverage_policy", "search_step_therapy", "check_pa_required",
            "get_turnaround_time", "search_continuity_of_care", "search_regulatory_guidance",
            "um_answer_question",
        ]:
            self.assertIn(expected, names, f"Missing UM tool: {expected}")

    def test_lightning_tools_present(self):
        names = [t.name for t in agent.TOOLS]
        for expected in [
            "lightning_list_episodes", "lightning_get_episode", "lightning_assign_reward",
            "lightning_list_rewards", "lightning_build_dataset", "lightning_list_datasets",
            "lightning_start_training", "lightning_get_training_status",
            "lightning_list_training_runs", "lightning_promote_deployment",
            "lightning_get_active_deployment", "lightning_list_deployments",
            "lightning_rollback_deployment", "lightning_deactivate_deployment",
            "lightning_get_stats",
        ]:
            self.assertIn(expected, names, f"Missing Lightning tool: {expected}")

    def test_memory_tools_present(self):
        names = [t.name for t in agent.TOOLS]
        self.assertIn("store_memory", names)
        self.assertIn("recall_memory", names)

    def test_no_duplicate_tool_names(self):
        names = [t.name for t in agent.TOOLS]
        self.assertEqual(len(names), len(set(names)), "Duplicate tool names found")


# ===========================================================================
#  Tests – execute_tool (Lightning unavailable path)
# ===========================================================================

class TestExecuteToolLightningUnavailable(unittest.TestCase):
    """Test Lightning tools return errors when Lightning is not available."""

    def _call(self, tool_name, args=None):
        return _run(agent.execute_tool(tool_name, args or {}))

    def test_lightning_list_episodes_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_list_episodes")
            self.assertTrue(result.isError)
            self.assertIn("not available", result.content[0]["text"])

    def test_lightning_get_episode_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_get_episode", {"episode_id": "ep-1"})
            self.assertTrue(result.isError)

    def test_lightning_assign_reward_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_assign_reward", {"episode_id": "ep-1", "reward_value": 0.5})
            self.assertTrue(result.isError)

    def test_lightning_list_rewards_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_list_rewards")
            self.assertTrue(result.isError)

    def test_lightning_build_dataset_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_build_dataset", {"name": "test"})
            self.assertTrue(result.isError)

    def test_lightning_list_datasets_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_list_datasets")
            self.assertTrue(result.isError)

    def test_lightning_start_training_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_start_training", {"dataset_id": "ds-1"})
            self.assertTrue(result.isError)

    def test_lightning_get_training_status_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_get_training_status", {"training_run_id": "tr-1"})
            self.assertTrue(result.isError)

    def test_lightning_list_training_runs_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_list_training_runs")
            self.assertTrue(result.isError)

    def test_lightning_promote_deployment_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_promote_deployment", {"training_run_id": "tr-1"})
            self.assertTrue(result.isError)

    def test_lightning_get_active_deployment_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_get_active_deployment")
            self.assertTrue(result.isError)

    def test_lightning_list_deployments_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_list_deployments")
            self.assertTrue(result.isError)

    def test_lightning_rollback_deployment_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_rollback_deployment")
            self.assertTrue(result.isError)

    def test_lightning_deactivate_deployment_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_deactivate_deployment")
            self.assertTrue(result.isError)

    def test_lightning_get_stats_unavailable(self):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            result = self._call("lightning_get_stats")
            self.assertTrue(result.isError)


# ===========================================================================
#  Tests – execute_tool (UM domain tools)
# ===========================================================================

class TestExecuteToolUMDomain(unittest.TestCase):
    """Test UM domain tools dispatch correctly."""

    def _call(self, tool_name, args=None):
        return _run(agent.execute_tool(tool_name, args or {}))

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_search_coverage_policy(self, mock_search):
        mock_search.return_value = [{"source": "coverage_policy", "content": "PA required for biologics", "score": 0.9}]
        result = self._call("search_coverage_policy", {"service": "dupilumab", "condition": "asthma"})
        self.assertFalse(result.isError)
        data = json.loads(result.content[0]["text"])
        self.assertIn("results", data)

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_search_step_therapy(self, mock_search):
        mock_search.return_value = [{"source": "step_therapy", "content": "Fail first required", "score": 0.8}]
        result = self._call("search_step_therapy", {"drug_or_service": "biologic"})
        self.assertFalse(result.isError)

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_check_pa_required(self, mock_search):
        mock_search.return_value = [{"source": "coverage_policy", "content": "PA required", "score": 0.9}]
        result = self._call("check_pa_required", {"service": "MRI", "urgency": "urgent"})
        self.assertFalse(result.isError)
        data = json.loads(result.content[0]["text"])
        self.assertIn("pa_check", data)

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_get_turnaround_time(self, mock_search):
        mock_search.return_value = [{"source": "operations", "content": "72 hours standard", "score": 0.85}]
        result = self._call("get_turnaround_time", {"request_type": "prior_auth", "urgency": "standard"})
        self.assertFalse(result.isError)
        data = json.loads(result.content[0]["text"])
        self.assertIn("turnaround", data)

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_search_continuity_of_care(self, mock_search):
        mock_search.return_value = [{"source": "regulatory", "content": "90-day transition period", "score": 0.8}]
        result = self._call("search_continuity_of_care", {"provider_network_status": "out-of-network"})
        self.assertFalse(result.isError)

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_search_regulatory_guidance(self, mock_search):
        mock_search.return_value = [{"source": "regulatory", "content": "CMS guidance", "score": 0.7}]
        result = self._call("search_regulatory_guidance", {"topic": "urgent PA", "regulation_type": "cms"})
        self.assertFalse(result.isError)

    @patch.object(agent, "decompose_and_retrieve", new_callable=AsyncMock)
    @patch.object(agent, "_synthesize_response")
    def test_um_answer_question(self, mock_synth, mock_decompose):
        mock_decompose.return_value = {
            "sub_questions": [{"question": "PA?", "source_type": "coverage_policy"}],
            "evidence": {"PA?": {"source_type": "coverage_policy", "results": [], "result_count": 0}},
            "conflicts": [],
            "total_sources": 0,
        }
        mock_synth.return_value = "PA is required."
        result = self._call("um_answer_question", {"query": "Is PA required for this biologic?"})
        self.assertFalse(result.isError)
        data = json.loads(result.content[0]["text"])
        self.assertEqual(data["answer"], "PA is required.")

    def test_um_answer_question_no_query(self):
        result = self._call("um_answer_question", {})
        self.assertTrue(result.isError)

    def test_unknown_tool(self):
        result = self._call("nonexistent_tool")
        self.assertTrue(result.isError)
        self.assertIn("Unknown tool", result.content[0]["text"])


# ===========================================================================
#  Tests – Memory Tools
# ===========================================================================

class TestMemoryTools(unittest.TestCase):
    def _call(self, tool_name, args=None):
        return _run(agent.execute_tool(tool_name, args or {}))

    def test_store_memory_no_provider(self):
        with patch.object(agent, "short_term_memory", None):
            result = self._call("store_memory", {"content": "test", "session_id": "s1"})
            self.assertTrue(result.isError)
            self.assertIn("not configured", result.content[0]["text"])

    def test_recall_memory_no_provider(self):
        with patch.object(agent, "short_term_memory", None):
            result = self._call("recall_memory", {"query": "test", "session_id": "s1"})
            self.assertTrue(result.isError)
            self.assertIn("not configured", result.content[0]["text"])


# ===========================================================================
#  Tests – FastAPI Endpoints (using TestClient)
# ===========================================================================

class TestHealthEndpoint(unittest.TestCase):
    def test_health_returns_healthy(self):
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)


class TestRootEndpoint(unittest.TestCase):
    def test_root_returns_info(self):
        from fastapi.testclient import TestClient
        client = TestClient(agent.app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "UM Agent MCP Server")
        self.assertEqual(data["domain"], "utilization_management")
        self.assertIn("endpoints", data)


class TestMCPMessageEndpoint(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(agent.app)

    def test_initialize(self):
        response = self.client.post("/runtime/webhooks/mcp/message", json={
            "jsonrpc": "2.0",
            "id": "test-1",
            "method": "initialize",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["jsonrpc"], "2.0")
        self.assertEqual(data["id"], "test-1")
        result = data["result"]
        self.assertEqual(result["protocolVersion"], "2024-11-05")
        self.assertEqual(result["serverInfo"]["name"], "um-agent")

    def test_tools_list(self):
        response = self.client.post("/runtime/webhooks/mcp/message", json={
            "jsonrpc": "2.0",
            "id": "test-2",
            "method": "tools/list",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        tools = data["result"]["tools"]
        self.assertGreater(len(tools), 0)
        names = [t["name"] for t in tools]
        self.assertIn("search_coverage_policy", names)
        self.assertIn("lightning_list_episodes", names)

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_tools_call_coverage_policy(self, mock_search):
        mock_search.return_value = [{"source": "policy", "content": "PA required", "score": 0.9}]
        response = self.client.post("/runtime/webhooks/mcp/message", json={
            "jsonrpc": "2.0",
            "id": "test-3",
            "method": "tools/call",
            "params": {
                "name": "search_coverage_policy",
                "arguments": {"service": "MRI"},
            },
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("result", data)
        self.assertFalse(data["result"]["isError"])

    def test_invalid_jsonrpc_version(self):
        response = self.client.post("/runtime/webhooks/mcp/message", json={
            "jsonrpc": "1.0",
            "id": "test-4",
            "method": "initialize",
        })
        self.assertEqual(response.status_code, 400)

    def test_unknown_method(self):
        response = self.client.post("/runtime/webhooks/mcp/message", json={
            "jsonrpc": "2.0",
            "id": "test-5",
            "method": "resources/list",
        })
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
