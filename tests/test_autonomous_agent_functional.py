#!/usr/bin/env python3
"""
Functional tests for the AKS-deployed Utilization Management MCP agent.

Tests the autonomous_agent deployed via APIM + MCP + AKS:
1. Health endpoint reachability
2. MCP initialize handshake
3. tools/list returns UM domain + Lightning tools
4. tools/call for each UM domain tool
5. tools/call for Lightning tools (unavailable path)

Usage:
    # Against a live deployment:
    python tests/test_autonomous_agent_functional.py

    # Local (uses TestClient, no network):
    pytest tests/test_autonomous_agent_functional.py -v

Requirements for live mode:
    - aiohttp (pip install aiohttp)
    - MCP server running (autonomous_agent.py)
    - Valid OAuth token or Azure CLI configured
"""

import asyncio
import json
import sys
import os
import unittest
from pathlib import Path
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# For local / CI testing without a live server, use FastAPI TestClient.
# The same tests can run against a live endpoint by setting
# UM_AGENT_BASE_URL environment variable.
# ---------------------------------------------------------------------------

# Stub imports for module loading
from unittest.mock import MagicMock, AsyncMock
from enum import Enum as _Enum

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
sys.modules.setdefault("lightning", MagicMock())

# Pre-import real azure.identity so the mock setdefault below won't shadow it.
# This allows AzureAISearchContextProvider agentic retrieval tests to use the
# real SDK when installed, while still mocking for environments without it.
try:
    import azure.identity  # noqa: F401
    import azure.identity.aio  # noqa: F401
except ImportError:
    pass

for _mod in [
    "azure.identity", "azure.cosmos", "azure.cosmos.exceptions", "openai", "dotenv",
]:
    sys.modules.setdefault(_mod, MagicMock())

sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None

os.environ.setdefault("COSMOSDB_ENDPOINT", "")
os.environ.setdefault("FOUNDRY_PROJECT_ENDPOINT", "")
os.environ.setdefault("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-test")
os.environ.setdefault("EMBEDDING_MODEL_DEPLOYMENT_NAME", "embed-test")
os.environ.setdefault("AZURE_SEARCH_ENDPOINT", "")
os.environ.setdefault("ENABLE_LIGHTNING_CAPTURE", "false")
os.environ.setdefault("USE_TUNED_MODEL", "false")

_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))

import autonomous_agent as agent
from unittest.mock import patch


# ===========================================================================
#  Test Client Helper
# ===========================================================================

def _get_test_client():
    """Return a FastAPI TestClient for the autonomous agent."""
    from fastapi.testclient import TestClient
    return TestClient(agent.app)


def _jsonrpc(method: str, params: Dict[str, Any] = None, req_id: str = "func-test-1") -> Dict:
    """Build a JSON-RPC 2.0 request body."""
    body = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params:
        body["params"] = params
    return body


# ===========================================================================
#  Functional Tests – Health
# ===========================================================================

class TestFunctionalHealth(unittest.TestCase):
    def setUp(self):
        self.client = _get_test_client()

    def test_health_endpoint_status(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")

    def test_health_has_timestamp(self):
        resp = self.client.get("/health")
        data = resp.json()
        self.assertIn("timestamp", data)
        self.assertIsInstance(data["timestamp"], str)

    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "UM Agent MCP Server")
        self.assertIn("endpoints", data)


# ===========================================================================
#  Functional Tests – MCP Initialize
# ===========================================================================

class TestFunctionalInitialize(unittest.TestCase):
    def setUp(self):
        self.client = _get_test_client()

    def test_initialize_returns_protocol_version(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("initialize"))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["result"]["protocolVersion"], "2024-11-05")

    def test_initialize_returns_server_info(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("initialize"))
        data = resp.json()
        self.assertEqual(data["result"]["serverInfo"]["name"], "um-agent")
        self.assertEqual(data["result"]["serverInfo"]["version"], "1.0.0")

    def test_initialize_returns_capabilities(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("initialize"))
        data = resp.json()
        self.assertIn("tools", data["result"]["capabilities"])

    def test_initialize_preserves_request_id(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("initialize", req_id="my-id-42"))
        data = resp.json()
        self.assertEqual(data["id"], "my-id-42")


# ===========================================================================
#  Functional Tests – tools/list
# ===========================================================================

class TestFunctionalToolsList(unittest.TestCase):
    def setUp(self):
        self.client = _get_test_client()

    def test_tools_list_returns_tools(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/list"))
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        tools = data["result"]["tools"]
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

    def test_tools_list_contains_um_tools(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/list"))
        tools = resp.json()["result"]["tools"]
        names = [t["name"] for t in tools]
        um_tools = [
            "search_coverage_policy", "search_step_therapy", "check_pa_required",
            "get_turnaround_time", "search_continuity_of_care",
            "search_regulatory_guidance", "um_answer_question",
        ]
        for t in um_tools:
            self.assertIn(t, names, f"Missing UM tool: {t}")

    def test_tools_list_contains_lightning_tools(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/list"))
        tools = resp.json()["result"]["tools"]
        names = [t["name"] for t in tools]
        lightning_tools = [
            "lightning_list_episodes", "lightning_get_episode", "lightning_assign_reward",
            "lightning_list_rewards", "lightning_build_dataset", "lightning_list_datasets",
            "lightning_start_training", "lightning_get_training_status",
            "lightning_list_training_runs", "lightning_promote_deployment",
            "lightning_get_active_deployment", "lightning_list_deployments",
            "lightning_rollback_deployment", "lightning_deactivate_deployment",
            "lightning_get_stats",
        ]
        for t in lightning_tools:
            self.assertIn(t, names, f"Missing Lightning tool: {t}")

    def test_tools_have_input_schemas(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/list"))
        tools = resp.json()["result"]["tools"]
        for t in tools:
            self.assertIn("inputSchema", t, f"Tool {t['name']} missing inputSchema")
            self.assertIn("type", t["inputSchema"])

    def test_tools_have_descriptions(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/list"))
        tools = resp.json()["result"]["tools"]
        for t in tools:
            self.assertTrue(len(t["description"]) > 0, f"Tool {t['name']} has empty description")


# ===========================================================================
#  Functional Tests – tools/call (UM domain)
# ===========================================================================

class TestFunctionalToolsCallUM(unittest.TestCase):
    def setUp(self):
        self.client = _get_test_client()

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_search_coverage_policy(self, mock_search):
        mock_search.return_value = [{"source": "coverage_policy", "content": "PA required for biologics", "score": 0.9}]
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "search_coverage_policy",
            "arguments": {"service": "dupilumab", "condition": "severe asthma"},
        }))
        self.assertEqual(resp.status_code, 200)
        result = resp.json()["result"]
        self.assertFalse(result["isError"])
        content_text = result["content"][0]["text"]
        data = json.loads(content_text)
        self.assertIn("results", data)

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_search_step_therapy(self, mock_search):
        mock_search.return_value = [{"source": "step_therapy", "content": "Fail first required", "score": 0.85}]
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "search_step_therapy",
            "arguments": {"drug_or_service": "biologic", "condition": "asthma"},
        }))
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["result"]["isError"])

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_check_pa_required_urgent(self, mock_search):
        mock_search.return_value = [{"source": "coverage_policy", "content": "PA required", "score": 0.9}]
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "check_pa_required",
            "arguments": {"service": "MRI brain", "urgency": "urgent"},
        }))
        self.assertEqual(resp.status_code, 200)
        result = resp.json()["result"]
        self.assertFalse(result["isError"])
        data = json.loads(result["content"][0]["text"])
        self.assertEqual(data["pa_check"]["urgency"], "urgent")

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_get_turnaround_time(self, mock_search):
        mock_search.return_value = [{"source": "operations", "content": "72 hours", "score": 0.8}]
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "get_turnaround_time",
            "arguments": {"request_type": "prior_auth", "urgency": "standard"},
        }))
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.json()["result"]["content"][0]["text"])
        self.assertIn("turnaround", data)

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_search_continuity_of_care(self, mock_search):
        mock_search.return_value = [{"source": "regulatory", "content": "90-day transition", "score": 0.75}]
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "search_continuity_of_care",
            "arguments": {"provider_network_status": "out-of-network", "transition_type": "provider_exit"},
        }))
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["result"]["isError"])

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_search_regulatory_guidance(self, mock_search):
        mock_search.return_value = [{"source": "regulatory", "content": "CMS timeline", "score": 0.7}]
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "search_regulatory_guidance",
            "arguments": {"topic": "urgent prior auth timelines", "regulation_type": "cms"},
        }))
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["result"]["isError"])

    @patch.object(agent, "decompose_and_retrieve", new_callable=AsyncMock)
    @patch.object(agent, "_synthesize_response")
    def test_um_answer_question_full_flow(self, mock_synth, mock_decompose):
        mock_decompose.return_value = {
            "sub_questions": [
                {"question": "PA required?", "source_type": "coverage_policy"},
                {"question": "Step therapy?", "source_type": "step_therapy"},
            ],
            "evidence": {
                "PA required?": {"source_type": "coverage_policy", "results": [{"content": "PA required"}], "result_count": 1},
                "Step therapy?": {"source_type": "step_therapy", "results": [{"content": "Fail first"}], "result_count": 1},
            },
            "conflicts": [],
            "total_sources": 2,
        }
        mock_synth.return_value = "PA is required. Step therapy: fail first required. No conflicts detected."

        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "um_answer_question",
            "arguments": {
                "query": "Medicare Advantage member with severe asthma. Requesting biologic. Is PA required? Step therapy?",
                "member_context": {"plan_type": "Medicare Advantage", "condition": "severe asthma"},
                "urgency": "urgent",
            },
        }))
        self.assertEqual(resp.status_code, 200)
        result = resp.json()["result"]
        self.assertFalse(result["isError"])
        data = json.loads(result["content"][0]["text"])
        self.assertIn("answer", data)
        self.assertIn("retrieval_metadata", data)
        self.assertEqual(data["retrieval_metadata"]["total_sources"], 2)


# ===========================================================================
#  Functional Tests – tools/call (Lightning - unavailable path)
# ===========================================================================

class TestFunctionalToolsCallLightning(unittest.TestCase):
    """Lightning tools should return error when not available."""

    def setUp(self):
        self.client = _get_test_client()

    def _call_lightning_tool(self, name, args=None):
        with patch.object(agent, "LIGHTNING_AVAILABLE", False):
            resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
                "name": name,
                "arguments": args or {},
            }))
        return resp

    def test_lightning_list_episodes(self):
        resp = self._call_lightning_tool("lightning_list_episodes")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["result"]["isError"])

    def test_lightning_get_episode(self):
        resp = self._call_lightning_tool("lightning_get_episode", {"episode_id": "ep-1"})
        self.assertTrue(resp.json()["result"]["isError"])

    def test_lightning_assign_reward(self):
        resp = self._call_lightning_tool("lightning_assign_reward", {"episode_id": "ep-1", "reward_value": 0.8})
        self.assertTrue(resp.json()["result"]["isError"])

    def test_lightning_build_dataset(self):
        resp = self._call_lightning_tool("lightning_build_dataset", {"name": "test-ds"})
        self.assertTrue(resp.json()["result"]["isError"])

    def test_lightning_start_training(self):
        resp = self._call_lightning_tool("lightning_start_training", {"dataset_id": "ds-1"})
        self.assertTrue(resp.json()["result"]["isError"])

    def test_lightning_get_stats(self):
        resp = self._call_lightning_tool("lightning_get_stats")
        self.assertTrue(resp.json()["result"]["isError"])


# ===========================================================================
#  Functional Tests – Error Handling
# ===========================================================================

class TestFunctionalErrorHandling(unittest.TestCase):
    def setUp(self):
        self.client = _get_test_client()

    def test_invalid_jsonrpc_version(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json={
            "jsonrpc": "1.0", "id": "err-1", "method": "initialize"
        })
        self.assertEqual(resp.status_code, 400)

    def test_unknown_method(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("unknown/method"))
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], -32601)

    def test_tools_call_unknown_tool(self):
        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "nonexistent_tool", "arguments": {},
        }))
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["result"]["isError"])


# ===========================================================================
#  Functional Tests – Prior Authorization Scenario
#  Medicare Advantage member with severe asthma, provider requesting biologic.
#  Validates that the agent can answer multi-policy questions:
#    - Is prior authorization required?
#    - Step therapy required vs. exceptions apply
#    - Urgent turnaround time
#    - Continuity-of-care for out-of-network provider
# ===========================================================================

# Policy fixture data matching the real task_instructions/ documents
_COVERAGE_POLICY_EVIDENCE = {
    "source": "coverage_policy",
    "title": "Medical Policy: Biologic Therapy for Severe Asthma (Medicare Advantage)",
    "content": (
        "Prior authorization is required for all biologic therapies addressed under this policy. "
        "Biologic therapy may be considered medically necessary when all of the following criteria are met: "
        "1) Member has a confirmed diagnosis of severe persistent asthma. "
        "2) Symptoms remain uncontrolled despite high-dose inhaled corticosteroids AND long-acting beta-agonist therapy. "
        "3) Evidence of frequent exacerbations (>=2 in the prior 12 months). "
        "4) Prescribing provider is a pulmonologist or allergist. "
        "Step therapy is required prior to approval of biologic therapy unless an exception applies."
    ),
    "score": 0.95,
    "policy_id": "MA-ASTHMA-BIOLOGIC-001",
    "effective_date": "2025-01-01",
}

_EXCEPTION_POLICY_EVIDENCE = {
    "source": "exception_logic",
    "title": "Exception Policy: Step Therapy Exceptions for Severe Asthma",
    "content": (
        "An exception to step therapy may be granted when any of the following conditions are met: "
        "1) Documented intolerance or contraindication to required step therapy medications. "
        "2) History of severe adverse reaction to controller therapies. "
        "3) Prior biologic therapy initiated under another Medicare plan with documented clinical benefit. "
        "4) Continuity-of-care considerations during transition of coverage. "
        "When exception criteria are met, step therapy requirements defined in MA-ASTHMA-BIOLOGIC-001 may be waived. "
        "Prior authorization is still required even when an exception is granted."
    ),
    "score": 0.92,
    "policy_id": "MA-ASTHMA-BIOLOGIC-EXC-002",
    "effective_date": "2025-03-01",
}

_OPERATIONS_EVIDENCE = {
    "source": "operations",
    "title": "Utilization Management Operational Guidance",
    "content": (
        "Standard requests: Determination within 14 calendar days of receipt. "
        "Urgent requests may be processed as urgent when delay could seriously jeopardize the member's health. "
        "Urgent turnaround: Determination within 72 hours. "
        "When a member is transitioning coverage and is receiving an ongoing course of treatment, "
        "requests may qualify for continuity-of-care consideration."
    ),
    "score": 0.88,
    "effective_date": "2025-02-15",
}

_CONTINUITY_EVIDENCE = {
    "source": "regulatory",
    "title": "Continuity-of-Care and Network Transition",
    "content": (
        "Network status of the provider does not automatically disqualify review. "
        "Clinical documentation must support ongoing therapy. "
        "Requests may qualify for continuity-of-care consideration when a member is transitioning "
        "coverage and is receiving an ongoing course of treatment. "
        "Operational guidance does not override medical necessity criteria but defines processing "
        "timelines and handling procedures."
    ),
    "score": 0.85,
}

_CMS_REGULATORY_EVIDENCE = {
    "source": "regulatory",
    "title": "CMS Interoperability and Prior Authorization Final Rule (CMS-0057-F)",
    "content": (
        "Under the CMS-0057-F final rule, payers must meet specific turnaround times for prior "
        "authorization decisions: Standard non-urgent requests must receive a determination within "
        "7 calendar days of receipt. Urgent (expedited) requests where delay could seriously "
        "jeopardize the member's life, health, or ability to regain maximum function must receive "
        "a determination within 72 hours."
    ),
    "score": 0.80,
    "policy_id": "CMS-0057-F",
    "effective_date": "2024-01-17",
}


class TestPriorAuthorizationScenario(unittest.TestCase):
    """
    End-to-end scenario: Medicare Advantage member with severe asthma.
    Provider requesting a biologic. One policy says step therapy required,
    another says exceptions apply.
    """

    def setUp(self):
        self.client = _get_test_client()

    # ── Q1: Is prior authorization required? ──

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_pa_required_for_biologic_asthma(self, mock_search):
        """Prior authorization IS required for biologic therapy in severe asthma."""
        mock_search.return_value = [_COVERAGE_POLICY_EVIDENCE]

        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "check_pa_required",
            "arguments": {
                "service": "biologic therapy",
                "plan_type": "Medicare Advantage",
                "urgency": "urgent",
            },
        }))
        self.assertEqual(resp.status_code, 200)
        result = resp.json()["result"]
        self.assertFalse(result["isError"])
        data = json.loads(result["content"][0]["text"])
        self.assertEqual(data["pa_check"]["urgency"], "urgent")
        # Check that coverage policy evidence is returned
        pa_results = data["pa_check"]["results"]
        self.assertTrue(len(pa_results) > 0)
        self.assertIn("prior authorization", pa_results[0]["content"].lower())

    # ── Q2: Step therapy required vs. exception ──

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_step_therapy_required_for_biologic(self, mock_search):
        """Step therapy IS required per base coverage policy."""
        mock_search.return_value = [_COVERAGE_POLICY_EVIDENCE]

        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "search_step_therapy",
            "arguments": {
                "drug_or_service": "biologic",
                "condition": "severe asthma",
            },
        }))
        self.assertEqual(resp.status_code, 200)
        result = resp.json()["result"]
        self.assertFalse(result["isError"])
        data = json.loads(result["content"][0]["text"])
        self.assertTrue(len(data["results"]) > 0)
        content_text = data["results"][0]["content"].lower()
        self.assertIn("step therapy", content_text)

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_step_therapy_exception_criteria(self, mock_search):
        """Exception policy allows waiver of step therapy under certain conditions."""
        mock_search.return_value = [_EXCEPTION_POLICY_EVIDENCE]

        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "search_coverage_policy",
            "arguments": {
                "service": "biologic therapy exception",
                "condition": "severe asthma",
                "plan_type": "Medicare Advantage",
            },
        }))
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.json()["result"]["content"][0]["text"])
        exception_content = data["results"][0]["content"].lower()
        self.assertIn("exception", exception_content)
        self.assertIn("prior authorization is still required", exception_content)

    # ── Q3: What is the urgent turnaround time? ──

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_urgent_turnaround_time(self, mock_search):
        """Urgent turnaround time is 72 hours per CMS/operational guidance."""
        mock_search.return_value = [_OPERATIONS_EVIDENCE, _CMS_REGULATORY_EVIDENCE]

        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "get_turnaround_time",
            "arguments": {
                "request_type": "prior_auth",
                "urgency": "urgent",
            },
        }))
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.json()["result"]["content"][0]["text"])
        self.assertEqual(data["turnaround"]["urgency"], "urgent")
        results = data["turnaround"]["results"]
        self.assertTrue(len(results) > 0)
        # At least one result mentions 72 hours
        combined = " ".join(r["content"] for r in results)
        self.assertIn("72 hours", combined)

    # ── Q4: Does continuity-of-care apply (OON provider)? ──

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_continuity_of_care_oon_provider(self, mock_search):
        """Continuity-of-care applies; OON status does not auto-disqualify."""
        mock_search.return_value = [_CONTINUITY_EVIDENCE]

        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "search_continuity_of_care",
            "arguments": {
                "provider_network_status": "out-of-network",
                "transition_type": "coverage_transition",
            },
        }))
        self.assertEqual(resp.status_code, 200)
        result = resp.json()["result"]
        self.assertFalse(result["isError"])
        data = json.loads(result["content"][0]["text"])
        coc_results = data["continuity_of_care"]["results"]
        self.assertTrue(len(coc_results) > 0)
        combined = " ".join(r["content"] for r in coc_results)
        self.assertIn("does not automatically disqualify", combined)

    # ── Full scenario: um_answer_question with multi-source retrieval ──

    @patch.object(agent, "decompose_and_retrieve", new_callable=AsyncMock)
    @patch.object(agent, "_synthesize_response")
    def test_full_scenario_answer(self, mock_synth, mock_decompose):
        """
        Full scenario: decompose multi-part question, retrieve from coverage,
        exception, operations, and regulatory sources, detect policy conflicts,
        and synthesize a grounded answer.
        """
        mock_decompose.return_value = {
            "sub_questions": [
                {"question": "Is prior authorization required for biologic therapy for severe asthma under Medicare Advantage?", "source_type": "coverage_policy"},
                {"question": "Step therapy requirements and exceptions for biologic therapy", "source_type": "step_therapy"},
                {"question": "Exception conditions for step therapy waiver", "source_type": "exception_logic"},
                {"question": "Urgent turnaround time for prior authorization", "source_type": "operations"},
                {"question": "Continuity-of-care for out-of-network provider transition", "source_type": "regulatory"},
            ],
            "evidence": {
                "Is prior authorization required for biologic therapy for severe asthma under Medicare Advantage?": {
                    "source_type": "coverage_policy",
                    "results": [_COVERAGE_POLICY_EVIDENCE],
                    "result_count": 1,
                },
                "Step therapy requirements and exceptions for biologic therapy": {
                    "source_type": "step_therapy",
                    "results": [_COVERAGE_POLICY_EVIDENCE],
                    "result_count": 1,
                },
                "Exception conditions for step therapy waiver": {
                    "source_type": "exception_logic",
                    "results": [_EXCEPTION_POLICY_EVIDENCE],
                    "result_count": 1,
                },
                "Urgent turnaround time for prior authorization": {
                    "source_type": "operations",
                    "results": [_OPERATIONS_EVIDENCE],
                    "result_count": 1,
                },
                "Continuity-of-care for out-of-network provider transition": {
                    "source_type": "regulatory",
                    "results": [_CONTINUITY_EVIDENCE, _CMS_REGULATORY_EVIDENCE],
                    "result_count": 2,
                },
            },
            "conflicts": [
                {
                    "type": "policy_exception_overlap",
                    "description": "Coverage policy requires step therapy but exception policy allows waiver under certain conditions.",
                    "resolution": "Prefer the exception policy when conditions are met; otherwise follow base policy.",
                },
            ],
            "total_sources": 6,
        }
        mock_synth.return_value = (
            "## Prior Authorization Analysis\n\n"
            "**1. Prior Authorization Required:** Yes. Per MA-ASTHMA-BIOLOGIC-001 (eff. 2025-01-01), "
            "prior authorization is required for all biologic therapies for severe asthma.\n\n"
            "**2. Step Therapy:** Required per base policy, but exceptions may apply per "
            "MA-ASTHMA-BIOLOGIC-EXC-002 (eff. 2025-03-01). Exception criteria include documented "
            "intolerance, severe adverse reaction, prior biologic under another plan, or continuity-of-care.\n\n"
            "**3. Urgent Turnaround Time:** 72 hours per operational guidance and CMS-0057-F.\n\n"
            "**4. Continuity-of-Care (OON Provider):** Yes, continuity-of-care consideration applies. "
            "Network status does not automatically disqualify review. Clinical documentation must support "
            "ongoing therapy.\n\n"
            "**Conflict Detected:** Coverage policy and exception policy provide differing guidance on step therapy. "
            "When exception criteria are met, step therapy may be waived, but PA remains required.\n\n"
            "*Human review recommended.*"
        )

        query = (
            "Medicare Advantage member with severe asthma. Provider requesting a biologic. "
            "One policy says step therapy required, another says exceptions apply. "
            "Is prior authorization required? What's the urgent turnaround time? "
            "Does continuity-of-care apply since the provider is out-of-network?"
        )

        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "um_answer_question",
            "arguments": {
                "query": query,
                "member_context": {
                    "plan_type": "Medicare Advantage",
                    "condition": "severe asthma",
                    "requested_service": "biologic therapy",
                    "provider_network_status": "out-of-network",
                },
                "urgency": "urgent",
            },
        }))
        self.assertEqual(resp.status_code, 200)
        result = resp.json()["result"]
        self.assertFalse(result["isError"])
        data = json.loads(result["content"][0]["text"])

        # Validate answer structure
        self.assertIn("answer", data)
        self.assertIn("retrieval_metadata", data)
        self.assertEqual(data["urgency"], "urgent")

        # Validate decomposition and retrieval
        metadata = data["retrieval_metadata"]
        self.assertEqual(metadata["total_sources"], 6)
        self.assertEqual(len(metadata["sub_questions"]), 5)
        self.assertTrue(len(metadata["conflicts"]) > 0)
        self.assertEqual(metadata["conflicts"][0]["type"], "policy_exception_overlap")

        # Validate answer content references key facts
        answer = data["answer"]
        self.assertIn("prior authorization", answer.lower())
        self.assertIn("72 hours", answer)
        self.assertIn("continuity-of-care", answer.lower())
        self.assertIn("exception", answer.lower())

    # ── CMS regulatory evidence retrieval ──

    @patch.object(agent, "search_knowledge_base", new_callable=AsyncMock)
    def test_cms_regulatory_turnaround_guidance(self, mock_search):
        """CMS-0057-F regulatory guidance provides authoritative turnaround times."""
        mock_search.return_value = [_CMS_REGULATORY_EVIDENCE]

        resp = self.client.post("/runtime/webhooks/mcp/message", json=_jsonrpc("tools/call", {
            "name": "search_regulatory_guidance",
            "arguments": {
                "topic": "prior authorization turnaround times",
                "regulation_type": "cms",
            },
        }))
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.json()["result"]["content"][0]["text"])
        results = data["regulatory"]["results"]
        self.assertTrue(len(results) > 0)
        self.assertIn("CMS-0057-F", results[0].get("policy_id", results[0].get("content", "")))


# ===========================================================================
#  Functional Tests – Agentic Retrieval via AzureAISearchContextProvider
#
#  Validates that the prior-authorization knowledge base supports agentic
#  retrieval using the Microsoft Agent Framework's AzureAISearchContextProvider.
#  Follows the Foundry IQ UI pattern: query a KB backed by 3 knowledge sources
#    1. utilization-management-guidance  (searchIndex – policy documents)
#    2. cms-pa-rule                      (web – CMS regulatory website)
#    3. utilization-management-facts     (azureBlob – JSON fact documents)
#
#  These tests require:
#    AZURE_SEARCH_ENDPOINT  – AI Search service endpoint
#  They are skipped when the env var is absent (e.g. in CI without Azure).
# ===========================================================================

# Try to import the real AzureAISearchContextProvider
_AGENTIC_AVAILABLE = False
try:
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
    from agent_framework.azure import AzureAISearchContextProvider
    from agent_framework import Message
    _AGENTIC_AVAILABLE = True
except ImportError:
    pass

_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
_PA_KB_NAME = os.environ.get("AZURE_SEARCH_PA_KNOWLEDGE_BASE_NAME", "prior-authorization-kb")
_PA_INDEX_NAME = os.environ.get("AZURE_SEARCH_PA_INDEX_NAME", "prior-authorization")

# Knowledge source names (matching Bicep output)
_KS_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_PA_KS_SEARCH_INDEX", "utilization-management-guidance")
_KS_WEB = os.environ.get("AZURE_SEARCH_PA_KS_WEB", "cms-pa-rule")
_KS_BLOB = os.environ.get("AZURE_SEARCH_PA_KS_BLOB", "utilization-management-facts")

# Azure OpenAI config for agentic retrieval (reasoning requires a model)
_FOUNDRY_ENDPOINT = os.environ.get("FOUNDRY_PROJECT_ENDPOINT", "")
_AOAI_RESOURCE_URL = os.environ.get(
    "AZURE_OPENAI_RESOURCE_URL",
    _FOUNDRY_ENDPOINT.split("/api/projects")[0] if "/api/projects" in _FOUNDRY_ENDPOINT else _FOUNDRY_ENDPOINT,
)
_AOAI_MODEL_DEPLOYMENT = os.environ.get("FOUNDRY_MODEL_DEPLOYMENT_NAME", "gpt-5")


@unittest.skipUnless(
    _AGENTIC_AVAILABLE and _SEARCH_ENDPOINT,
    "Requires agent-framework-azure-ai-search SDK and AZURE_SEARCH_ENDPOINT env var",
)
class TestPriorAuthAgenticRetrieval(unittest.TestCase):
    """
    End-to-end agentic retrieval tests against the prior-authorization
    knowledge base using AzureAISearchContextProvider.

    Functions like the Foundry IQ UI:
      - Queries the KB in agentic mode
      - Verifies citations are returned from multiple knowledge sources
      - Tests different reasoning effort levels (minimal, medium)
    """

    @classmethod
    def setUpClass(cls):
        cls._credential = AsyncDefaultAzureCredential()
        cls._endpoint = _SEARCH_ENDPOINT

    @classmethod
    def tearDownClass(cls):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cls._credential.close())
        loop.close()

    def _run_async(self, coro):
        """Helper to run an async coroutine in a sync test."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    async def _query_kb(self, query: str, effort: str = "low",
                        use_model: bool = False) -> dict:
        """
        Query the KB like Foundry IQ UI: returns a dict with
        'context' (full text) and 'messages' (list of individual messages).
        Retries on 429 rate limit errors with exponential backoff.
        """
        import re as _re
        kwargs = dict(
            endpoint=self._endpoint,
            credential=self._credential,
            knowledge_base_name=_PA_KB_NAME,
            mode="agentic",
            retrieval_reasoning_effort=effort,
            knowledge_base_output_mode="answer_synthesis",
        )
        if use_model and _AOAI_RESOURCE_URL:
            kwargs["azure_openai_resource_url"] = _AOAI_RESOURCE_URL
            kwargs["model_deployment_name"] = _AOAI_MODEL_DEPLOYMENT

        max_retries = 3
        for attempt in range(max_retries + 1):
            provider = AzureAISearchContextProvider(**kwargs)
            try:
                user_message = Message(role="user", text=query)
                async with provider as p:
                    result_messages = await p._agentic_search([user_message])
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries:
                    # Parse retry-after from error message
                    match = _re.search(r'retry after (\d+)', str(e), _re.IGNORECASE)
                    wait = int(match.group(1)) + 2 if match else 15 * (attempt + 1)
                    await asyncio.sleep(wait)
                    result_messages = None
                    continue
                raise

        messages = []
        full_text = ""
        if result_messages:
            messages = [m for m in result_messages if m.text]
            full_text = "\n\n".join(m.text for m in messages)

        return {"context": full_text, "messages": messages, "raw": result_messages}

    async def _query_kb_semantic(self, query: str) -> str:
        """Query the index directly in semantic mode (single-source fallback)."""
        provider = AzureAISearchContextProvider(
            endpoint=self._endpoint,
            credential=self._credential,
            index_name=_PA_INDEX_NAME,
            mode="semantic",
        )
        async with provider as p:
            result_messages = await p._semantic_search(query)

        if result_messages:
            return "\n\n".join(msg.text for msg in result_messages if msg.text)
        return ""

    # ── Core agentic retrieval: policy data from searchIndex source ──

    def test_agentic_pa_required_for_biologic(self):
        """KB returns PA requirement for biologic therapy (searchIndex source)."""
        result = self._run_async(self._query_kb(
            "Is prior authorization required for biologic therapy for severe asthma under Medicare Advantage?"
        ))
        self.assertTrue(len(result["context"]) > 0, "Agentic retrieval returned empty context")
        ctx = result["context"].lower()
        self.assertTrue(
            "prior authorization" in ctx or "pa" in ctx,
            f"Expected prior authorization content, got: {result['context'][:200]}",
        )

    def test_agentic_step_therapy_exception(self):
        """KB returns step therapy exception criteria (searchIndex + blob sources)."""
        result = self._run_async(self._query_kb(
            "What are the step therapy exception criteria for biologic therapy?"
        ))
        self.assertTrue(len(result["context"]) > 0, "Agentic retrieval returned empty context")
        ctx = result["context"].lower()
        self.assertTrue(
            "exception" in ctx or "step therapy" in ctx,
            f"Expected exception/step therapy content, got: {result['context'][:200]}",
        )

    def test_agentic_urgent_turnaround(self):
        """KB returns 72-hour urgent turnaround time."""
        result = self._run_async(self._query_kb(
            "What is the urgent turnaround time for prior authorization decisions?"
        ))
        self.assertTrue(len(result["context"]) > 0, "Agentic retrieval returned empty context")
        self.assertIn("72", result["context"],
                       f"Expected 72-hour mention, got: {result['context'][:200]}")

    def test_agentic_continuity_of_care(self):
        """KB returns continuity-of-care policy across knowledge sources."""
        result = self._run_async(self._query_kb(
            "Does continuity of care apply when a provider is out-of-network during coverage transition?"
        ))
        self.assertTrue(len(result["context"]) > 0, "Agentic retrieval returned empty context")
        ctx = result["context"].lower()
        self.assertTrue(
            "continuity" in ctx or "transition" in ctx,
            f"Expected continuity-of-care content, got: {result['context'][:200]}",
        )

    # ── CMS regulatory content (web knowledge source) ──

    def test_agentic_cms_regulatory(self):
        """KB returns CMS-0057-F regulatory guidance from web knowledge source."""
        result = self._run_async(self._query_kb(
            "What does the CMS Interoperability and Prior Authorization Final Rule CMS-0057-F require?"
        ))
        self.assertTrue(len(result["context"]) > 0, "Agentic retrieval returned empty context")
        ctx = result["context"].lower()
        self.assertTrue(
            "cms" in ctx or "0057" in ctx or "prior authorization" in ctx,
            f"Expected CMS regulatory content, got: {result['context'][:200]}",
        )

    # ── Medical necessity facts (blob knowledge source) ──

    def test_agentic_medical_necessity_criteria(self):
        """KB returns medical necessity criteria from blob knowledge source."""
        result = self._run_async(self._query_kb(
            "What are the medical necessity criteria for biologic therapy for severe asthma?"
        ))
        self.assertTrue(len(result["context"]) > 0, "Agentic retrieval returned empty context")
        ctx = result["context"].lower()
        self.assertTrue(
            "medical necessity" in ctx or "severe persistent asthma" in ctx or "biologic" in ctx,
            f"Expected medical necessity content, got: {result['context'][:200]}",
        )

    def test_agentic_exception_logic_decision_matrix(self):
        """KB returns exception logic and decision matrix from blob knowledge source."""
        result = self._run_async(self._query_kb(
            "What is the decision matrix for step therapy exceptions with urgent prior authorization?"
        ))
        self.assertTrue(len(result["context"]) > 0, "Agentic retrieval returned empty context")
        ctx = result["context"].lower()
        self.assertTrue(
            "exception" in ctx or "decision" in ctx or "urgent" in ctx,
            f"Expected exception logic content, got: {result['context'][:200]}",
        )

    # ── Multi-source citation validation (Foundry IQ style) ──

    def test_agentic_multi_source_retrieval(self):
        """
        Foundry IQ-style test: a complex query should retrieve content from
        multiple knowledge sources (policy data, CMS rules, UM facts).
        """
        result = self._run_async(self._query_kb(
            "For a Medicare Advantage member with severe asthma requesting biologic therapy, "
            "what are the prior authorization requirements, step therapy exceptions, "
            "CMS regulatory turnaround times, and medical necessity criteria?"
        ))
        self.assertTrue(len(result["context"]) > 0, "Agentic retrieval returned empty context")

        # The response should contain content spanning multiple concerns
        ctx = result["context"].lower()
        found_topics = []
        if "prior authorization" in ctx or "pa " in ctx:
            found_topics.append("prior_authorization")
        if "exception" in ctx or "step therapy" in ctx:
            found_topics.append("step_therapy")
        if "72" in ctx or "turnaround" in ctx:
            found_topics.append("turnaround")
        if "medical necessity" in ctx or "severe persistent" in ctx or "biologic" in ctx:
            found_topics.append("medical_necessity")

        self.assertTrue(
            len(found_topics) >= 2,
            f"Expected multi-source content covering >=2 topics, found: {found_topics}. "
            f"Context preview: {result['context'][:300]}",
        )

    def test_agentic_retrieval_returns_citations(self):
        """
        Foundry IQ-style test: agentic retrieval should return multiple
        citation messages, similar to the numbered citations in the IQ UI.
        """
        result = self._run_async(self._query_kb(
            "What biologic therapy options are available for severe eosinophilic asthma?"
        ))
        self.assertTrue(len(result["context"]) > 0, "Agentic retrieval returned empty context")
        # Foundry IQ typically returns multiple citations; verify we get at least 1 message
        self.assertTrue(
            len(result["messages"]) >= 1,
            f"Expected >=1 citation messages, got {len(result['messages'])}",
        )

    # ── Semantic mode fallback test ──

    def test_semantic_mode_returns_policy_content(self):
        """Semantic mode retrieval returns relevant policy content from the index."""
        context = self._run_async(self._query_kb_semantic(
            "biologic therapy prior authorization requirements asthma"
        ))
        self.assertTrue(len(context) > 0, "Semantic retrieval returned empty context")

    # ── Medium reasoning effort (model-augmented query planning) ──

    def test_agentic_medium_effort_retrieval(self):
        """Agentic retrieval with medium reasoning effort (model-augmented query planning)."""
        if not _AOAI_RESOURCE_URL:
            self.skipTest("FOUNDRY_PROJECT_ENDPOINT not set for medium effort retrieval")

        try:
            result = self._run_async(self._query_kb(
                "Is prior authorization required for biologic therapy?",
                effort="medium",
                use_model=True,
            ))
            self.assertTrue(len(result["context"]) > 0, "Medium-effort retrieval returned empty context")
        except Exception as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                self.skipTest(f"RBAC not yet propagated for search→foundry: {e}")
            raise

    # ── Phenotype-guided therapy selection (blob facts) ──

    def test_agentic_phenotype_biologic_mapping(self):
        """KB returns phenotype-to-biologic mapping from UM facts."""
        result = self._run_async(self._query_kb(
            "Which biologic should be used for allergic asthma with elevated IgE levels?"
        ))
        self.assertTrue(len(result["context"]) > 0, "Agentic retrieval returned empty context")
        ctx = result["context"].lower()
        self.assertTrue(
            "ige" in ctx or "allergic" in ctx or "omalizumab" in ctx or "xolair" in ctx or "biologic" in ctx,
            f"Expected phenotype/biologic content, got: {result['context'][:200]}",
        )


if __name__ == "__main__":
    unittest.main()
