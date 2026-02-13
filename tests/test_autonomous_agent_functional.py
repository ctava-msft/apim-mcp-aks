#!/usr/bin/env python3
"""
Functional tests for the Customer Churn Analysis Agent (CCA-001).

These tests run against a live (or locally running) instance of autonomous_agent.py
via the MCP SSE + message protocol, following the same pattern as the
next_best_action functional tests.

Usage:
    # Against port-forwarded / local server (default port 8001):
    python tests/test_autonomous_agent_functional.py --direct

    # Against APIM gateway (uses mcp_test_config.json):
    python tests/test_autonomous_agent_functional.py

Requirements:
    - aiohttp
    - autonomous_agent.py running (standalone or in AKS)
"""

import asyncio
import json
import aiohttp
import sys
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

# Configuration file (shared with other functional tests)
CONFIG_FILE = Path(__file__).parent / "mcp_test_config.json"

# Default direct-mode base URL for the autonomous agent
DEFAULT_DIRECT_BASE_URL = "http://localhost:8001/runtime/webhooks/mcp"


def load_config() -> Dict[str, Any]:
    """Load configuration from mcp_test_config.json"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            print(f"✅ Loaded configuration from {CONFIG_FILE}")
            return config
    else:
        print(f"⚠️  Config file not found: {CONFIG_FILE} – using defaults")
        return {}


class MCPClient:
    """MCP Client that maintains SSE session"""

    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.session = None
        self.sse_response = None
        self.session_message_url = None

    async def __aenter__(self):
        headers = {}
        if self.auth_token and not self.auth_token.startswith("direct-mode"):
            headers["Authorization"] = f"Bearer {self.auth_token}"
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.sse_response and not self.sse_response.closed:
            self.sse_response.close()
        if self.session:
            await self.session.close()

    async def check_health(self) -> Dict[str, Any]:
        """Hit the /health endpoint"""
        health_url = self.base_url.replace("/runtime/webhooks/mcp", "") + "/health"
        try:
            async with self.session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def establish_sse_session(self) -> bool:
        """Establish SSE connection and extract session URL"""
        try:
            print(f"\n📡 Establishing SSE session to: {self.base_url}/sse")
            self.sse_response = await self.session.get(
                f"{self.base_url}/sse",
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
            print(f"   SSE Response Status: {self.sse_response.status}")

            if self.sse_response.status == 200:
                async for chunk in self.sse_response.content.iter_chunked(1024):
                    if chunk:
                        data = chunk.decode("utf-8", errors="ignore")
                        match = re.search(r"data: (message\?[^\n\r]+)", data)
                        if match:
                            session_path = match.group(1)
                            self.session_message_url = f"{self.base_url}/{session_path}"
                            print(f"✅ Got session URL: {self.session_message_url}")
                            return True
                        break
                print("⚠️  SSE connected but no session URL found")
                return False
            else:
                response_text = await self.sse_response.text()
                print(f"❌ SSE connection failed: {response_text}")
                return False
        except Exception as e:
            print(f"❌ SSE connection error: {e}")
            return False

    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC 2.0 request"""
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": f"test-{method}",
            "method": method,
        }
        if params:
            jsonrpc_request["params"] = params

        message_url = self.session_message_url or f"{self.base_url}/message"
        try:
            async with self.session.post(
                message_url,
                json=jsonrpc_request,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                response_text = await response.text()
                if response.status == 200:
                    return json.loads(response_text)
                return {"error": f"HTTP {response.status}", "body": response_text}
        except asyncio.TimeoutError:
            return {"error": "Request timed out"}
        except Exception as e:
            return {"error": str(e)}

    async def initialize(self) -> Dict[str, Any]:
        return await self.send_request("initialize")

    async def list_tools(self) -> Optional[List[Dict[str, Any]]]:
        result = await self.send_request("tools/list")
        if "error" not in result:
            return result.get("result", {}).get("tools", [])
        return None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return await self.send_request("tools/call", {"name": tool_name, "arguments": arguments})


# ===========================================================================
#  Test runner
# ===========================================================================

async def run_tests():
    """Run all functional tests against a live autonomous agent."""

    print("=" * 70)
    print("🧪 Functional Tests – Customer Churn Analysis Agent (CCA-001)")
    print("=" * 70)

    use_direct = "--direct" in sys.argv
    config = load_config()

    if use_direct:
        direct_cfg = config.get("direct", {})
        base_url = direct_cfg.get("autonomous_base_url", DEFAULT_DIRECT_BASE_URL)
        token = "direct-mode-no-token-needed"
        print(f"\n🔗 Direct Mode URL: {base_url}")
    else:
        apim_cfg = config.get("apim", {})
        base_url = apim_cfg.get("autonomous_base_url", "")
        if not base_url:
            print("❌ No APIM autonomous_base_url configured – use --direct for local testing")
            return False
        token = None  # TODO: token retrieval via APIM OAuth

    all_passed = True
    tests_run = 0
    tests_passed = 0

    async with MCPClient(base_url, token) as client:

        # ---- 1. Health endpoint -------------------------------------------
        print("\n" + "-" * 60)
        print("1️⃣  Health endpoint")
        tests_run += 1
        health = await client.check_health()
        if health.get("status") == "healthy":
            print(f"   ✅ PASSED – status=healthy, timestamp={health.get('timestamp')}")
            tests_passed += 1
        else:
            print(f"   ❌ FAILED – {health}")
            all_passed = False

        # ---- 2. SSE session -----------------------------------------------
        print("\n" + "-" * 60)
        print("2️⃣  SSE session establishment")
        tests_run += 1
        if await client.establish_sse_session():
            print("   ✅ PASSED – SSE session established")
            tests_passed += 1
        else:
            print("   ❌ FAILED – could not establish SSE session")
            all_passed = False
            # Cannot continue without session
            return False

        await asyncio.sleep(1)

        # ---- 3. MCP Initialize --------------------------------------------
        print("\n" + "-" * 60)
        print("3️⃣  MCP initialize")
        tests_run += 1
        init_resp = await client.initialize()
        if "error" not in init_resp:
            result = init_resp.get("result", {})
            server_name = result.get("serverInfo", {}).get("name", "")
            proto = result.get("protocolVersion", "")
            if server_name == "customer-churn-agent" and proto == "2024-11-05":
                print(f"   ✅ PASSED – server={server_name}, protocol={proto}")
                tests_passed += 1
            else:
                print(f"   ❌ FAILED – unexpected result: {result}")
                all_passed = False
        else:
            print(f"   ❌ FAILED – {init_resp}")
            all_passed = False

        # ---- 4. tools/list ------------------------------------------------
        print("\n" + "-" * 60)
        print("4️⃣  tools/list")
        tests_run += 1
        tools = await client.list_tools()
        expected_tools = [
            "get_customer_facts",
            "get_churn_prediction",
            "get_customer_segment",
            "search_similar_churned",
            "recommend_retention_action",
            "create_retention_case",
            "log_analysis_result",
        ]
        if tools is not None:
            tool_names = [t["name"] for t in tools]
            missing = [n for n in expected_tools if n not in tool_names]
            if not missing:
                print(f"   ✅ PASSED – {len(tools)} tools found, all expected tools present")
                for t in tools:
                    print(f"      • {t['name']}: {t.get('description', '')[:60]}")
                tests_passed += 1
            else:
                print(f"   ❌ FAILED – missing tools: {missing}")
                all_passed = False
        else:
            print("   ❌ FAILED – tools/list returned None")
            all_passed = False

        # ---- 5. tools/call – get_customer_facts ---------------------------
        print("\n" + "-" * 60)
        print("5️⃣  tools/call – get_customer_facts")
        tests_run += 1
        resp = await client.call_tool("get_customer_facts", {"customer_id": "CUST-001"})
        tool_result = resp.get("result", {})
        if not tool_result.get("isError", True):
            payload = json.loads(tool_result["content"][0]["text"])
            if payload.get("customer_id") == "CUST-001" and "facts" in payload:
                print(f"   ✅ PASSED – customer_id={payload['customer_id']}, facts retrieved")
                tests_passed += 1
            else:
                print(f"   ❌ FAILED – unexpected payload: {payload}")
                all_passed = False
        else:
            print(f"   ❌ FAILED – tool error: {tool_result}")
            all_passed = False

        # ---- 6. tools/call – get_churn_prediction -------------------------
        print("\n" + "-" * 60)
        print("6️⃣  tools/call – get_churn_prediction")
        tests_run += 1
        resp = await client.call_tool("get_churn_prediction", {"customer_id": "CUST-001"})
        tool_result = resp.get("result", {})
        if not tool_result.get("isError", True):
            payload = json.loads(tool_result["content"][0]["text"])
            if "churn_score" in payload and "drivers" in payload:
                print(f"   ✅ PASSED – churn_score={payload['churn_score']}, risk={payload.get('risk_level')}")
                tests_passed += 1
            else:
                print(f"   ❌ FAILED – unexpected payload")
                all_passed = False
        else:
            print(f"   ❌ FAILED – tool error: {tool_result}")
            all_passed = False

        # ---- 7. tools/call – recommend_retention_action -------------------
        print("\n" + "-" * 60)
        print("7️⃣  tools/call – recommend_retention_action")
        tests_run += 1
        resp = await client.call_tool("recommend_retention_action", {
            "customer_id": "CUST-001",
            "risk_level": "critical",
        })
        tool_result = resp.get("result", {})
        if not tool_result.get("isError", True):
            payload = json.loads(tool_result["content"][0]["text"])
            if payload.get("recommendation", {}).get("priority") == "immediate":
                print(f"   ✅ PASSED – priority=immediate for critical risk")
                tests_passed += 1
            else:
                print(f"   ❌ FAILED – unexpected recommendation: {payload}")
                all_passed = False
        else:
            print(f"   ❌ FAILED – tool error: {tool_result}")
            all_passed = False

        # ---- 8. tools/call – create_retention_case ------------------------
        print("\n" + "-" * 60)
        print("8️⃣  tools/call – create_retention_case")
        tests_run += 1
        resp = await client.call_tool("create_retention_case", {
            "customer_id": "CUST-001",
            "action": {"type": "personal_outreach", "description": "Executive call"},
        })
        tool_result = resp.get("result", {})
        if not tool_result.get("isError", True):
            payload = json.loads(tool_result["content"][0]["text"])
            if payload.get("case_id", "").startswith("CASE-") and payload.get("status") == "open":
                print(f"   ✅ PASSED – case_id={payload['case_id']}, status=open")
                tests_passed += 1
            else:
                print(f"   ❌ FAILED – unexpected payload: {payload}")
                all_passed = False
        else:
            print(f"   ❌ FAILED – tool error: {tool_result}")
            all_passed = False

    # ---- Summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 Functional Test Summary")
    print("=" * 70)
    print(f"\n   Tests run:    {tests_run}")
    print(f"   Tests passed: {tests_passed}")
    print(f"   Tests failed: {tests_run - tests_passed}")

    if all_passed:
        print("\n✅ All functional tests PASSED!")
        print("\nCapabilities verified:")
        print("   ✓ Health endpoint")
        print("   ✓ SSE session establishment")
        print("   ✓ MCP initialize (JSON-RPC 2.0)")
        print("   ✓ tools/list (7 CCA-001 tools)")
        print("   ✓ tools/call – get_customer_facts")
        print("   ✓ tools/call – get_churn_prediction")
        print("   ✓ tools/call – recommend_retention_action")
        print("   ✓ tools/call – create_retention_case")
    else:
        print("\n❌ Some functional tests FAILED")
        print("\nTroubleshooting:")
        print("  1. Ensure autonomous_agent.py is running (uvicorn or kubectl port-forward)")
        print("  2. Check the base URL (default: http://localhost:8001)")
        print("  3. Review agent logs for errors")

    return all_passed


def main():
    try:
        result = asyncio.run(run_tests())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
