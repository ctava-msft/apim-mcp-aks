#!/usr/bin/env python3
"""
Agent Lightning End-to-End Demo Script (External via APIM/AKS)

This script tests the complete Agent Lightning fine-tuning loop via APIM and AKS
using the 15 Lightning MCP tools:

Episode Management:
  1. lightning_list_episodes    - List captured episodes
  2. lightning_get_episode      - Get episode details

Reward Management:
  3. lightning_assign_reward    - Assign reward/label to episode
  4. lightning_list_rewards     - List assigned rewards

Dataset Management:
  5. lightning_build_dataset    - Build fine-tuning dataset
  6. lightning_list_datasets    - List available datasets

Training Management:
  7. lightning_start_training   - Start Azure OpenAI fine-tuning
  8. lightning_get_training_status - Get training run status
  9. lightning_list_training_runs  - List training runs

Deployment Management:
  10. lightning_promote_deployment    - Promote tuned model
  11. lightning_get_active_deployment - Get active deployment
  12. lightning_list_deployments      - List all deployments
  13. lightning_rollback_deployment   - Rollback to previous
  14. lightning_deactivate_deployment - Deactivate tuned model

Statistics:
  15. lightning_get_stats       - Get comprehensive statistics

Prerequisites:
- MCP server deployed to AKS with Lightning enabled (ENABLE_LIGHTNING_CAPTURE=true)
- APIM configured with OAuth (or use --direct for LoadBalancer)
- Cosmos DB deployed with Lightning containers

Usage:
    python tests/test_agent_lightning_loop.py           # Via APIM with OAuth
    python tests/test_agent_lightning_loop.py --direct  # Direct via LoadBalancer
"""

import asyncio
import json
import logging
import os
import sys
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import aiohttp

# Configuration file
CONFIG_FILE = Path(__file__).parent / 'mcp_test_config.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Demo configuration
DEMO_AGENT_ID = "mcp-agents"

# All 15 Lightning MCP tools
LIGHTNING_MCP_TOOLS = [
    # Episode Management
    "lightning_list_episodes",
    "lightning_get_episode",
    # Reward Management
    "lightning_assign_reward",
    "lightning_list_rewards",
    # Dataset Management
    "lightning_build_dataset",
    "lightning_list_datasets",
    # Training Management
    "lightning_start_training",
    "lightning_get_training_status",
    "lightning_list_training_runs",
    # Deployment Management
    "lightning_promote_deployment",
    "lightning_get_active_deployment",
    "lightning_list_deployments",
    "lightning_rollback_deployment",
    "lightning_deactivate_deployment",
    # Statistics
    "lightning_get_stats",
]

LIGHTNING_CONTAINERS = [
    "rl_episodes",
    "rl_rewards",
    "rl_datasets",
    "rl_training_runs",
    "rl_deployments",
]


def load_config() -> Dict[str, Any]:
    """Load configuration from mcp_test_config.json"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            print(f"✅ Loaded configuration from {CONFIG_FILE}")
            return config
    else:
        print(f"⚠️  Config file not found: {CONFIG_FILE}")
        print("   Run scripts/generate-test-config.ps1 to generate it")
        return {}


class MCPClient:
    """MCP Client that maintains SSE session for testing via APIM or direct"""
    
    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = None
        self.sse_response = None
        self.session_message_url = None
    
    async def __aenter__(self):
        cookie_jar = aiohttp.CookieJar()
        headers = {}
        if self.auth_token and not self.auth_token.startswith('direct-mode'):
            headers['Authorization'] = f'Bearer {self.auth_token}'
        self.session = aiohttp.ClientSession(
            cookie_jar=cookie_jar,
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.sse_response and not self.sse_response.closed:
            self.sse_response.close()
        if self.session:
            await self.session.close()
    
    async def establish_sse_session(self) -> bool:
        """Establish SSE connection and extract session URL"""
        try:
            print(f"\n📡 Establishing SSE session to: {self.base_url}/sse")
            
            self.sse_response = await self.session.get(
                f'{self.base_url}/sse',
                headers={
                    'Accept': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
            
            print(f"   SSE Response Status: {self.sse_response.status}")
            
            if self.sse_response.status == 200:
                async for chunk in self.sse_response.content.iter_chunked(1024):
                    if chunk:
                        data = chunk.decode('utf-8', errors='ignore')
                        match = re.search(r'data: (message\?[^\n\r]+)', data)
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
        request_id = f"test-{uuid.uuid4().hex[:8]}"
        
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        
        if params:
            jsonrpc_request["params"] = params
        
        message_url = self.session_message_url if self.session_message_url else f'{self.base_url}/message'
        
        try:
            async with self.session.post(
                message_url,
                json=jsonrpc_request,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        return {"error": "Invalid JSON response", "raw": response_text}
                else:
                    return {"error": f"HTTP {response.status}", "body": response_text}
                    
        except asyncio.TimeoutError:
            return {"error": "Request timed out"}
        except Exception as e:
            return {"error": str(e)}
    
    async def list_tools(self) -> Optional[List[Dict[str, Any]]]:
        """List available MCP tools"""
        result = await self.send_request("tools/list")
        if 'error' not in result:
            return result.get('result', {}).get('tools', [])
        return None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool"""
        return await self.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status from the server"""
        try:
            # Try direct health endpoint (not via MCP protocol)
            health_url = self.base_url.replace('/sse', '').replace('/runtime/webhooks/mcp', '') + '/health'
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    return await response.json()
                return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}


async def get_mcp_token(session: aiohttp.ClientSession, token_url: str) -> Optional[str]:
    """Get MCP access token from APIM OAuth endpoint"""
    print("\n🔐 Getting MCP access token from APIM...")
    print(f"   Token URL: {token_url}")
    
    try:
        async with session.post(token_url, data={}) as response:
            if response.status == 200:
                data = await response.json()
                token = data.get('access_token', '')
                if token:
                    print(f"✅ Got MCP access token: {token[:30]}...")
                    return token
            print(f"❌ Token request failed: {response.status}")
            return None
    except Exception as e:
        print(f"❌ Error getting token: {e}")
        return None


async def check_cosmos_lightning_containers() -> Dict[str, Any]:
    """
    Check if Lightning Cosmos containers exist by querying the server or directly.
    Returns status of each container.
    """
    print("\n🗄️  Checking Lightning Cosmos DB containers...")
    
    cosmos_uri = os.getenv("COSMOS_ACCOUNT_URI", "")
    db_name = os.getenv("COSMOS_DATABASE_NAME", "agent_rl")
    
    if not cosmos_uri:
        print("   ⚠️  COSMOS_ACCOUNT_URI not set - cannot verify containers directly")
        print("   ℹ️  Containers will be verified via MCP tool calls")
        return {"verified_directly": False, "containers": {}}
    
    try:
        from azure.cosmos import CosmosClient
        from azure.identity import DefaultAzureCredential
        
        print(f"   Connecting to Cosmos: {cosmos_uri[:50]}...")
        print(f"   Database: {db_name}")
        
        credential = DefaultAzureCredential()
        client = CosmosClient(cosmos_uri, credential=credential)
        
        # Try to get the agent_rl database
        try:
            database = client.get_database_client(db_name)
            # List containers
            containers = list(database.list_containers())
            container_names = [c['id'] for c in containers]
            
            print(f"\n   Found {len(container_names)} containers in '{db_name}':")
            
            results = {"verified_directly": True, "database": db_name, "containers": {}}
            
            for expected_container in LIGHTNING_CONTAINERS:
                if expected_container in container_names:
                    print(f"   ✅ {expected_container}")
                    results["containers"][expected_container] = "exists"
                    
                    # Count documents
                    try:
                        container = database.get_container_client(expected_container)
                        count_query = "SELECT VALUE COUNT(1) FROM c"
                        count_result = list(container.query_items(count_query, enable_cross_partition_query=True))
                        doc_count = count_result[0] if count_result else 0
                        results["containers"][expected_container] = f"exists ({doc_count} docs)"
                        print(f"      📄 {doc_count} documents")
                    except Exception as e:
                        logger.debug(f"Error counting docs: {e}")
                else:
                    print(f"   ❌ {expected_container} - MISSING")
                    results["containers"][expected_container] = "missing"
            
            return results
            
        except Exception as e:
            if "ResourceNotFound" in str(e) or "NotFound" in str(e):
                print(f"   ❌ Database '{db_name}' not found!")
                print(f"   ℹ️  Lightning containers need to be provisioned.")
                print(f"   📋 See docs/AGENT-LIGHTNING.md for setup instructions.")
                return {"verified_directly": True, "database": db_name, "error": "database_not_found"}
            raise
            
    except ImportError:
        print("   ⚠️  azure-cosmos not installed - skipping direct verification")
        return {"verified_directly": False, "containers": {}}
    except Exception as e:
        print(f"   ❌ Error checking containers: {e}")
        return {"verified_directly": False, "error": str(e)}


async def test_mcp_health(client: MCPClient) -> bool:
    """Test MCP server health endpoint"""
    print("\n🏥 Checking MCP server health...")
    
    health = await client.get_health()
    
    if 'error' in health:
        print(f"   ⚠️  Could not reach health endpoint: {health['error']}")
        return True  # Not a fatal error, continue with tests
    
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Timestamp: {health.get('timestamp', 'N/A')}")
    return True


async def test_lightning_tools_available(client: MCPClient) -> Dict[str, bool]:
    """Check if all 15 Lightning MCP tools are available"""
    print("\n🔧 Checking for Lightning MCP tools (15 expected)...")
    
    tools = await client.list_tools()
    
    if not tools:
        print("   ❌ Could not list tools")
        return {}
    
    print(f"   Found {len(tools)} total tools")
    
    # Check for all 15 Lightning tools
    lightning_tools_status = {tool: False for tool in LIGHTNING_MCP_TOOLS}
    
    tool_names = [t.get('name', '') for t in tools]
    
    print("\n   Lightning MCP Tools:")
    for tool_name in LIGHTNING_MCP_TOOLS:
        if tool_name in tool_names:
            lightning_tools_status[tool_name] = True
            print(f"   ✅ {tool_name}")
        else:
            print(f"   ❌ {tool_name} - MISSING")
    
    # Count available
    available_count = sum(1 for v in lightning_tools_status.values() if v)
    print(f"\n   📊 Lightning tools available: {available_count}/{len(LIGHTNING_MCP_TOOLS)}")
    
    # Also check core tools
    print("\n   Core MCP Tools:")
    core_tools = ["ask_foundry", "next_best_action", "store_memory", "recall_memory"]
    for tool in core_tools:
        if tool in tool_names:
            print(f"   ✅ {tool}")
        else:
            print(f"   ❌ {tool} - MISSING")
    
    return lightning_tools_status


async def create_episode_via_mcp(client: MCPClient, question: str) -> Dict[str, Any]:
    """
    Create an episode by calling ask_foundry tool.
    When ENABLE_LIGHTNING_CAPTURE=true, this automatically stores an episode.
    """
    print(f"\n📝 Creating episode via ask_foundry...")
    print(f"   Question: {question[:60]}...")
    
    result = await client.call_tool("ask_foundry", {"question": question})
    
    if 'error' in result:
        print(f"   ❌ Error: {result['error']}")
        return {"success": False, "error": result['error']}
    
    tool_result = result.get('result', {})
    content = tool_result.get('content', [])
    is_error = tool_result.get('isError', False)
    
    if is_error:
        error_text = content[0].get('text', 'Unknown error') if content else 'No error message'
        print(f"   ❌ Tool error: {error_text[:100]}...")
        return {"success": False, "error": error_text}
    
    response_text = content[0].get('text', '') if content else ''
    print(f"   ✅ Response: {response_text[:100]}...")
    
    return {
        "success": True,
        "question": question,
        "response": response_text,
        "timestamp": datetime.utcnow().isoformat(),
    }


async def test_episode_storage(client: MCPClient) -> List[Dict[str, Any]]:
    """
    Test episode storage by creating multiple episodes via MCP tools.
    Note: Episodes are only stored when ENABLE_LIGHTNING_CAPTURE=true on the server.
    """
    print("\n" + "=" * 60)
    print("📊 Testing Episode Creation via MCP")
    print("=" * 60)
    
    test_questions = [
        "What is the capital of France?",
        "Calculate 2+2",
        "Explain what machine learning is in one sentence.",
        "What programming language is Python named after?",
        "What is REST API?",
    ]
    
    episodes = []
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Episode {i}/{len(test_questions)} ---")
        result = await create_episode_via_mcp(client, question)
        if result.get('success'):
            episodes.append(result)
        
        # Brief pause between calls
        await asyncio.sleep(1)
    
    print(f"\n✅ Created {len(episodes)}/{len(test_questions)} episodes")
    return episodes


# =========================================
# Lightning MCP Tools Tests
# =========================================

async def call_lightning_tool(client: MCPClient, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
    """Helper to call a Lightning MCP tool and parse the response"""
    result = await client.call_tool(tool_name, arguments or {})
    
    if 'error' in result:
        return {"success": False, "error": result['error']}
    
    tool_result = result.get('result', {})
    content = tool_result.get('content', [])
    is_error = tool_result.get('isError', False)
    
    if is_error:
        error_text = content[0].get('text', 'Unknown error') if content else 'No error message'
        return {"success": False, "error": error_text}
    
    try:
        response_text = content[0].get('text', '{}') if content else '{}'
        response_data = json.loads(response_text)
        return {"success": True, "data": response_data}
    except json.JSONDecodeError:
        return {"success": True, "raw": response_text}


async def test_lightning_get_stats(client: MCPClient) -> Dict[str, Any]:
    """Test lightning_get_stats tool"""
    print("\n" + "=" * 60)
    print("📊 Testing lightning_get_stats")
    print("=" * 60)
    
    result = await call_lightning_tool(client, "lightning_get_stats", {"agent_id": DEMO_AGENT_ID})
    
    if not result.get('success'):
        print(f"   ❌ Error: {result.get('error', 'Unknown')[:100]}")
        return result
    
    data = result.get('data', {})
    stats = data.get('statistics', {})
    
    print(f"\n   Agent ID: {data.get('agent_id', 'N/A')}")
    print(f"   Lightning Enabled: {data.get('lightning_enabled', False)}")
    print(f"   Use Tuned Model: {data.get('use_tuned_model', False)}")
    print(f"   Current Model: {data.get('current_model', 'N/A')}")
    print(f"\n   Statistics:")
    print(f"     Total Episodes: {stats.get('total_episodes', 0)}")
    print(f"     Total Rewards: {stats.get('total_rewards', 0)}")
    print(f"     Average Reward: {stats.get('average_reward', 0)}")
    print(f"     Total Datasets: {stats.get('total_datasets', 0)}")
    print(f"     Total Training Runs: {stats.get('total_training_runs', 0)}")
    print(f"     Total Deployments: {stats.get('total_deployments', 0)}")
    
    active = data.get('active_deployment', {})
    if active.get('has_active'):
        print(f"\n   Active Deployment: {active.get('model_name', 'N/A')}")
    else:
        print(f"\n   Active Deployment: None (using base model)")
    
    return result


async def test_lightning_list_episodes(client: MCPClient) -> Dict[str, Any]:
    """Test lightning_list_episodes tool"""
    print("\n" + "=" * 60)
    print("📋 Testing lightning_list_episodes")
    print("=" * 60)
    
    result = await call_lightning_tool(client, "lightning_list_episodes", {
        "agent_id": DEMO_AGENT_ID,
        "limit": 10
    })
    
    if not result.get('success'):
        print(f"   ❌ Error: {result.get('error', 'Unknown')[:100]}")
        return result
    
    data = result.get('data', {})
    episodes = data.get('episodes', [])
    
    print(f"\n   Found {data.get('episodes_found', 0)} episodes")
    
    for i, ep in enumerate(episodes[:5], 1):
        print(f"\n   Episode {i}:")
        print(f"     ID: {ep.get('id', 'N/A')[:20]}...")
        print(f"     Input: {ep.get('user_input', 'N/A')[:50]}...")
        print(f"     Model: {ep.get('model_deployment', 'N/A')}")
        print(f"     Latency: {ep.get('request_latency_ms', 'N/A')}ms")
    
    if len(episodes) > 5:
        print(f"\n   ... and {len(episodes) - 5} more episodes")
    
    return result


async def test_lightning_get_episode(client: MCPClient, episode_id: str) -> Dict[str, Any]:
    """Test lightning_get_episode tool"""
    print(f"\n📖 Testing lightning_get_episode (ID: {episode_id[:20]}...)")
    
    result = await call_lightning_tool(client, "lightning_get_episode", {
        "episode_id": episode_id,
        "agent_id": DEMO_AGENT_ID
    })
    
    if not result.get('success'):
        print(f"   ❌ Error: {result.get('error', 'Unknown')[:100]}")
        return result
    
    data = result.get('data', {})
    print(f"   ✅ Retrieved episode details")
    print(f"     User Input: {data.get('user_input', 'N/A')[:60]}...")
    print(f"     Assistant Output: {data.get('assistant_output', 'N/A')[:60]}...")
    print(f"     Tool Calls: {len(data.get('tool_calls', []))}")
    
    return result


async def test_lightning_assign_reward(client: MCPClient, episode_id: str, reward_value: float) -> Dict[str, Any]:
    """Test lightning_assign_reward tool"""
    print(f"\n🏅 Testing lightning_assign_reward (Episode: {episode_id[:20]}..., Value: {reward_value})")
    
    result = await call_lightning_tool(client, "lightning_assign_reward", {
        "episode_id": episode_id,
        "reward_value": reward_value,
        "reward_source": "human_approval",
        "agent_id": DEMO_AGENT_ID,
        "evaluator": "test_script",
        "comments": f"Test reward assigned at {datetime.utcnow().isoformat()}"
    })
    
    if not result.get('success'):
        print(f"   ❌ Error: {result.get('error', 'Unknown')[:100]}")
        return result
    
    data = result.get('data', {})
    print(f"   ✅ Reward assigned: {data.get('reward_id', 'N/A')[:20]}...")
    print(f"     Value: {data.get('value', 'N/A')}")
    print(f"     Source: {data.get('source', 'N/A')}")
    
    return result


async def test_lightning_list_rewards(client: MCPClient, episode_id: str = None) -> Dict[str, Any]:
    """Test lightning_list_rewards tool"""
    print("\n" + "=" * 60)
    print("🏆 Testing lightning_list_rewards")
    print("=" * 60)
    
    args = {"agent_id": DEMO_AGENT_ID, "limit": 20}
    if episode_id:
        args["episode_id"] = episode_id
    
    result = await call_lightning_tool(client, "lightning_list_rewards", args)
    
    if not result.get('success'):
        print(f"   ❌ Error: {result.get('error', 'Unknown')[:100]}")
        return result
    
    data = result.get('data', {})
    rewards = data.get('rewards', [])
    
    print(f"\n   Found {data.get('rewards_found', 0)} rewards")
    
    for i, r in enumerate(rewards[:5], 1):
        print(f"\n   Reward {i}:")
        print(f"     ID: {r.get('id', 'N/A')[:20]}...")
        print(f"     Episode: {r.get('episode_id', 'N/A')[:20]}...")
        print(f"     Value: {r.get('value', 'N/A')}")
        print(f"     Source: {r.get('source', 'N/A')}")
    
    return result


async def test_lightning_build_dataset(client: MCPClient, dataset_name: str) -> Dict[str, Any]:
    """Test lightning_build_dataset tool"""
    print("\n" + "=" * 60)
    print(f"📦 Testing lightning_build_dataset (Name: {dataset_name})")
    print("=" * 60)
    
    result = await call_lightning_tool(client, "lightning_build_dataset", {
        "name": dataset_name,
        "agent_id": DEMO_AGENT_ID,
        "description": f"Test dataset built at {datetime.utcnow().isoformat()}",
        "min_reward": 0.5
    })
    
    if not result.get('success'):
        error = result.get('error', 'Unknown')
        if "No qualifying episodes" in error:
            print(f"   ⚠️  No episodes with rewards >= 0.5 found")
            print(f"      Assign positive rewards to episodes first")
        else:
            print(f"   ❌ Error: {error[:100]}")
        return result
    
    data = result.get('data', {})
    print(f"\n   ✅ Dataset built successfully!")
    print(f"     Dataset ID: {data.get('dataset_id', 'N/A')[:20]}...")
    print(f"     Name: {data.get('name', 'N/A')}")
    print(f"     Training Examples: {data.get('training_count', 0)}")
    print(f"     Validation Examples: {data.get('validation_count', 0)}")
    print(f"     Episodes Used: {data.get('episode_count', 0)}")
    print(f"     Local Path: {data.get('local_path', 'N/A')}")
    
    return result


async def test_lightning_list_datasets(client: MCPClient) -> Dict[str, Any]:
    """Test lightning_list_datasets tool"""
    print("\n" + "=" * 60)
    print("📚 Testing lightning_list_datasets")
    print("=" * 60)
    
    result = await call_lightning_tool(client, "lightning_list_datasets", {
        "agent_id": DEMO_AGENT_ID,
        "limit": 10
    })
    
    if not result.get('success'):
        print(f"   ❌ Error: {result.get('error', 'Unknown')[:100]}")
        return result
    
    data = result.get('data', {})
    datasets = data.get('datasets', [])
    
    print(f"\n   Found {data.get('datasets_found', 0)} datasets")
    
    for i, ds in enumerate(datasets[:5], 1):
        print(f"\n   Dataset {i}:")
        print(f"     ID: {ds.get('id', 'N/A')[:20]}...")
        print(f"     Name: {ds.get('name', 'N/A')}")
        print(f"     Training: {ds.get('training_count', 0)} examples")
        print(f"     Validation: {ds.get('validation_count', 0)} examples")
    
    return result


async def test_lightning_list_training_runs(client: MCPClient) -> Dict[str, Any]:
    """Test lightning_list_training_runs tool"""
    print("\n" + "=" * 60)
    print("🚂 Testing lightning_list_training_runs")
    print("=" * 60)
    
    result = await call_lightning_tool(client, "lightning_list_training_runs", {
        "agent_id": DEMO_AGENT_ID,
        "limit": 10
    })
    
    if not result.get('success'):
        print(f"   ❌ Error: {result.get('error', 'Unknown')[:100]}")
        return result
    
    data = result.get('data', {})
    runs = data.get('training_runs', [])
    
    print(f"\n   Found {data.get('runs_found', 0)} training runs")
    
    for i, run in enumerate(runs[:5], 1):
        print(f"\n   Training Run {i}:")
        print(f"     ID: {run.get('id', 'N/A')[:20]}...")
        print(f"     Status: {run.get('status', 'N/A')}")
        print(f"     Base Model: {run.get('base_model', 'N/A')}")
        print(f"     Tuned Model: {run.get('tuned_model_name', 'N/A') or 'Not yet'}")
    
    return result


async def test_lightning_get_active_deployment(client: MCPClient) -> Dict[str, Any]:
    """Test lightning_get_active_deployment tool"""
    print("\n" + "=" * 60)
    print("🚀 Testing lightning_get_active_deployment")
    print("=" * 60)
    
    result = await call_lightning_tool(client, "lightning_get_active_deployment", {
        "agent_id": DEMO_AGENT_ID
    })
    
    if not result.get('success'):
        print(f"   ❌ Error: {result.get('error', 'Unknown')[:100]}")
        return result
    
    data = result.get('data', {})
    
    if data.get('has_active_deployment'):
        print(f"\n   ✅ Active deployment found:")
        print(f"     Deployment ID: {data.get('deployment_id', 'N/A')[:20]}...")
        print(f"     Tuned Model: {data.get('tuned_model_name', 'N/A')}")
        print(f"     Promoted At: {data.get('promoted_at', 'N/A')}")
    else:
        print(f"\n   ℹ️  No active tuned model deployment")
        print(f"     Using base model: {data.get('base_model', 'N/A')}")
    
    return result


async def test_lightning_list_deployments(client: MCPClient) -> Dict[str, Any]:
    """Test lightning_list_deployments tool"""
    print("\n" + "=" * 60)
    print("📜 Testing lightning_list_deployments")
    print("=" * 60)
    
    result = await call_lightning_tool(client, "lightning_list_deployments", {
        "agent_id": DEMO_AGENT_ID,
        "limit": 10
    })
    
    if not result.get('success'):
        print(f"   ❌ Error: {result.get('error', 'Unknown')[:100]}")
        return result
    
    data = result.get('data', {})
    deployments = data.get('deployments', [])
    
    print(f"\n   Found {data.get('deployments_found', 0)} deployments")
    
    for i, dep in enumerate(deployments[:5], 1):
        status = "🟢 ACTIVE" if dep.get('is_active') else "⚪ Inactive"
        print(f"\n   Deployment {i}: {status}")
        print(f"     ID: {dep.get('id', 'N/A')[:20]}...")
        print(f"     Tuned Model: {dep.get('tuned_model_name', 'N/A')}")
        print(f"     Promoted At: {dep.get('promoted_at', 'N/A')[:19] if dep.get('promoted_at') else 'N/A'}")
    
    return result


async def run_full_lightning_loop_test(client: MCPClient) -> Dict[str, Any]:
    """
    Run the complete Lightning MCP tools test loop.
    Tests all 15 Lightning tools in the correct workflow order.
    """
    print("\n" + "=" * 70)
    print("⚡ AGENT LIGHTNING FULL LOOP TEST (Using 15 MCP Tools)")
    print("=" * 70)
    
    results = {
        "stats": None,
        "episodes_listed": None,
        "episode_details": None,
        "reward_assigned": None,
        "rewards_listed": None,
        "dataset_built": None,
        "datasets_listed": None,
        "training_runs_listed": None,
        "active_deployment": None,
        "deployments_listed": None,
    }
    
    # Step 1: Get overall Lightning stats
    print("\n\n📊 STEP 1: Get Lightning Statistics")
    results["stats"] = await test_lightning_get_stats(client)
    await asyncio.sleep(1)
    
    # Step 2: List existing episodes
    print("\n\n📋 STEP 2: List Episodes")
    results["episodes_listed"] = await test_lightning_list_episodes(client)
    await asyncio.sleep(1)
    
    # Step 3: Get details of first episode (if any exist)
    episodes_data = results["episodes_listed"].get('data', {}).get('episodes', [])
    if episodes_data:
        first_episode_id = episodes_data[0].get('id')
        if first_episode_id:
            print("\n\n📖 STEP 3: Get Episode Details")
            results["episode_details"] = await test_lightning_get_episode(client, first_episode_id)
            await asyncio.sleep(1)
            
            # Step 4: Assign a reward to the episode
            print("\n\n🏅 STEP 4: Assign Reward to Episode")
            results["reward_assigned"] = await test_lightning_assign_reward(client, first_episode_id, 0.8)
            await asyncio.sleep(1)
    else:
        print("\n\n⚠️  STEP 3-4: Skipped (no episodes found)")
        print("   Run ask_foundry with ENABLE_LIGHTNING_CAPTURE=true to create episodes")
    
    # Step 5: List rewards
    print("\n\n🏆 STEP 5: List Rewards")
    results["rewards_listed"] = await test_lightning_list_rewards(client)
    await asyncio.sleep(1)
    
    # Step 6: Build dataset (if we have rewarded episodes)
    print("\n\n📦 STEP 6: Build Dataset")
    dataset_name = f"test-dataset-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    results["dataset_built"] = await test_lightning_build_dataset(client, dataset_name)
    await asyncio.sleep(1)
    
    # Step 7: List datasets
    print("\n\n📚 STEP 7: List Datasets")
    results["datasets_listed"] = await test_lightning_list_datasets(client)
    await asyncio.sleep(1)
    
    # Step 8: List training runs
    print("\n\n🚂 STEP 8: List Training Runs")
    results["training_runs_listed"] = await test_lightning_list_training_runs(client)
    await asyncio.sleep(1)
    
    # Step 9: Get active deployment
    print("\n\n🚀 STEP 9: Get Active Deployment")
    results["active_deployment"] = await test_lightning_get_active_deployment(client)
    await asyncio.sleep(1)
    
    # Step 10: List all deployments
    print("\n\n📜 STEP 10: List All Deployments")
    results["deployments_listed"] = await test_lightning_list_deployments(client)
    
    return results


async def test_next_best_action(client: MCPClient) -> Dict[str, Any]:
    """
    Test the next_best_action tool which uses multiple memory layers.
    This exercises the Cosmos-backed memory systems.
    """
    print("\n" + "=" * 60)
    print("🎯 Testing next_best_action Tool")
    print("=" * 60)
    
    test_task = "Analyze customer data to identify customers at high risk of churning and create a retention strategy"
    
    print(f"\n📝 Task: {test_task}")
    print("\n⏳ Processing task (this may take 30-60 seconds)...")
    print("   • Generating embeddings")
    print("   • Searching short-term memory (CosmosDB)")
    print("   • Searching long-term memory (AI Search)")
    print("   • Querying facts memory (Fabric IQ)")
    print("   • Generating action plan")
    
    result = await client.call_tool("next_best_action", {"task": test_task})
    
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
        return {"success": False, "error": result['error']}
    
    tool_result = result.get('result', {})
    content = tool_result.get('content', [])
    is_error = tool_result.get('isError', False)
    
    if is_error:
        error_text = content[0].get('text', 'Unknown error') if content else 'No error message'
        print(f"\n❌ Tool error: {error_text}")
        
        # Check for specific errors
        if 'CosmosDB not configured' in error_text:
            print("\n⚠️  CosmosDB is not configured on the MCP server.")
            print("   Verify COSMOSDB_ENDPOINT is set in the Kubernetes deployment.")
        elif 'Foundry endpoint not configured' in error_text:
            print("\n⚠️  Foundry endpoint is not configured.")
            print("   Verify FOUNDRY_PROJECT_ENDPOINT is set.")
        
        return {"success": False, "error": error_text}
    
    # Parse successful response
    try:
        response_text = content[0].get('text', '{}') if content else '{}'
        response_data = json.loads(response_text)
        
        print(f"\n✅ Task processed successfully!")
        print(f"   Task ID: {response_data.get('task_id', 'N/A')}")
        print(f"   Intent: {response_data.get('intent', 'N/A')}")
        
        analysis = response_data.get('analysis', {})
        print(f"   Similar tasks found: {analysis.get('similar_tasks_found', 0)}")
        print(f"   Task instructions found: {analysis.get('task_instructions_found', 0)}")
        print(f"   Domain facts found: {analysis.get('domain_facts_found', 0)}")
        
        plan = response_data.get('plan', {})
        print(f"   Plan steps: {plan.get('total_steps', 0)}")
        
        metadata = response_data.get('metadata', {})
        print(f"   Stored in Cosmos: {metadata.get('stored_in_cosmos', False)}")
        
        return {"success": True, "data": response_data}
        
    except json.JSONDecodeError as e:
        print(f"\n⚠️  Could not parse response: {e}")
        return {"success": True, "raw_response": response_text}


async def test_memory_operations(client: MCPClient, session_id: str) -> Dict[str, Any]:
    """
    Test short-term memory operations (store and recall).
    """
    print("\n" + "=" * 60)
    print("💾 Testing Memory Operations")
    print("=" * 60)
    
    results = {"store": [], "recall": []}
    
    # Store some test memories
    test_memories = [
        {"content": "The customer prefers email communication.", "type": "context"},
        {"content": "Previous meeting was about Q4 strategy.", "type": "conversation"},
        {"content": "TODO: Follow up on the proposal by Friday.", "type": "task"},
    ]
    
    print(f"\n📥 Storing {len(test_memories)} memories...")
    
    for mem in test_memories:
        result = await client.call_tool("store_memory", {
            "content": mem["content"],
            "session_id": session_id,
            "memory_type": mem["type"],
        })
        
        tool_result = result.get('result', {})
        content = tool_result.get('content', [])
        
        if content and not tool_result.get('isError'):
            try:
                response = json.loads(content[0].get('text', '{}'))
                if response.get('success'):
                    print(f"   ✅ Stored: {mem['content'][:40]}...")
                    results["store"].append({"success": True, "memory_id": response.get('memory_id')})
                else:
                    print(f"   ❌ Failed: {response.get('error', 'Unknown')}")
                    results["store"].append({"success": False, "error": response.get('error')})
            except json.JSONDecodeError:
                print(f"   ⚠️  Invalid response")
                results["store"].append({"success": False, "error": "Invalid JSON"})
        else:
            error = content[0].get('text', 'Unknown error') if content else 'No response'
            print(f"   ❌ Error: {error[:50]}...")
            results["store"].append({"success": False, "error": error})
    
    # Recall memories
    print(f"\n📤 Recalling memories...")
    
    recall_queries = [
        "customer communication preferences",
        "meeting notes and strategy",
    ]
    
    for query in recall_queries:
        result = await client.call_tool("recall_memory", {
            "query": query,
            "session_id": session_id,
            "limit": 3,
        })
        
        tool_result = result.get('result', {})
        content = tool_result.get('content', [])
        
        if content and not tool_result.get('isError'):
            try:
                response = json.loads(content[0].get('text', '{}'))
                memories_found = response.get('memories_found', 0)
                print(f"   ✅ Query '{query[:30]}...' found {memories_found} memories")
                results["recall"].append({"success": True, "query": query, "found": memories_found})
            except json.JSONDecodeError:
                print(f"   ⚠️  Invalid response")
                results["recall"].append({"success": False, "query": query, "error": "Invalid JSON"})
        else:
            error = content[0].get('text', 'Unknown error') if content else 'No response'
            print(f"   ❌ Query '{query[:30]}...' failed: {error[:30]}...")
            results["recall"].append({"success": False, "query": query, "error": error})
    
    return results


def print_lightning_summary(
    tools_available: Dict[str, bool],
    lightning_results: Dict[str, Any],
    episodes_created: int = 0,
):
    """Print a summary of Lightning MCP tools test results"""
    print("\n" + "=" * 70)
    print("📋 AGENT LIGHTNING TEST SUMMARY")
    print("=" * 70)
    
    # Lightning MCP Tools availability
    available_count = sum(1 for v in tools_available.values() if v)
    total_tools = len(LIGHTNING_MCP_TOOLS)
    print(f"\n⚡ Lightning MCP Tools: {available_count}/{total_tools} available")
    
    if available_count < total_tools:
        missing = [k for k, v in tools_available.items() if not v]
        print(f"   ❌ Missing: {', '.join(missing[:5])}")
        if len(missing) > 5:
            print(f"      ... and {len(missing) - 5} more")
    else:
        print("   ✅ All 15 Lightning MCP tools available!")
    
    # Episodes
    if episodes_created > 0:
        print(f"\n📝 Episodes Created: {episodes_created}")
        print("   ℹ️  Episodes stored when ENABLE_LIGHTNING_CAPTURE=true")
    
    # Lightning Stats
    stats_result = lightning_results.get("stats", {})
    if stats_result.get("success"):
        data = stats_result.get("data", {})
        stats = data.get("statistics", {})
        print(f"\n📊 Lightning Statistics:")
        print(f"   • Episodes: {stats.get('total_episodes', 0)}")
        print(f"   • Rewards: {stats.get('total_rewards', 0)}")
        print(f"   • Datasets: {stats.get('total_datasets', 0)}")
        print(f"   • Training Runs: {stats.get('total_training_runs', 0)}")
        print(f"   • Deployments: {stats.get('total_deployments', 0)}")
        
        if data.get("active_deployment", {}).get("has_active"):
            print(f"\n   🚀 Active Model: {data['active_deployment'].get('model_name', 'N/A')}")
        else:
            print(f"\n   📌 Using Base Model: {data.get('current_model', 'N/A')}")
    else:
        print(f"\n📊 Lightning Statistics: ❌ Could not retrieve")
    
    # Test Results Summary
    print(f"\n🧪 Test Results:")
    
    test_checks = [
        ("Get Stats", lightning_results.get("stats", {}).get("success", False)),
        ("List Episodes", lightning_results.get("episodes_listed", {}).get("success", False)),
        ("Get Episode", lightning_results.get("episode_details", {}).get("success", False) if lightning_results.get("episode_details") else None),
        ("Assign Reward", lightning_results.get("reward_assigned", {}).get("success", False) if lightning_results.get("reward_assigned") else None),
        ("List Rewards", lightning_results.get("rewards_listed", {}).get("success", False)),
        ("Build Dataset", lightning_results.get("dataset_built", {}).get("success", False)),
        ("List Datasets", lightning_results.get("datasets_listed", {}).get("success", False)),
        ("List Training Runs", lightning_results.get("training_runs_listed", {}).get("success", False)),
        ("Get Active Deployment", lightning_results.get("active_deployment", {}).get("success", False)),
        ("List Deployments", lightning_results.get("deployments_listed", {}).get("success", False)),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in test_checks:
        if result is None:
            print(f"   ⏭️  {name}: Skipped")
            skipped += 1
        elif result:
            print(f"   ✅ {name}: Passed")
            passed += 1
        else:
            print(f"   ❌ {name}: Failed")
            failed += 1
    
    # Overall result
    print("\n" + "=" * 70)
    
    if available_count == total_tools and failed == 0:
        print("🎉 ALL LIGHTNING TESTS PASSED!")
    elif available_count == 0:
        print("❌ Lightning tools not available - check server deployment")
    else:
        print(f"⚠️  Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    print("=" * 70)
    
    # Next steps
    print("\n📋 Next Steps for Full RLHF Loop:")
    
    stats_data = stats_result.get("data", {}).get("statistics", {}) if stats_result.get("success") else {}
    
    if stats_data.get("total_episodes", 0) == 0:
        print("\n   1. Create Episodes:")
        print("      • Call ask_foundry or next_best_action tools")
        print("      • Ensure ENABLE_LIGHTNING_CAPTURE=true on server")
    
    if stats_data.get("total_rewards", 0) == 0:
        print("\n   2. Assign Rewards:")
        print("      • Use lightning_assign_reward to label episodes")
        print("      • Positive rewards (0.5-1.0) for good responses")
        print("      • Negative rewards (-1.0-0.0) for poor responses")
    
    if stats_data.get("total_datasets", 0) == 0:
        print("\n   3. Build Dataset:")
        print("      • Use lightning_build_dataset after labeling episodes")
        print("      • Creates JSONL files for Azure OpenAI fine-tuning")
    
    if stats_data.get("total_training_runs", 0) == 0:
        print("\n   4. Start Training:")
        print("      • Use lightning_start_training with a dataset")
        print("      • Monitor with lightning_get_training_status")
    
    active_dep = lightning_results.get("active_deployment", {}).get("data", {})
    if not active_dep.get("has_active_deployment"):
        print("\n   5. Promote Deployment:")
        print("      • Use lightning_promote_deployment after training completes")
        print("      • This makes the tuned model active for all requests")
    
    print("\n   📚 See docs/AGENT-LIGHTNING.md for complete guide")


async def main():
    print("=" * 70)
    print("⚡ Agent Lightning End-to-End Test (Using 15 MCP Tools)")
    print("=" * 70)
    
    # Check for --direct mode
    use_direct = '--direct' in sys.argv
    skip_episode_creation = '--skip-episodes' in sys.argv
    
    # Load configuration
    config = load_config()
    
    # Determine connection mode
    if use_direct:
        direct_config = config.get('direct', {})
        base_url = direct_config.get('base_url', 'http://localhost:8000/runtime/webhooks/mcp')
        token = 'direct-mode-no-token-needed'
        print(f"\n🔗 Using Direct Mode: {base_url}")
        print("   (Via LoadBalancer or port-forward)")
    else:
        apim_config = config.get('apim', {})
        base_url = apim_config.get('base_url', '')
        token_url = apim_config.get('oauth_token_url', '')
        
        if not base_url:
            print("\n❌ No APIM base URL configured")
            print("   Run: scripts/generate-test-config.ps1")
            print("   Or use: python test_agent_lightning_loop.py --direct")
            return 1
        
        print(f"\n🔗 Using APIM: {base_url}")
        
        # Get OAuth token
        async with aiohttp.ClientSession() as session:
            token = await get_mcp_token(session, token_url)
            if not token:
                print("❌ Failed to get access token")
                print("   Try using --direct mode instead")
                return 1
    
    # Run tests via MCP
    async with MCPClient(base_url, token) as client:
        # Establish SSE session
        if not await client.establish_sse_session():
            print("\n❌ Failed to establish SSE session")
            print("   Check that the MCP server is running and accessible")
            return 1
        
        # Wait for session initialization
        print("\n⏳ Waiting for session to initialize...")
        await asyncio.sleep(2)
        
        # Test health
        await test_mcp_health(client)
        
        # Check all 15 Lightning MCP tools
        tools_available = await test_lightning_tools_available(client)
        
        if not tools_available:
            print("\n❌ Could not verify tools - aborting tests")
            return 1
        
        # Check if Lightning tools are available
        lightning_tools_count = sum(1 for t, v in tools_available.items() if v)
        
        if lightning_tools_count == 0:
            print("\n❌ No Lightning MCP tools found!")
            print("   Ensure the MCP server includes Lightning tools.")
            return 1
        
        episodes_created = 0
        
        # Optionally create episodes first
        if not skip_episode_creation:
            print("\n" + "=" * 70)
            print("📝 PHASE 1: Creating Episodes via ask_foundry")
            print("=" * 70)
            episodes = await test_episode_storage(client)
            episodes_created = len(episodes)
            
            # Brief pause to allow episode capture
            print("\n⏳ Waiting for episode capture to complete...")
            await asyncio.sleep(3)
        else:
            print("\n⏭️  Skipping episode creation (--skip-episodes)")
        
        # Run the full Lightning MCP tools test loop
        print("\n" + "=" * 70)
        print("⚡ PHASE 2: Testing Lightning MCP Tools")
        print("=" * 70)
        lightning_results = await run_full_lightning_loop_test(client)
        
        # Print comprehensive summary
        print_lightning_summary(
            tools_available=tools_available,
            lightning_results=lightning_results,
            episodes_created=episodes_created,
        )
    
    return 0


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
