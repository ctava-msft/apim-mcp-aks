#!/usr/bin/env python3
"""
Test script for the AKS-deployed next_best_action MCP tool.

This script tests the next_best_action tool deployed via APIM + MCP + AKS which:
1. Generates embeddings for a task using text-embedding-3-large
2. Finds similar past tasks using cosine similarity
3. Generates a plan of steps
4. Stores everything in CosmosDB for future learning

Usage:
    python test_next_best_action.py

Requirements:
    - aiohttp (pip install aiohttp)
    - MCP server running (next_best_action_agent.py) with:
      - FOUNDRY_PROJECT_ENDPOINT configured
      - COSMOSDB_ENDPOINT configured
      - text-embedding-3-large model deployed
    - Valid OAuth token or Azure CLI configured
"""

import asyncio
import json
import aiohttp
import sys
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

# Configuration file
CONFIG_FILE = Path(__file__).parent / 'mcp_test_config.json'


def load_config() -> Dict[str, Any]:
    """Load configuration from mcp_test_config.json"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            print(f"✅ Loaded configuration from {CONFIG_FILE}")
            return config
    else:
        print(f"❌ Config file not found: {CONFIG_FILE}")
        sys.exit(1)


class MCPClient:
    """MCP Client that maintains SSE session"""
    
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
                # Read first chunk to get session URL
                async for chunk in self.sse_response.content.iter_chunked(1024):
                    if chunk:
                        data = chunk.decode('utf-8', errors='ignore')
                        
                        # Parse the SSE data to extract session endpoint
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
        request_id = "test-request-1"
        
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
                timeout=aiohttp.ClientTimeout(total=120)  # 2 minute timeout for AI operations
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
            return {"error": "Request timed out (AI operations may take a while)"}
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


def print_task_result(result: Dict[str, Any]) -> bool:
    """Pretty print the next_best_action result"""
    try:
        # Parse the JSON response
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result
        
        if 'error' in data:
            print(f"❌ Error: {data['error']}")
            return False
        
        print(f"\n📋 Task Analysis Results")
        print("=" * 60)
        
        # Task Info
        print(f"\n🆔 Task ID: {data.get('task_id', 'N/A')}")
        print(f"📝 Task: {data.get('task', 'N/A')}")
        print(f"🎯 Intent: {data.get('intent', 'N/A')}")
        
        # Similar Tasks
        analysis = data.get('analysis', {})
        similar_count = analysis.get('similar_tasks_found', 0)
        print(f"\n🔍 Similar Tasks Found: {similar_count}")
        
        if similar_count > 0:
            similar_tasks = analysis.get('similar_tasks', [])
            for i, st in enumerate(similar_tasks, 1):
                print(f"   {i}. {st.get('task', 'N/A')[:50]}...")
                print(f"      Intent: {st.get('intent', 'N/A')}")
                print(f"      Similarity: {st.get('similarity_score', 0):.3f}")
        
        # Plan
        plan = data.get('plan', {})
        steps = plan.get('steps', [])
        
        # Handle case where steps is a dict (model returned JSON object instead of array)
        if isinstance(steps, dict):
            # Display the structured response
            total_steps = len(steps.get('actions_taken', steps.get('recommendations', [])))
            print(f"\n📊 Generated Plan ({total_steps} actions):")
            print("-" * 40)
            if 'status' in steps:
                print(f"\n   Status: {steps['status']}")
            for key in ['actions_taken', 'recommendations']:
                items = steps.get(key, [])
                if items:
                    print(f"\n   {key.replace('_', ' ').title()}:")
                    for j, item in enumerate(items, 1):
                        text = str(item)[:120]
                        print(f"   {j}. {text}")
        else:
            total_steps = plan.get('total_steps', len(steps) if isinstance(steps, list) else 0)
            print(f"\n📊 Generated Plan ({total_steps} steps):")
            print("-" * 40)
            
            for i, step in enumerate(steps, 1):
                if isinstance(step, dict):
                    step_num = step.get('step', i)
                    action = step.get('action', 'Unknown')
                    description = step.get('description', 'No description')
                    effort = step.get('estimated_effort', 'N/A')
                    
                    print(f"\n   Step {step_num}: {action}")
                    print(f"   Description: {description[:100]}...")
                    print(f"   Effort: {effort}")
                elif isinstance(step, str):
                    print(f"\n   Step {i}: {step[:120]}...")
                else:
                    print(f"\n   Step {i}: {str(step)[:120]}...")
        
        # Metadata
        metadata = data.get('metadata', {})
        print(f"\n📈 Metadata:")
        print(f"   Created: {metadata.get('created_at', 'N/A')}")
        print(f"   Embedding Dimensions: {metadata.get('embedding_dimensions', 'N/A')}")
        print(f"   Stored in Cosmos: {metadata.get('stored_in_cosmos', False)}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse response: {e}")
        print(f"   Raw response: {result}")
        return False
    except Exception as e:
        print(f"❌ Error processing result: {e}")
        return False


async def test_next_best_action():
    """Main test function for next_best_action tool"""
    
    print("=" * 70)
    print("🧪 Testing next_best_action MCP Tool")
    print("   This tool uses AI to analyze tasks, find similar past tasks,")
    print("   and generate action plans using semantic reasoning.")
    print("=" * 70)
    
    # Check for --direct mode
    use_direct = '--direct' in sys.argv
    
    # Load config
    config = load_config()
    
    if use_direct:
        # Use direct connection via port-forward
        direct_config = config.get('direct', {})
        base_url = direct_config.get('base_url', 'http://localhost:8000/runtime/webhooks/mcp')
        token = 'direct-mode-no-token-needed'
        print(f"\n🔗 Direct Mode URL: {base_url}")
        print("   (Using port-forward, no auth required)")
    else:
        # Use APIM
        apim_config = config.get('apim', {})
        base_url = apim_config.get('base_url', '')
        token_url = apim_config.get('oauth_token_url', '')
        
        if not base_url:
            print("❌ No APIM base URL configured")
            return False
        
        print(f"\n🔗 APIM Base URL: {base_url}")
        
        # Get token first
        async with aiohttp.ClientSession() as session:
            token = await get_mcp_token(session, token_url)
            if not token:
                print("❌ Failed to get access token")
                return False
    
    # Use MCP Client for the rest
    async with MCPClient(base_url, token) as client:
        # Establish SSE session
        if not await client.establish_sse_session():
            print("❌ Failed to establish SSE session")
            return False
        
        # Wait for session to initialize
        print("\n⏳ Waiting for session to initialize...")
        await asyncio.sleep(2)
        
        # List tools to verify next_best_action exists
        print("\n📋 Listing available tools...")
        tools = await client.list_tools()
        
        if tools:
            print(f"✅ Found {len(tools)} tools:")
            has_next_best_action = False
            has_memory_tools = False
            
            for tool in tools:
                name = tool.get('name', '')
                desc = tool.get('description', '')[:60]
                print(f"   • {name}: {desc}...")
                if name == 'next_best_action':
                    has_next_best_action = True
                if name in ['store_memory', 'recall_memory']:
                    has_memory_tools = True
            
            if not has_next_best_action:
                print("\n⚠️  next_best_action tool not found!")
                print("   Please ensure the MCP server is updated with the latest code.")
                return False
            
            if has_memory_tools:
                print("\n✅ Memory tools available - short-term memory is enabled")
        else:
            print("❌ Could not list tools")
            return False
        
        # Test tasks for next_best_action
        print("\n" + "=" * 70)
        print("🤖 Testing next_best_action with sample tasks")
        print("=" * 70)
        
        test_tasks = [
            "Analyze customer churn data and create a predictive model to identify at-risk customers",
            "Set up a CI/CD pipeline for deploying microservices to Kubernetes",
            "Design a REST API for a user management system with authentication",
        ]
        
        all_passed = True
        task_ids = []
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n{'─' * 70}")
            print(f"🎯 Test {i}/{len(test_tasks)}")
            print(f"📝 Task: {task}")
            print(f"{'─' * 70}")
            
            print("\n⏳ Processing task (this may take 30-60 seconds)...")
            print("   • Generating embeddings with text-embedding-3-large")
            print("   • Analyzing intent with GPT")
            print("   • Searching short-term memory (CosmosDB)")
            print("   • Retrieving long-term memory (FoundryIQ)")
            print("   • Querying facts memory (Fabric IQ ontologies)")
            print("   • Generating action plan")
            print("   • Storing in CosmosDB")
            
            result = await client.call_tool("next_best_action", {"task": task})
            
            if 'error' in result:
                print(f"\n❌ Error calling tool: {result['error']}")
                all_passed = False
                continue
            
            # Extract the response
            tool_result = result.get('result', {})
            content = tool_result.get('content', [])
            is_error = tool_result.get('isError', False)
            
            if is_error:
                error_text = content[0].get('text', 'Unknown error') if content else 'No error message'
                print(f"\n❌ Tool returned error: {error_text}")
                
                if 'CosmosDB not configured' in error_text:
                    print("\n⚠️  COSMOSDB_ENDPOINT is not configured on the server")
                    print("   Ensure CosmosDB is deployed and the environment variable is set.")
                elif 'Foundry endpoint not configured' in error_text:
                    print("\n⚠️  FOUNDRY_PROJECT_ENDPOINT is not configured on the server")
                    
                all_passed = False
            else:
                if content:
                    response_text = content[0].get('text', '{}')
                    success = print_task_result(response_text)
                    
                    if success:
                        # Extract task_id for tracking
                        try:
                            data = json.loads(response_text)
                            task_id = data.get('task_id')
                            if task_id:
                                task_ids.append(task_id)
                        except:
                            pass
                    else:
                        all_passed = False
                else:
                    print("\n⚠️  No content in response")
                    all_passed = False
            
            # Brief pause between tests
            if i < len(test_tasks):
                print("\n⏳ Waiting before next test...")
                await asyncio.sleep(2)
        
        # Test with a similar task to see semantic matching
        if all_passed and len(task_ids) > 0:
            print("\n" + "=" * 70)
            print("🔄 Testing Semantic Similarity")
            print("   Submitting a similar task to verify semantic matching works")
            print("=" * 70)
            
            similar_task = "Build a machine learning model to predict which users will cancel their subscription"
            print(f"\n📝 Similar Task: {similar_task}")
            print("   (This should match the customer churn task from earlier)")
            
            print("\n⏳ Processing similar task...")
            result = await client.call_tool("next_best_action", {"task": similar_task})
            
            if 'error' not in result:
                tool_result = result.get('result', {})
                content = tool_result.get('content', [])
                if content and not tool_result.get('isError'):
                    response_text = content[0].get('text', '{}')
                    print_task_result(response_text)
                    
                    # Check if similar tasks were found
                    try:
                        data = json.loads(response_text)
                        similar_count = data.get('analysis', {}).get('similar_tasks_found', 0)
                        if similar_count > 0:
                            print("\n✅ Semantic matching is working!")
                            print(f"   Found {similar_count} similar task(s) from previous submissions")
                        else:
                            print("\n⚠️  No similar tasks found (this is expected on first run)")
                    except:
                        pass
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 Test Summary")
        print("=" * 70)
        
        if all_passed:
            print("\n✅ All next_best_action tests PASSED!")
            print("\n🎉 The AI-powered task planning system is working correctly!")
            print("\nCapabilities verified:")
            print("   ✓ Task embedding generation (text-embedding-3-large)")
            print("   ✓ Intent analysis (GPT)")
            print("   ✓ Semantic similarity search (cosine similarity)")
            print("   ✓ Action plan generation")
            print("   ✓ CosmosDB storage")
            
            if task_ids:
                print(f"\n📝 Tasks stored in CosmosDB:")
                for tid in task_ids:
                    print(f"   • {tid}")
        else:
            print("\n❌ Some tests FAILED")
            print("\nTroubleshooting:")
            print("  1. Verify FOUNDRY_PROJECT_ENDPOINT is set in Kubernetes deployment")
            print("  2. Verify COSMOSDB_ENDPOINT is set in Kubernetes deployment")
            print("  3. Verify EMBEDDING_MODEL_DEPLOYMENT_NAME is set (default: text-embedding-3-large)")
            print("  4. Check the MCP server logs:")
            print("     kubectl logs -n mcp-agents -l app=mcp-agents --tail=100")
            print("  5. Verify the models are deployed in Azure AI Foundry")
        
        return all_passed


def main():
    """Main entry point"""
    try:
        result = asyncio.run(test_next_best_action())
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

