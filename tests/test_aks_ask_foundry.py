#!/usr/bin/env python3
"""
Test script for the ask_foundry MCP tool.

This script tests the ask_foundry tool which calls Azure AI Foundry to get AI responses.

Usage:
    python test_ask_foundry.py

Requirements:
    - aiohttp (pip install aiohttp)
    - MCP server running with FOUNDRY_PROJECT_ENDPOINT configured
    - Valid OAuth token or Azure CLI configured
"""

import asyncio
import json
import aiohttp
import sys
import os
import pytest
import re
from pathlib import Path
from typing import Optional, Dict, Any

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
                headers={'Content-Type': 'application/json'}
            ) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        return {"error": "Invalid JSON response", "raw": response_text}
                else:
                    return {"error": f"HTTP {response.status}", "body": response_text}
                    
        except Exception as e:
            return {"error": str(e)}
    
    async def list_tools(self) -> Optional[list]:
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


@pytest.mark.asyncio
async def test_ask_foundry():
    """Main test function for ask_foundry tool"""
    
    print("=" * 60)
    print("🧪 Testing ask_foundry MCP Tool")
    print("=" * 60)
    
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
        print("⏳ Waiting for session to initialize...")
        await asyncio.sleep(2)
        
        # List tools to verify ask_foundry exists
        print("\n📋 Listing available tools...")
        tools = await client.list_tools()
        
        if tools:
            print(f"✅ Found {len(tools)} tools:")
            has_ask_foundry = False
            for tool in tools:
                name = tool.get('name', '')
                desc = tool.get('description', '')[:50]
                print(f"   • {name}: {desc}")
                if name == 'ask_foundry':
                    has_ask_foundry = True
            
            if not has_ask_foundry:
                print("\n⚠️  ask_foundry tool not found in available tools")
                print("   The MCP server may need to be updated to include this tool")
                return False
        else:
            print("❌ Could not list tools")
            return False
        
        # Test ask_foundry with questions
        print("\n" + "=" * 60)
        print("🤖 Testing ask_foundry tool")
        print("=" * 60)
        
        test_questions = [
            "What is 2 + 2?",
            "Say hello in French.",
        ]
        
        all_passed = True
        
        for question in test_questions:
            print(f"\n📝 Question: {question}")
            print("-" * 40)
            
            result = await client.call_tool("ask_foundry", {"question": question})
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                all_passed = False
                continue
            
            # Extract the response
            tool_result = result.get('result', {})
            content = tool_result.get('content', [])
            is_error = tool_result.get('isError', False)
            
            if is_error:
                error_text = content[0].get('text', 'Unknown error') if content else 'No error message'
                print(f"❌ Tool returned error: {error_text}")
                
                if 'Foundry endpoint not configured' in error_text:
                    print("\n⚠️  FOUNDRY_PROJECT_ENDPOINT is not configured on the server")
                    print("   Set this environment variable in the Kubernetes deployment:")
                    print("   kubectl set env deployment/mcp-agents -n mcp-agents FOUNDRY_PROJECT_ENDPOINT=<your-endpoint>")
                all_passed = False
            else:
                if content:
                    response_text = content[0].get('text', 'No response')
                    print(f"✅ Response from AI Foundry:")
                    print("-" * 40)
                    print(response_text)
                    print("-" * 40)
                else:
                    print("⚠️  No content in response")
                    all_passed = False
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)
        
        if all_passed:
            print("✅ All ask_foundry tests PASSED!")
            print("🎉 The AI Foundry integration is working correctly!")
        else:
            print("❌ Some tests FAILED")
            print("\nTroubleshooting:")
            print("  1. Verify FOUNDRY_PROJECT_ENDPOINT is set in Kubernetes deployment")
            print("  2. Check the MCP server logs: kubectl logs -n mcp-agents -l app=mcp-agents")
            print("  3. Ensure the Foundry model deployment exists and is accessible")
        
        return all_passed


def main():
    """Main entry point"""
    try:
        result = asyncio.run(test_ask_foundry())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

