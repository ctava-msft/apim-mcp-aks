#!/usr/bin/env python3
"""
Test script for Fabric Data Agents MCP tools.

This script tests all Fabric Data Agent tools:
1. fabric_query_lakehouse - Execute Spark SQL queries
2. fabric_query_warehouse - Execute T-SQL queries
3. fabric_trigger_pipeline - Trigger data pipeline execution
4. fabric_get_pipeline_status - Get pipeline run status
5. fabric_query_semantic_model - Query semantic models via DAX/MDX
6. fabric_list_resources - List available Fabric resources

Usage:
    python test_fabric_agents.py

Requirements:
    - aiohttp (pip install aiohttp)
    - MCP server running with Fabric Data Agents enabled
    - Valid OAuth token or Azure CLI configured
    - Fabric workspace with lakehouses, warehouses, pipelines, and semantic models
"""

import asyncio
import json
import aiohttp
import sys
import os
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
        print("⚠️  Using direct mode (assumes local MCP server at http://localhost:8000)")
        return {
            "mcp_server_url": "http://localhost:8000",
            "fabric": {
                "workspace_id": os.getenv("FABRIC_WORKSPACE_ID", ""),
                "lakehouse_id": os.getenv("FABRIC_LAKEHOUSE_ID", ""),
                "warehouse_id": os.getenv("FABRIC_WAREHOUSE_ID", ""),
                "pipeline_id": os.getenv("FABRIC_PIPELINE_ID", ""),
                "dataset_id": os.getenv("FABRIC_DATASET_ID", "")
            }
        }


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
            self.sse_response = await self.session.get(f"{self.base_url}/sse", timeout=None)
            
            if self.sse_response.status != 200:
                print(f"❌ Failed to establish SSE session: HTTP {self.sse_response.status}")
                return False
            
            async for line in self.sse_response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data:'):
                    data = json.loads(line[5:].strip())
                    if 'endpoint' in data:
                        self.session_message_url = f"{self.base_url}{data['endpoint']}"
                        print(f"✅ SSE session established: {self.session_message_url}")
                        return True
            
            return False
        except Exception as e:
            print(f"❌ Error establishing SSE session: {e}")
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call an MCP tool"""
        if not self.session_message_url:
            print("❌ No active SSE session")
            return None
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        print(f"\n🔧 Calling tool: {tool_name}")
        print(f"📋 Arguments: {json.dumps(arguments, indent=2)}")
        
        async with self.session.post(
            self.session_message_url,
            json=request,
            headers={'Content-Type': 'application/json'}
        ) as response:
            if response.status != 200:
                print(f"❌ Tool call failed: HTTP {response.status}")
                return None
            
            result = await response.json()
            
            if 'error' in result:
                print(f"❌ Tool error: {result['error']}")
                return None
            
            if 'result' in result:
                print(f"✅ Tool executed successfully")
                return result['result']
            
            return result


async def test_fabric_list_resources(client: MCPClient):
    """Test listing Fabric resources"""
    print("\n" + "="*70)
    print("TEST 1: List Fabric Resources")
    print("="*70)
    
    result = await client.call_tool("fabric_list_resources", {
        "resource_type": "all"
    })
    
    if result and 'content' in result:
        content_text = result['content'][0].get('text', '')
        data = json.loads(content_text)
        
        if data.get('success'):
            resources = data.get('resources', {})
            print(f"\n📊 Fabric Resources Summary:")
            print(f"   Workspace ID: {data.get('workspace_id', 'N/A')}")
            print(f"   Lakehouses: {len(resources.get('lakehouses', []))}")
            print(f"   Warehouses: {len(resources.get('warehouses', []))}")
            print(f"   Pipelines: {len(resources.get('pipelines', []))}")
            print(f"   Semantic Models: {len(resources.get('semantic_models', []))}")
            
            return resources
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
    
    return None


async def test_fabric_query_lakehouse(client: MCPClient, lakehouse_id: str):
    """Test querying a Fabric Lakehouse"""
    print("\n" + "="*70)
    print("TEST 2: Query Fabric Lakehouse")
    print("="*70)
    
    if not lakehouse_id:
        print("⚠️  Skipping: No lakehouse_id configured")
        return
    
    # Using explicit columns for better query performance and schema stability
    query = "SELECT customer_id, name, email, churn_risk, segment FROM customers LIMIT 10"
    
    result = await client.call_tool("fabric_query_lakehouse", {
        "lakehouse_id": lakehouse_id,
        "query": query,
        "lakehouse_name": "test-lakehouse"
    })
    
    if result and 'content' in result:
        content_text = result['content'][0].get('text', '')
        data = json.loads(content_text)
        
        if data.get('success'):
            print(f"\n✅ Lakehouse query successful")
            print(f"   Query: {query}")
            print(f"   Results: {json.dumps(data.get('results', {}), indent=2)[:500]}...")
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")


async def test_fabric_query_warehouse(client: MCPClient, warehouse_id: str):
    """Test querying a Fabric Data Warehouse"""
    print("\n" + "="*70)
    print("TEST 3: Query Fabric Data Warehouse")
    print("="*70)
    
    if not warehouse_id:
        print("⚠️  Skipping: No warehouse_id configured")
        return
    
    # Using explicit columns for better query performance and schema stability
    query = "SELECT TOP 10 sale_id, region, revenue, customer_id FROM sales ORDER BY revenue DESC"
    
    result = await client.call_tool("fabric_query_warehouse", {
        "warehouse_id": warehouse_id,
        "query": query,
        "warehouse_name": "test-warehouse"
    })
    
    if result and 'content' in result:
        content_text = result['content'][0].get('text', '')
        data = json.loads(content_text)
        
        if data.get('success'):
            print(f"\n✅ Warehouse query successful")
            print(f"   Query: {query}")
            print(f"   Results: {json.dumps(data.get('results', {}), indent=2)[:500]}...")
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")


async def test_fabric_trigger_pipeline(client: MCPClient, pipeline_id: str):
    """Test triggering a Fabric Data Pipeline"""
    print("\n" + "="*70)
    print("TEST 4: Trigger Fabric Data Pipeline")
    print("="*70)
    
    if not pipeline_id:
        print("⚠️  Skipping: No pipeline_id configured")
        return None
    
    result = await client.call_tool("fabric_trigger_pipeline", {
        "pipeline_id": pipeline_id,
        "pipeline_name": "test-pipeline",
        "parameters": json.dumps({"test_run": True})
    })
    
    if result and 'content' in result:
        content_text = result['content'][0].get('text', '')
        data = json.loads(content_text)
        
        if data.get('success'):
            run_id = data.get('run_id')
            print(f"\n✅ Pipeline triggered successfully")
            print(f"   Pipeline ID: {pipeline_id}")
            print(f"   Run ID: {run_id}")
            print(f"   Status: {data.get('status', 'Unknown')}")
            return run_id
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
    
    return None


async def test_fabric_get_pipeline_status(client: MCPClient, pipeline_id: str, run_id: str):
    """Test getting pipeline status"""
    print("\n" + "="*70)
    print("TEST 5: Get Fabric Pipeline Status")
    print("="*70)
    
    if not pipeline_id or not run_id:
        print("⚠️  Skipping: No pipeline_id or run_id available")
        return
    
    result = await client.call_tool("fabric_get_pipeline_status", {
        "pipeline_id": pipeline_id,
        "run_id": run_id,
        "pipeline_name": "test-pipeline"
    })
    
    if result and 'content' in result:
        content_text = result['content'][0].get('text', '')
        data = json.loads(content_text)
        
        if data.get('success'):
            print(f"\n✅ Pipeline status retrieved")
            print(f"   Pipeline ID: {pipeline_id}")
            print(f"   Run ID: {run_id}")
            print(f"   Status: {data.get('status', 'Unknown')}")
            print(f"   Start Time: {data.get('start_time', 'N/A')}")
            print(f"   End Time: {data.get('end_time', 'N/A')}")
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")


async def test_fabric_query_semantic_model(client: MCPClient, dataset_id: str):
    """Test querying a semantic model"""
    print("\n" + "="*70)
    print("TEST 6: Query Semantic Model (Power BI)")
    print("="*70)
    
    if not dataset_id:
        print("⚠️  Skipping: No dataset_id configured")
        return
    
    # Example DAX query
    query = "EVALUATE TOPN(10, Customer, [TotalSales], DESC)"
    
    result = await client.call_tool("fabric_query_semantic_model", {
        "dataset_id": dataset_id,
        "query": query,
        "dataset_name": "test-semantic-model",
        "query_language": "DAX"
    })
    
    if result and 'content' in result:
        content_text = result['content'][0].get('text', '')
        data = json.loads(content_text)
        
        if data.get('success'):
            print(f"\n✅ Semantic model query successful")
            print(f"   Query: {query}")
            print(f"   Language: {data.get('query_language', 'DAX')}")
            print(f"   Results: {json.dumps(data.get('results', {}), indent=2)[:500]}...")
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")


async def main():
    """Main test runner"""
    print("=" * 70)
    print("Fabric Data Agents - MCP Tools Test Suite")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    
    mcp_server_url = config.get('mcp_server_url', 'http://localhost:8000')
    auth_token = config.get('auth_token', 'direct-mode')
    fabric_config = config.get('fabric', {})
    
    print(f"\n🔗 MCP Server URL: {mcp_server_url}")
    print(f"🔐 Auth Mode: {'OAuth Token' if not auth_token.startswith('direct-mode') else 'Direct (No Auth)'}")
    
    if not fabric_config.get('workspace_id'):
        print("\n⚠️  WARNING: Fabric workspace not configured")
        print("   Set FABRIC_WORKSPACE_ID and resource IDs in config or environment")
        print("   Tests will demonstrate tool interfaces but may not execute fully")
    
    async with MCPClient(mcp_server_url, auth_token) as client:
        # Establish SSE session
        if not await client.establish_sse_session():
            print("❌ Failed to establish SSE session. Exiting.")
            sys.exit(1)
        
        # Run tests
        resources = await test_fabric_list_resources(client)
        
        # Extract resource IDs from list if not provided
        lakehouse_id = fabric_config.get('lakehouse_id')
        warehouse_id = fabric_config.get('warehouse_id')
        pipeline_id = fabric_config.get('pipeline_id')
        dataset_id = fabric_config.get('dataset_id')
        
        if resources and not lakehouse_id:
            lakehouses = resources.get('lakehouses', [])
            if lakehouses:
                lakehouse_id = lakehouses[0].get('id')
        
        if resources and not warehouse_id:
            warehouses = resources.get('warehouses', [])
            if warehouses:
                warehouse_id = warehouses[0].get('id')
        
        if resources and not pipeline_id:
            pipelines = resources.get('pipelines', [])
            if pipelines:
                pipeline_id = pipelines[0].get('id')
        
        if resources and not dataset_id:
            datasets = resources.get('semantic_models', [])
            if datasets:
                dataset_id = datasets[0].get('id')
        
        # Run individual tool tests
        await test_fabric_query_lakehouse(client, lakehouse_id)
        await test_fabric_query_warehouse(client, warehouse_id)
        
        # Pipeline tests (trigger and status check)
        run_id = await test_fabric_trigger_pipeline(client, pipeline_id)
        if run_id:
            # Wait a bit for pipeline to start
            await asyncio.sleep(2)
            await test_fabric_get_pipeline_status(client, pipeline_id, run_id)
        
        await test_fabric_query_semantic_model(client, dataset_id)
    
    print("\n" + "="*70)
    print("✅ All Fabric Data Agents tests completed")
    print("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
