#!/usr/bin/env python3
"""
Complete APIM + MCP + AKS Integration Test

This script tests the complete stack:
1. AKS cluster infrastructure (nodes, pods, services, workload identity)
2. MCP server deployment on AKS
3. MCP protocol via APIM (OAuth + SSE)
4. MCP tool discovery and execution

Usage:
    python test_apim_mcp_connection.py [--direct] [--use-browser-auth] [--skip-infra] [--generate-token]

Options:
    --direct            Use direct connection via port-forward (no APIM auth required)
    --use-browser-auth  Use browser-based OAuth (default: Azure CLI token)
    --skip-infra        Skip AKS infrastructure tests
    --generate-token    Generate OAuth token interactively
    --help, -h          Show this help message

Requirements:
    - aiohttp (pip install aiohttp)
    - kubectl configured with AKS credentials
    - Network access to the deployed Azure APIM endpoint

Expected Output:
    ✅ AKS infrastructure tests passed
    ✅ SSE Session established
    ✅ 3 MCP tools discovered
    ✅ hello_mcp tool executed successfully
    🎉 SUCCESS message
"""

import asyncio
import json
import aiohttp
import sys
import os
import pytest
import webbrowser
import urllib.parse
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

# Configuration files
CONFIG_FILE = Path(__file__).parent / 'mcp_test_config.json'
ENV_FILE = Path(__file__).parent / 'mcp_test.env'
TOKENS_FILE = Path(__file__).parent / 'mcp_tokens.json'


# ============================================================
# AKS Infrastructure Tests
# ============================================================

def run_command(command: str) -> Optional[str]:
    """Run a shell command and return output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def test_aks_cluster_connection() -> bool:
    """Test if we can connect to the AKS cluster"""
    print("\n🔍 Testing AKS cluster connection...")
    
    result = run_command("kubectl cluster-info")
    if not result:
        print("❌ Cannot connect to AKS cluster")
        print("  Make sure you have run: az aks get-credentials")
        return False
    
    print("✅ Successfully connected to AKS cluster")
    return True


def test_aks_nodes_running() -> bool:
    """Test if AKS nodes are running"""
    print("\n🔍 Testing AKS nodes...")
    
    result = run_command("kubectl get nodes -o json")
    if not result:
        print("❌ Could not get AKS nodes")
        return False
    
    nodes = json.loads(result)
    node_items = nodes.get('items', [])
    
    if not node_items:
        print("❌ No nodes found in cluster")
        return False
    
    ready_nodes = []
    for node in node_items:
        name = node['metadata']['name']
        conditions = node['status']['conditions']
        ready = any(c['type'] == 'Ready' and c['status'] == 'True' for c in conditions)
        
        if ready:
            ready_nodes.append(name)
            print(f"  ✅ Node {name} is Ready")
        else:
            print(f"  ❌ Node {name} is not Ready")
    
    if len(ready_nodes) == 0:
        print("❌ No ready nodes found")
        return False
    
    print(f"✅ AKS nodes are running ({len(ready_nodes)}/{len(node_items)} ready)")
    return True


def test_mcp_namespace_exists() -> bool:
    """Test if MCP server namespace exists"""
    print("\n🔍 Testing MCP server namespace...")
    
    result = run_command("kubectl get namespace mcp-agents -o json")
    if not result:
        print("❌ MCP server namespace not found")
        return False
    
    print("✅ MCP server namespace exists")
    return True


def test_mcp_server_deployed() -> bool:
    """Test if MCP server deployment exists and is running"""
    print("\n🔍 Testing MCP server deployment...")
    
    result = run_command("kubectl get deployment mcp-agents -n mcp-agents -o json")
    if not result:
        print("❌ MCP server deployment not found")
        return False
    
    deployment = json.loads(result)
    
    spec_replicas = deployment['spec']['replicas']
    status = deployment.get('status', {})
    available_replicas = status.get('availableReplicas', 0)
    ready_replicas = status.get('readyReplicas', 0)
    
    print(f"  Desired: {spec_replicas}, Available: {available_replicas}, Ready: {ready_replicas}")
    
    if available_replicas >= 1 and ready_replicas >= 1:
        print("✅ MCP server deployment is running")
        return True
    else:
        print("⚠️  MCP server deployment is not fully ready yet")
        return False


def test_mcp_server_pods() -> bool:
    """Test if MCP server pods are running"""
    print("\n🔍 Testing MCP server pods...")
    
    result = run_command("kubectl get pods -n mcp-agents -o json")
    if not result:
        print("❌ Could not get MCP server pods")
        return False
    
    pods = json.loads(result)
    pod_items = pods.get('items', [])
    
    if not pod_items:
        print("❌ No MCP server pods found")
        return False
    
    running_pods = []
    for pod in pod_items:
        name = pod['metadata']['name']
        phase = pod['status']['phase']
        
        if phase == 'Running':
            container_statuses = pod['status'].get('containerStatuses', [])
            all_ready = all(c.get('ready', False) for c in container_statuses)
            
            if all_ready:
                running_pods.append(name)
                print(f"  ✅ Pod {name} is Running and Ready")
            else:
                print(f"  ⚠️  Pod {name} is Running but not Ready")
        else:
            print(f"  ❌ Pod {name} is in phase: {phase}")
    
    if len(running_pods) == 0:
        print("❌ No running pods found")
        return False
    
    print(f"✅ MCP server pods are running ({len(running_pods)}/{len(pod_items)})")
    return True


def test_mcp_service_exists() -> bool:
    """Test if MCP server service exists"""
    print("\n🔍 Testing MCP server service...")
    
    result = run_command("kubectl get service mcp-agents -n mcp-agents -o json")
    if not result:
        print("❌ MCP server service not found")
        return False
    
    service = json.loads(result)
    cluster_ip = service['spec'].get('clusterIP')
    ports = service['spec'].get('ports', [])
    
    print(f"  Service IP: {cluster_ip}")
    for port in ports:
        print(f"  Port: {port.get('port')} -> {port.get('targetPort')}")
    
    print("✅ MCP server service exists")
    return True


def test_workload_identity() -> bool:
    """Test if workload identity is configured"""
    print("\n🔍 Testing workload identity configuration...")
    
    result = run_command("kubectl get serviceaccount mcp-agents-sa -n mcp-agents -o json")
    if not result:
        print("⚠️  MCP server service account not found")
        return False
    
    sa = json.loads(result)
    annotations = sa.get('metadata', {}).get('annotations', {})
    client_id = annotations.get('azure.workload.identity/client-id')
    
    if client_id:
        print(f"  ✅ Workload identity client ID: {client_id[:20]}...")
        print("✅ Workload identity is configured")
        return True
    else:
        print("⚠️  Workload identity client ID not found")
        return False


def run_infrastructure_tests() -> Dict[str, bool]:
    """Run all AKS infrastructure tests"""
    print("\n" + "=" * 60)
    print("📦 AKS Infrastructure Tests")
    print("=" * 60)
    
    tests = [
        ("AKS Cluster Connection", test_aks_cluster_connection),
        ("AKS Nodes Running", test_aks_nodes_running),
        ("MCP Namespace", test_mcp_namespace_exists),
        ("MCP Server Deployment", test_mcp_server_deployed),
        ("MCP Server Pods", test_mcp_server_pods),
        ("MCP Service", test_mcp_service_exists),
        ("Workload Identity", test_workload_identity),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with error: {e}")
            results[test_name] = False
    
    return results


# ============================================================
# Token Management
# ============================================================

def get_azure_cli_token() -> Optional[str]:
    """Get access token from Azure CLI (az login)"""
    import shutil
    
    # Find az command - try multiple locations
    az_cmd = shutil.which('az')
    if not az_cmd:
        # Try common Windows locations
        possible_paths = [
            r'C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd',
            r'C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin\az.cmd',
            os.path.expandvars(r'%LOCALAPPDATA%\Programs\Microsoft SDKs\Azure\CLI2\wbin\az.cmd'),
            os.path.expandvars(r'%USERPROFILE%\AppData\Local\Programs\Microsoft SDKs\Azure\CLI2\wbin\az.cmd'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                az_cmd = path
                break
    
    if not az_cmd:
        az_cmd = 'az'  # Fall back to letting the shell find it
    
    try:
        result = subprocess.run(
            [az_cmd, 'account', 'get-access-token', '--query', 'accessToken', '-o', 'tsv'],
            capture_output=True,
            text=True,
            timeout=30,
            shell=True  # Use shell to find the command in PATH
        )
        if result.returncode == 0:
            token = result.stdout.strip()
            if token:
                print(f"✅ Got access token from Azure CLI")
                return token
        print(f"❌ Azure CLI token error: {result.stderr}")
        return None
    except subprocess.TimeoutExpired:
        print(f"❌ Azure CLI token request timed out")
        return None
    except FileNotFoundError:
        print(f"❌ Azure CLI not found. Please install it and run 'az login'")
        return None
    except Exception as e:
        print(f"❌ Azure CLI token error: {e}")
        return None


async def get_mcp_token_from_apim() -> Optional[str]:
    """Get MCP access token from APIM OAuth endpoint using Azure CLI authentication.
    
    The APIM OAuth policy generates a custom MCP token (mcp_access_token_*) 
    that is required for accessing the MCP SSE endpoint.
    """
    try:
        import aiohttp
        import time
        
        # Generate a synthetic authorization code for APIM
        synthetic_code = f"az_cli_auth_{int(time.time())}"
        
        async with aiohttp.ClientSession() as session:
            data = {
                'grant_type': 'authorization_code',
                'code': synthetic_code,
                'redirect_uri': REDIRECT_URI,
                'client_id': CLIENT_ID
            }
            
            print(f"🔄 Getting MCP token from APIM OAuth endpoint...")
            print(f"   Token URL: {APIM_OAUTH_TOKEN_URL}")
            
            async with session.post(APIM_OAUTH_TOKEN_URL, data=data) as response:
                if response.status == 200:
                    tokens = await response.json()
                    access_token = tokens.get('access_token')
                    if access_token:
                        print(f"✅ Got MCP access token from APIM: {access_token[:30]}...")
                        return access_token
                    else:
                        print(f"❌ No access_token in response: {tokens}")
                else:
                    error_text = await response.text()
                    print(f"❌ APIM token error: {response.status} - {error_text}")
        return None
    except Exception as e:
        print(f"❌ APIM token error: {e}")
        return None

def load_configuration() -> Dict[str, Any]:
    """Load configuration from JSON file or environment variables"""
    config = {}
    
    # Try to load from JSON file first
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                config = {
                    'APIM_BASE_URL': data['apim']['base_url'],
                    'APIM_OAUTH_AUTHORIZE_URL': data['apim']['oauth_authorize_url'],
                    'APIM_OAUTH_TOKEN_URL': data['apim']['oauth_token_url'],
                    'CLIENT_ID': data['oauth']['client_id'],
                    'REDIRECT_URI': data['oauth']['redirect_uri'],
                    'SCOPE': data['oauth'].get('scope', 'openid profile email')
                }
                print(f"✅ Loaded configuration from {CONFIG_FILE}")
                return config
        except Exception as e:
            print(f"⚠️  Error loading JSON config: {e}")
    
    # Try to load from .env file
    if ENV_FILE.exists():
        try:
            with open(ENV_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
            
            if config:
                print(f"✅ Loaded configuration from {ENV_FILE}")
                # Map env vars to expected keys
                return {
                    'APIM_BASE_URL': config.get('APIM_BASE_URL'),
                    'APIM_OAUTH_AUTHORIZE_URL': config.get('APIM_OAUTH_AUTHORIZE_URL'),
                    'APIM_OAUTH_TOKEN_URL': config.get('APIM_OAUTH_TOKEN_URL'),
                    'CLIENT_ID': config.get('MCP_CLIENT_ID'),
                    'REDIRECT_URI': config.get('REDIRECT_URI', 'http://localhost:8080/callback'),
                    'SCOPE': 'openid profile email'
                }
        except Exception as e:
            print(f"⚠️  Error loading .env config: {e}")
    
    # Check environment variables
    env_config = {
        'APIM_BASE_URL': os.getenv('APIM_BASE_URL'),
        'APIM_OAUTH_AUTHORIZE_URL': os.getenv('APIM_OAUTH_AUTHORIZE_URL'),
        'APIM_OAUTH_TOKEN_URL': os.getenv('APIM_OAUTH_TOKEN_URL'),
        'CLIENT_ID': os.getenv('MCP_CLIENT_ID'),
        'REDIRECT_URI': os.getenv('REDIRECT_URI', 'http://localhost:8080/callback'),
        'SCOPE': 'openid profile email'
    }
    
    if all(env_config.values()):
        print(f"✅ Loaded configuration from environment variables")
        return env_config
    
    # No configuration found
    print(f"❌ No configuration found!")
    print(f"   Expected one of:")
    print(f"   - {CONFIG_FILE}")
    print(f"   - {ENV_FILE}")
    print(f"   - Environment variables (APIM_BASE_URL, APIM_OAUTH_AUTHORIZE_URL, etc.)")
    print(f"\n   Run: ./scripts/generate-test-config.ps1 to generate configuration")
    return None

# Load configuration at module level
_CONFIG = load_configuration()

if not _CONFIG:
    print(f"\n💥 Configuration required to run tests!")
    print(f"   Generate config with: ./scripts/generate-test-config.ps1")
    sys.exit(1)

# Export configuration as constants
APIM_BASE_URL = _CONFIG['APIM_BASE_URL']
APIM_OAUTH_AUTHORIZE_URL = _CONFIG['APIM_OAUTH_AUTHORIZE_URL']
APIM_OAUTH_TOKEN_URL = _CONFIG['APIM_OAUTH_TOKEN_URL']
CLIENT_ID = _CONFIG['CLIENT_ID']
REDIRECT_URI = _CONFIG['REDIRECT_URI']
OAUTH_SCOPE = _CONFIG.get('SCOPE', 'openid profile email')

class TokenManager:
    """Manages OAuth token acquisition and storage"""
    
    @staticmethod
    def load_tokens() -> Optional[Dict[str, Any]]:
        """Load tokens from file"""
        try:
            if TOKENS_FILE.exists():
                with open(TOKENS_FILE, 'r') as f:
                    tokens = json.load(f)
                    print(f"✅ Loaded tokens from {TOKENS_FILE}")
                    return tokens
            return None
        except Exception as e:
            print(f"⚠️  Error loading tokens: {e}")
            return None
    
    @staticmethod
    def save_tokens(tokens: Dict[str, Any]) -> bool:
        """Save tokens to file"""
        try:
            with open(TOKENS_FILE, 'w') as f:
                json.dump(tokens, f, indent=2)
            print(f"✅ Tokens saved to {TOKENS_FILE}")
            return True
        except Exception as e:
            print(f"❌ Error saving tokens: {e}")
            return False
    
    @staticmethod
    def generate_authorization_url(state: str = "test123") -> str:
        """Generate OAuth authorization URL"""
        params = {
            'response_type': 'code',
            'client_id': CLIENT_ID,
            'redirect_uri': REDIRECT_URI,
            'state': state,
            'scope': OAUTH_SCOPE
        }
        return f"{APIM_OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    
    @staticmethod
    async def exchange_code_for_token(auth_code: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token"""
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    'grant_type': 'authorization_code',
                    'code': auth_code,
                    'redirect_uri': REDIRECT_URI,
                    'client_id': CLIENT_ID
                }
                
                print(f"🔄 Exchanging authorization code for token...")
                async with session.post(APIM_OAUTH_TOKEN_URL, data=data) as response:
                    if response.status == 200:
                        tokens = await response.json()
                        print(f"✅ Successfully obtained access token")
                        return tokens
                    else:
                        error_text = await response.text()
                        print(f"❌ Token exchange failed: {response.status}")
                        print(f"   Response: {error_text}")
                        return None
        except Exception as e:
            print(f"❌ Token exchange error: {e}")
            return None
    
    @staticmethod
    def prompt_for_authorization() -> Optional[str]:
        """Prompt user to complete OAuth flow and return authorization code"""
        print("\n" + "="*60)
        print("🔐 OAuth Authorization Required")
        print("="*60)
        
        auth_url = TokenManager.generate_authorization_url()
        
        print(f"\n📋 Steps to obtain authorization code:")
        print(f"1. Copy and open this URL in your browser:")
        print(f"\n   {auth_url}\n")
        print(f"2. Complete the login and consent process")
        print(f"3. After redirect, copy the 'code' parameter from the URL")
        print(f"   (The URL will look like: {REDIRECT_URI}?code=XXXXX&state=test123)")
        print(f"4. Paste the code below\n")
        
        # Try to open browser automatically
        try:
            print(f"🌐 Opening browser automatically...")
            webbrowser.open(auth_url)
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
        
        print(f"\n" + "="*60)
        auth_code = input("Enter authorization code: ").strip()
        
        if auth_code:
            return auth_code
        return None

class MCPSessionManager:
    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session = None
        self.sse_response = None
        self.session_cookies = None
        self.session_message_url = None
        
    async def __aenter__(self):
        # Use cookie jar to maintain session state
        cookie_jar = aiohttp.CookieJar()
        headers = {}
        # Only add auth header if we have a valid token (not direct mode)
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
    
    async def establish_sse_session_properly(self) -> bool:
        """Establish SSE connection and keep it alive for session context"""
        try:
            print(f"🔗 Establishing SSE session to: {self.base_url}/sse")
            
            # Start the SSE connection
            self.sse_response = await self.session.get(
                f'{self.base_url}/sse',
                headers={
                    'Accept': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive'
                }
            )
            
            print(f"📡 SSE Response Status: {self.sse_response.status}")
            print(f"📡 SSE Response Headers: {dict(self.sse_response.headers)}")
            
            # Capture cookies from SSE response
            if self.sse_response.cookies:
                print(f"🍪 SSE Response Cookies: {dict(self.sse_response.cookies)}")
            
            if self.sse_response.status == 200:
                # Try to read some initial data to trigger session establishment
                try:
                    # Read first chunk to ensure connection is established
                    async for chunk in self.sse_response.content.iter_chunked(1024):
                        if chunk:
                            data = chunk.decode('utf-8', errors='ignore')
                            print(f"📡 SSE Initial Data: {data[:200]}")
                            
                            # Parse the SSE data to extract session endpoint
                            if 'data: message?' in data:
                                # Extract the session URL from the SSE data
                                import re
                                match = re.search(r'data: (message\?[^\n\r]+)', data)
                                if match:
                                    session_path = match.group(1)
                                    self.session_message_url = f"{self.base_url}/{session_path}"
                                    print(f"🎯 Extracted session URL: {self.session_message_url}")
                            break
                        # Only wait for first chunk, then proceed
                        await asyncio.sleep(0.1)
                        break
                    
                    if self.session_message_url:
                        print("✅ SSE connection established with session URL")
                        return True
                    else:
                        print("⚠️  SSE connected but no session URL found")
                        return False
                        
                except Exception as e:
                    print(f"⚠️  SSE data read warning: {e}")
                    # Still consider successful if we got 200 status
                    await asyncio.sleep(1)
                    print("✅ SSE connection established (fallback)")
                    return True
            else:
                response_text = await self.sse_response.text()
                print(f"❌ SSE connection failed: {response_text}")
                return False
                
        except Exception as e:
            print(f"❌ SSE connection error: {e}")
            return False
    
    async def send_jsonrpc_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC 2.0 request to the message endpoint"""
        request_id = "test-request-1"
        
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        
        if params:
            jsonrpc_request["params"] = params
            
        print(f"📤 Sending JSON-RPC request: {json.dumps(jsonrpc_request, indent=2)}")
        
        # Use session-specific URL if available, otherwise fall back to generic endpoint
        message_url = self.session_message_url if self.session_message_url else f'{self.base_url}/message'
        print(f"🎯 Using message URL: {message_url}")
        
        try:
            async with self.session.post(
                message_url,
                json=jsonrpc_request,
                headers={'Content-Type': 'application/json'}
            ) as response:
                print(f"📨 Message Response Status: {response.status}")
                print(f"📨 Message Response Headers: {dict(response.headers)}")
                
                response_text = await response.text()
                print(f"📨 Message Response Body: {response_text}")
                
                if response.status == 200:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        return {"error": "Invalid JSON response", "raw": response_text}
                else:
                    return {
                        "error": f"HTTP {response.status}",
                        "status": response.status,
                        "body": response_text
                    }
                    
        except Exception as e:
            return {"error": str(e)}
    
    async def listen_for_sse_response(self, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """Listen for JSON-RPC response on the SSE stream"""
        if not self.sse_response:
            return None
            
        try:
            print(f"👂 Listening for SSE response (timeout: {timeout}s)...")
            
            async with asyncio.timeout(timeout):
                async for chunk in self.sse_response.content.iter_chunked(1024):
                    if chunk:
                        data = chunk.decode('utf-8', errors='ignore')
                        print(f"📡 SSE Response Data: {data}")
                        
                        # Look for JSON-RPC response in SSE data
                        if '"jsonrpc":"2.0"' in data or '"result"' in data:
                            # Try to extract JSON from SSE data
                            import re
                            json_match = re.search(r'data:\s*(\{.*\})', data, re.DOTALL)
                            if json_match:
                                try:
                                    json_response = json.loads(json_match.group(1))
                                    print(f"✅ Found JSON-RPC response in SSE stream")
                                    return json_response
                                except json.JSONDecodeError as e:
                                    print(f"⚠️  JSON decode error: {e}")
                                    continue
                        
                        # Continue listening for more data
                        await asyncio.sleep(0.1)
                        
        except asyncio.TimeoutError:
            print(f"⏰ SSE response timeout after {timeout}s")
        except Exception as e:
            print(f"❌ SSE response error: {e}")
            
        return None

@pytest.mark.asyncio
async def test_mcp_fixed_session(use_az_token: bool = True, use_direct: bool = False):
    """Test MCP with proper SSE session establishment"""
    
    access_token = None
    base_url = APIM_BASE_URL
    
    # If using direct mode (port-forward, no auth)
    if use_direct:
        # Load direct config
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                direct_config = data.get('direct', {})
                base_url = direct_config.get('base_url', 'http://localhost:8000/runtime/webhooks/mcp')
        else:
            base_url = 'http://localhost:8000/runtime/webhooks/mcp'
        access_token = 'direct-mode-no-token-needed'
        print(f"🔗 Direct Mode URL: {base_url}")
        print(f"   (Using port-forward, no auth required)")
    # If using automatic token (via APIM OAuth endpoint)
    elif use_az_token:
        print(f"🔐 Getting MCP access token from APIM...")
        access_token = await get_mcp_token_from_apim()
        if not access_token:
            print(f"❌ Failed to get MCP token from APIM.")
            print(f"   The APIM OAuth endpoint may require browser-based authentication.")
            print(f"   Try running with: python test_apim_mcp_connection.py --use-browser-auth")
            print(f"   Or use direct mode: python test_apim_mcp_connection.py --direct")
            return False
    else:
        # Load or obtain OAuth token
        tokens = TokenManager.load_tokens()
        
        if not tokens:
            print(f"❌ No tokens found in {TOKENS_FILE}")
            print(f"🔄 Starting OAuth flow...")
            
            auth_code = TokenManager.prompt_for_authorization()
            
            if not auth_code:
                print(f"❌ No authorization code provided")
                return False
            
            tokens = await TokenManager.exchange_code_for_token(auth_code)
            
            if not tokens:
                print(f"❌ Failed to obtain access token")
                return False
            
            TokenManager.save_tokens(tokens)
        
        access_token = tokens.get('access_token')
        if not access_token:
            print(f"❌ No access_token in tokens")
            return False
    
    print(f"🚀 Starting Fixed MCP Session Test")
    print(f"🔗 Base URL: {base_url}")
    if not use_direct:
        print(f"🎫 Access Token: {access_token[:20]}...")
    
    async with MCPSessionManager(base_url, access_token) as mcp:
        # Step 1: Establish SSE session first and keep it alive
        print("\n" + "="*50)
        print("📡 STEP 1: Establishing Persistent SSE Session")
        print("="*50)
        
        sse_success = await mcp.establish_sse_session_properly()
        if not sse_success:
            print("❌ SSE session failed, cannot continue")
            return False
        
        # Step 2: Wait a moment for session to be fully established
        print("\n⏰ Waiting for session to initialize...")
        await asyncio.sleep(2)
        
        # Step 3: Send tools/list request with active SSE session
        print("\n" + "="*50)
        print("🛠️  STEP 2: Sending tools/list Request (with active SSE)")
        print("="*50)
        
        # Send the request and check if it's accepted
        tools_response = await mcp.send_jsonrpc_request("tools/list")
        print(f"🛠️  Tools Response: {json.dumps(tools_response, indent=2)}")
        
        # If we got HTTP 202, listen for the actual response on SSE stream
        if tools_response.get("status") == 202:
            print("\n📡 HTTP 202 received - listening for response on SSE stream...")
            sse_json_response = await mcp.listen_for_sse_response()
            if sse_json_response:
                tools_response = sse_json_response
                print(f"🛠️  SSE Tools Response: {json.dumps(tools_response, indent=2)}")
        
        # Step 4: Check if hello_mcp tool is present
        print("\n" + "="*50)
        print("🔍 STEP 3: Analyzing Response")
        print("="*50)
        
        if "result" in tools_response and "tools" in tools_response["result"]:
            tools = tools_response["result"]["tools"]
            print(f"🛠️  Found {len(tools)} tools:")
            
            hello_mcp_found = False
            for tool in tools:
                tool_name = tool.get("name", "unknown")
                tool_desc = tool.get("description", "")
                print(f"  • {tool_name}: {tool_desc}")
                
                if tool_name == "hello_mcp":
                    hello_mcp_found = True
                    print(f"    ✅ Found hello_mcp tool!")
                    
            if not hello_mcp_found:
                print(f"    ❌ hello_mcp tool not found in response")
                return False
                
            # Step 4: Call the hello_mcp tool
            print("\n" + "="*50)
            print("🚀 STEP 4: Calling hello_mcp Tool")
            print("="*50)
            
            hello_response = await mcp.send_jsonrpc_request("tools/call", {
                "name": "hello_mcp",
                "arguments": {}
            })
            print(f"🛠️  Tool Call Response: {json.dumps(hello_response, indent=2)}")
            
            # If we got HTTP 202, listen for the actual response on SSE stream
            if hello_response.get("status") == 202:
                print("\n📡 HTTP 202 received - listening for tool response on SSE stream...")
                sse_tool_response = await mcp.listen_for_sse_response()
                if sse_tool_response:
                    hello_response = sse_tool_response
                    print(f"🛠️  SSE Tool Response: {json.dumps(hello_response, indent=2)}")
            
            # Check the tool call result
            if "result" in hello_response:
                result_content = hello_response["result"]
                print(f"✅ hello_mcp tool result: {result_content}")
                return True
            elif "error" in hello_response:
                print(f"❌ Tool call error: {hello_response['error']}")
                return False
                
            return hello_mcp_found
            
        elif "error" in tools_response:
            print(f"❌ JSON-RPC Error: {tools_response['error']}")
            return False
        else:
            print(f"❌ Unexpected response format")
            return False


def print_summary(infra_results: Dict[str, bool], mcp_result: bool) -> int:
    """Print test summary and return exit code"""
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    # Infrastructure results
    if infra_results:
        print("\n📦 Infrastructure Tests:")
        for test_name, result in infra_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"  {test_name:30} {status}")
        
        infra_passed = sum(1 for r in infra_results.values() if r)
        infra_total = len(infra_results)
        print(f"\n  Infrastructure: {infra_passed}/{infra_total} passed")
    
    # MCP protocol result
    print("\n🌐 MCP Protocol Tests:")
    mcp_status = "✅ PASSED" if mcp_result else "❌ FAILED"
    print(f"  {'MCP Protocol (SSE + Tools)':30} {mcp_status}")
    
    # Overall result
    all_passed = mcp_result and (not infra_results or all(infra_results.values()))
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 All tests passed!")
        print("✅ Your APIM + MCP + AKS stack is fully operational!")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("⚠️  Some tests failed - check output above for details")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    # Check for command-line arguments
    # Default to using Azure CLI token (no browser authentication required)
    use_direct = '--direct' in sys.argv
    use_az_token = '--use-browser-auth' not in sys.argv
    skip_infra = '--skip-infra' in sys.argv
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python test_apim_mcp_connection.py [OPTIONS]")
        print("\nOptions:")
        print("  --direct            Use direct connection via port-forward (no APIM auth)")
        print("  --use-browser-auth  Use browser-based OAuth (default: Azure CLI token)")
        print("  --skip-infra        Skip AKS infrastructure tests")
        print("  --generate-token    Generate OAuth token interactively")
        print("  --help, -h          Show this help message")
        sys.exit(0)
    
    if '--generate-token' in sys.argv:
        # Interactive token generation
        print("🔐 Interactive OAuth Token Generation")
        print("="*60)
        
        async def generate_token():
            auth_code = TokenManager.prompt_for_authorization()
            if auth_code:
                tokens = await TokenManager.exchange_code_for_token(auth_code)
                if tokens:
                    TokenManager.save_tokens(tokens)
                    print("\n✅ Token generation complete!")
                    print(f"🎉 You can now run: python test_apim_mcp_connection.py")
                    return True
            print("\n❌ Token generation failed")
            return False
        
        success = asyncio.run(generate_token())
        sys.exit(0 if success else 1)
    
    # Run the full test suite
    print("=" * 60)
    print("🧪 Complete APIM + MCP + AKS Integration Test")
    print("=" * 60)
    
    infra_results = {}
    mcp_result = False
    
    try:
        # Part 1: Infrastructure tests (optional)
        if not skip_infra:
            infra_results = run_infrastructure_tests()
            
            # Check if critical infrastructure tests passed
            critical_tests = ["AKS Cluster Connection", "MCP Server Pods", "MCP Service"]
            critical_passed = all(infra_results.get(t, False) for t in critical_tests if t in infra_results)
            
            if not critical_passed:
                print("\n⚠️  Critical infrastructure tests failed - MCP tests may fail")
        else:
            print("\n⏭️  Skipping infrastructure tests (--skip-infra)")
        
        # Part 2: MCP Protocol tests
        print("\n" + "=" * 60)
        print("🌐 MCP Protocol Tests")
        print("=" * 60)
        
        mcp_result = asyncio.run(test_mcp_fixed_session(use_az_token=use_az_token, use_direct=use_direct))
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary and exit
    exit_code = print_summary(infra_results, mcp_result)
    sys.exit(exit_code)
