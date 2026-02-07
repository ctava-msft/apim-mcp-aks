# Connecting External Agents to Azure Agents Control Plane via MCP

## Overview

This guide provides a practical, step-by-step approach for connecting external agents running outside Azure (on GCP, AWS, Databricks, or Salesforce) to the Azure Agents Control Plane using the Model Context Protocol (MCP). 

### Purpose of the Azure Agents Control Plane

The Azure Agents Control Plane provides **centralized governance, observability, identity, and compliance** for enterprise AI agents — regardless of where they execute. Azure acts as the system of record for:

- **API Governance** - All agent tool calls flow through Azure API Management (APIM) with consistent policies, rate limits, and routing
- **Identity & Access** - Every agent receives a Microsoft Entra ID identity with role-based access control (RBAC)
- **Observability** - Azure Monitor and Application Insights provide unified telemetry, tracing, and compliance proof
- **Security & Compliance** - Policy enforcement at the gateway layer before any model or agent execution
- **Secrets Management** - Azure Key Vault and managed identities enable keyless authentication patterns

The control plane enables **multi-cloud agent architectures** where agents execute on any platform but governance remains centralized in Azure.

### Why MCP for Cross-Cloud Agent Connectivity

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) standardizes how AI agents discover and invoke tools across heterogeneous environments:

- **Protocol Standardization** - JSON-RPC 2.0 based protocol for tool discovery, invocation, and streaming responses
- **Tool Catalog Abstraction** - Agents discover available tools dynamically via `tools/list` operations
- **Cross-Platform Compatibility** - MCP clients exist for multiple languages and frameworks
- **Server-Sent Events (SSE)** - Efficient streaming for long-running operations and real-time responses
- **OpenAPI Alignment** - MCP tool schemas map to OpenAPI specifications for API governance

By exposing Azure-managed capabilities through MCP servers fronted by APIM, external agents can securely access enterprise tools, data, and services while maintaining governance boundaries.

### High-Level Architecture Summary

The architecture separates the **Control Plane** (governance and orchestration) from the **Execution Plane** (where agents run):

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXECUTION PLANE                             │
│  (External Cloud - GCP, AWS, Databricks, Salesforce)              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  External Agent Runtime                                 │      │
│  │  • Agent Logic (Vertex AI, SageMaker, Databricks, etc) │      │
│  │  • MCP Client Library                                   │      │
│  │  • Identity Token (OAuth 2.0 Bearer)                    │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │ HTTPS + OAuth 2.0
                               │ MCP Protocol (JSON-RPC 2.0)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AZURE CONTROL PLANE                              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  Azure API Management (APIM)                            │      │
│  │  • OAuth 2.0 Token Validation                           │      │
│  │  • Rate Limiting & Throttling                           │      │
│  │  • Request/Response Policies                            │      │
│  │  • Observability (App Insights)                         │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  MCP Server Endpoints (AKS / Azure AI Foundry)          │      │
│  │  • /sse (Server-Sent Events)                            │      │
│  │  • /message (JSON-RPC)                                  │      │
│  │  • Tool Execution Logic                                 │      │
│  └─────────────────────────────────────────────────────────┘      │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │  Azure Backend Services                                 │      │
│  │  • Azure AI Foundry (Agent Service)                     │      │
│  │  • Cosmos DB (Memory)                                   │      │
│  │  • AI Search (Long-term Memory)                         │      │
│  │  • Fabric IQ (Facts & Ontology)                         │      │
│  │  • Storage (Artifacts)                                  │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Flow:**
1. External agent (on GCP/AWS/Databricks/Salesforce) acts as an MCP client
2. Agent obtains OAuth 2.0 token from Azure Entra ID
3. Agent initiates MCP connection to Azure APIM endpoint (HTTPS)
4. APIM validates token, applies policies, and routes to MCP server
5. MCP server (running on AKS or Azure AI Foundry) executes tools
6. Responses stream back via SSE or return via JSON-RPC
7. All requests logged, monitored, and governed by Azure

---

## Architecture Diagram

### Request/Response Flow

The following describes the end-to-end flow when an external agent invokes a tool through the Azure Agents Control Plane:

```
External Agent            Azure APIM           MCP Server         Azure Services
(MCP Client)          (Control Plane)      (AKS/Foundry)       (Backend)
     │                       │                   │                   │
     │ 1. Obtain OAuth Token │                   │                   │
     ├──────────────────────►│                   │                   │
     │   (Entra ID)          │                   │                   │
     │                       │                   │                   │
     │ 2. Connect /sse       │                   │                   │
     ├──────────────────────►│                   │                   │
     │   + Bearer Token      │                   │                   │
     │                       │                   │                   │
     │                       │ 3. Validate Token │                   │
     │                       │ 4. Rate Limit     │                   │
     │                       │ 5. Apply Policies │                   │
     │                       │                   │                   │
     │                       │ 6. Route Request  │                   │
     │                       ├──────────────────►│                   │
     │                       │                   │                   │
     │ 7. SSE Connection Open (200)              │                   │
     │◄──────────────────────┴───────────────────┤                   │
     │                                           │                   │
     │ 8. tools/list request                     │                   │
     ├──────────────────────────────────────────►│                   │
     │   (JSON-RPC 2.0)                          │                   │
     │                                           │                   │
     │                       9. Return Tool Catalog                  │
     │◄──────────────────────────────────────────┤                   │
     │   {tools: [...]}                          │                   │
     │                                           │                   │
     │ 10. tools/call request                    │                   │
     ├──────────────────────────────────────────►│                   │
     │   {name: "query_data", ...}               │                   │
     │                                           │                   │
     │                                           │ 11. Execute Tool  │
     │                                           ├──────────────────►│
     │                                           │  (Cosmos, Search, │
     │                                           │   Fabric, etc.)   │
     │                                           │                   │
     │                                           │ 12. Results       │
     │                                           │◄──────────────────┤
     │                                           │                   │
     │                       13. Stream Response (SSE)               │
     │◄──────────────────────────────────────────┤                   │
     │   data: {result: ...}                     │                   │
     │                                           │                   │
     │                       14. Telemetry to App Insights           │
     │                       ├───────────────────────────────────────►
     │                                           │                   │
```

### Governance Checkpoints

At each stage, Azure enforces governance:

| Checkpoint | Layer | Enforcement |
|------------|-------|-------------|
| **Authentication** | APIM | OAuth 2.0 token validation against Entra ID |
| **Authorization** | APIM | Scope validation (e.g., `next_best_action` scope) |
| **Rate Limiting** | APIM | Per-identity quotas and throttling |
| **Request Validation** | APIM | JSON schema validation, size limits |
| **Policy Execution** | APIM | Custom policies (content filtering, routing logic) |
| **Tool Authorization** | MCP Server | RBAC checks against Azure resources |
| **Observability** | App Insights | Request/response logging, distributed tracing |
| **Audit Trail** | Azure Monitor | Complete lifecycle audit for compliance |

---

## Prerequisites

### Azure Resources

Before external agents can connect, the following Azure resources must be deployed:

#### 1. Azure API Management (APIM)

- **Tier**: Developer or Premium (for production use)
- **OAuth 2.0 Configuration**:
  - OAuth authorization server configured with Entra ID
  - Token validation policy enabled
  - MCP API endpoints (`/sse`, `/message`) published
- **Backend Configuration**:
  - Backend service pointing to MCP server (AKS LoadBalancer or Azure AI Foundry endpoint)
- **Policy Configuration**:
  - Rate limiting policies per agent identity
  - CORS policies for browser-based agents (if applicable)
  - Request/response transformation policies

**Deployment**: Use the provided Bicep templates in `infra/core/apim.bicep` and `infra/core/apim-api.bicep`.

#### 2. Azure AI Foundry Agent Service

- **Project**: Azure AI Foundry project with agent service enabled
- **Agent Runtime**: Deployed agent with MCP tool support
- **Connections**: Configured connections to Cosmos DB, AI Search, Storage, Fabric

**Deployment**: See `infra/ai/` Bicep modules.

#### 3. Microsoft Entra ID (Agent Identity)

- **Agent Identity**: Entra ID application or service principal representing the external agent
- **OAuth Scopes**: Define custom scopes for tool access (e.g., `next_best_action`, `data_query`)
- **Token Configuration**:
  - Access token lifetime (recommended: 1 hour)
  - Refresh token enabled (for long-running agents)
- **Application Registration**:
  - Redirect URI: `https://your-external-agent/callback` (for authorization code flow)
  - OR Client credentials flow (for service-to-service scenarios)

**Guidance**: See [docs/AGENTS_IDENTITY_DESIGN.md](AGENTS_IDENTITY_DESIGN.md) for identity architecture.

#### 4. Networking (Optional but Recommended)

- **Private Endpoints**: For production scenarios, expose APIM via private endpoint
- **VNet Integration**: External agents may connect via ExpressRoute or VPN for additional security
- **Public Endpoint**: For development/testing, APIM public endpoint can be used with OAuth 2.0

**Deployment**: See `infra/core/networking.bicep`.

#### 5. Observability

- **Application Insights**: Configured for APIM and MCP server telemetry
- **Log Analytics Workspace**: Centralized logging for compliance
- **Azure Monitor Alerts**: Configured for rate limit violations, authentication failures

**Deployment**: See `infra/core/monitor.bicep`.

---

### External Cloud Prerequisites

#### Google Cloud Platform (GCP)

**Compute Environment:**
- **Vertex AI**: Agent running in Vertex AI Workbench or custom prediction endpoint
- **Cloud Run**: Containerized agent service
- **Compute Engine**: VM-based agent runtime

**Identity Requirements:**
- Service account with workload identity federation to Azure Entra ID (recommended)
- OR Client credentials (client ID + secret) stored in GCP Secret Manager

**Network Requirements:**
- Outbound HTTPS access to Azure APIM endpoint (e.g., `https://apim-xyz.azure-api.net`)
- DNS resolution for Azure public endpoints

**Libraries:**
- Python MCP client: `pip install mcp-client`
- OAuth 2.0 client: `google-auth`, `requests-oauthlib`

**Minimal Example:**
```python
# GCP Vertex AI Agent connecting to Azure MCP
from mcp import ClientSession
from google.auth import default
from google.auth.transport.requests import Request

# Obtain Azure token via workload identity federation
# NOTE: Replace 'apim-xyz.azure-api.net' with your actual APIM instance URL
credentials, project = default(scopes=["https://apim-xyz.azure-api.net/.default"])
credentials.refresh(Request())
azure_token = credentials.token

# Connect to Azure MCP endpoint
# NOTE: Replace with your actual APIM endpoint URL
async with ClientSession(
    "https://apim-xyz.azure-api.net/mcp/sse",
    headers={"Authorization": f"Bearer {azure_token}"}
) as session:
    # Discover tools
    tools = await session.list_tools()
    # Invoke tool
    result = await session.call_tool("query_customer_data", {"customer_id": "123"})
```

---

#### Amazon Web Services (AWS)

**Compute Environment:**
- **AWS Lambda**: Serverless agent function
- **Amazon SageMaker**: Agent running in SageMaker notebook or endpoint
- **ECS/EKS**: Containerized agent on Elastic Container Service or Kubernetes
- **EC2**: VM-based agent runtime

**Identity Requirements:**
- IAM role with federated access to Azure Entra ID (via OIDC or SAML)
- OR Client credentials stored in AWS Secrets Manager

**Network Requirements:**
- Outbound HTTPS access to Azure APIM endpoint
- VPC endpoint or NAT Gateway for private subnet deployments

**Libraries:**
- Python MCP client: `pip install mcp-client`
- OAuth 2.0 client: `boto3`, `requests-oauthlib`

**Minimal Example:**
```python
# AWS Lambda connecting to Azure MCP
import os
import json
import boto3
from mcp import ClientSession
import requests

# Retrieve Azure client credentials from Secrets Manager
secrets = boto3.client('secretsmanager')
secret_value = secrets.get_secret_value(SecretId='azure-agent-credentials')
credentials = json.loads(secret_value['SecretString'])

# Obtain OAuth token from Azure Entra ID
token_url = f"https://login.microsoftonline.com/{credentials['tenant_id']}/oauth2/v2.0/token"
token_data = {
    'client_id': credentials['client_id'],
    'client_secret': credentials['client_secret'],
    'scope': f"api://{credentials['apim_app_id']}/.default",
    'grant_type': 'client_credentials'
}
token_response = requests.post(token_url, data=token_data)
azure_token = token_response.json()['access_token']

# Connect to Azure MCP endpoint
async with ClientSession(
    os.environ['AZURE_MCP_ENDPOINT'],
    headers={"Authorization": f"Bearer {azure_token}"}
) as session:
    tools = await session.list_tools()
    result = await session.call_tool("analyze_sales_data", {"region": "US"})
```

**AWS Lambda Considerations:**
- Token refresh logic in Lambda function or Layer
- Connection pooling for efficient SSE connections
- Timeout configuration (Lambda max: 15 minutes)

---

#### Databricks

**Compute Environment:**
- **Databricks Notebook**: Interactive agent development
- **Databricks Job**: Scheduled agent execution
- **Databricks SQL Warehouse**: SQL-based agent queries
- **MLflow Model Serving**: Agent as a deployed model endpoint

**Identity Requirements:**
- Databricks service principal with Azure Entra ID federation (recommended)
- OR Personal access token (PAT) for development only
- Azure credentials managed via Databricks secrets

**Network Requirements:**
- Outbound HTTPS access from Databricks workspace to Azure APIM
- VNet peering (optional) for private connectivity

**Libraries:**
- Python MCP client: `%pip install mcp-client`
- OAuth 2.0 client: `requests`, `msal` (Microsoft Authentication Library)

**Minimal Example:**
```python
# Databricks Notebook connecting to Azure MCP
%pip install mcp-client msal

from mcp import ClientSession
from msal import ConfidentialClientApplication
# NOTE: dbutils is Databricks-specific and automatically available in notebooks/jobs

# Retrieve credentials from Databricks secrets using dbutils
client_id = dbutils.secrets.get(scope="azure-creds", key="client-id")
client_secret = dbutils.secrets.get(scope="azure-creds", key="client-secret")
tenant_id = dbutils.secrets.get(scope="azure-creds", key="tenant-id")

# Obtain Azure token
app = ConfidentialClientApplication(
    client_id,
    authority=f"https://login.microsoftonline.com/{tenant_id}",
    client_credential=client_secret
)
token_response = app.acquire_token_for_client(scopes=["api://azure-mcp/.default"])
azure_token = token_response['access_token']

# Connect to Azure MCP
async with ClientSession(
    "https://apim-xyz.azure-api.net/mcp/sse",
    headers={"Authorization": f"Bearer {azure_token}"}
) as session:
    tools = await session.list_tools()
    result = await session.call_tool("query_lakehouse", {"table": "sales_data"})
    
    # Return result to Databricks
    display(result)
```

**Databricks Job Considerations:**
- Store refresh tokens in Databricks secrets for long-running jobs
- Use Databricks workflows for orchestration
- Integrate with Delta Lake for results storage

---

#### Salesforce

**Integration Patterns:**

##### 1. Apex External Service
- Define external service in Salesforce Setup
- Configure named credential with OAuth 2.0 authentication
- Invoke MCP endpoint via `HttpRequest` in Apex code

**Identity Requirements:**
- Salesforce connected app with OAuth 2.0 JWT Bearer flow
- Azure Entra ID app registration for Salesforce federation

**Minimal Example:**
```apex
// Salesforce Apex code calling Azure MCP
public class AzureMCPService {
    
    public static void callMCPTool(String toolName, Map<String, Object> params) {
        // Build request
        HttpRequest req = new HttpRequest();
        req.setEndpoint('callout:AzureMCP/message');  // Named credential
        req.setMethod('POST');
        req.setHeader('Content-Type', 'application/json');
        
        // MCP JSON-RPC request
        Map<String, Object> rpcRequest = new Map<String, Object>{
            'jsonrpc' => '2.0',
            'method' => 'tools/call',
            'params' => new Map<String, Object>{
                'name' => toolName,
                'arguments' => params
            },
            'id' => String.valueOf(DateTime.now().getTime())
        };
        req.setBody(JSON.serialize(rpcRequest));
        
        // Send request
        Http http = new Http();
        HttpResponse res = http.send(req);
        
        // Process response
        if (res.getStatusCode() == 200) {
            Map<String, Object> result = (Map<String, Object>)JSON.deserializeUntyped(res.getBody());
            System.debug('MCP Result: ' + result.get('result'));
        } else {
            System.debug('Error: ' + res.getStatus());
        }
    }
}
```

**Named Credential Configuration:**
- **Authentication Protocol**: OAuth 2.0
- **Token Endpoint**: `https://login.microsoftonline.com/{tenant-id}/oauth2/v2.0/token`
- **Scope**: `api://azure-mcp/.default`

##### 2. Salesforce Flow + External Service
- Create Salesforce Flow with External Service action
- Configure MCP tool invocation as Flow step
- Use Flow variables to pass parameters

##### 3. Event-Driven Integration
- Salesforce Platform Event triggers Azure Logic App or Function
- Azure service acts as MCP client on behalf of Salesforce
- Results written back to Salesforce via REST API

**Salesforce Considerations:**
- Governor limits (max 100 callouts per transaction)
- Timeout limits (120 seconds for synchronous callouts)
- Use `@future(callout=true)` for asynchronous MCP calls
- Store tokens in Custom Settings or Custom Metadata

---

### Identity and Network Assumptions

#### Identity (No Secrets in Code)

**Recommended Pattern**: Federated Workload Identity
- External cloud identity (GCP Service Account, AWS IAM Role, Databricks Principal) federates to Azure Entra ID
- No long-lived secrets required
- Token exchange at runtime via OIDC or SAML

**Alternative Pattern**: OAuth 2.0 Client Credentials Flow
- Client ID + Secret stored in cloud-native secrets manager (GCP Secret Manager, AWS Secrets Manager, Databricks Secrets)
- Token obtained via `/token` endpoint before each MCP connection
- Token cached for lifetime duration (recommended: 1 hour)

**Prohibited Patterns**:
- ❌ Hard-coded credentials in code
- ❌ Secrets in environment variables without encryption
- ❌ Long-lived personal access tokens (PATs)

#### Network (Governed Endpoints)

**Production Pattern**: Private Connectivity
- Azure Private Link exposes APIM via private endpoint
- External cloud connects via ExpressRoute, VPN, or cloud interconnect (GCP Interconnect, AWS Direct Connect)
- No public internet exposure

**Development Pattern**: Public Endpoint with OAuth 2.0
- APIM public endpoint accessible over HTTPS
- OAuth 2.0 token validation enforces authentication
- Rate limiting and IP filtering optional

**Traffic Flow**:
- All traffic is HTTPS (TLS 1.2+)
- All requests include `Authorization: Bearer <token>` header
- All responses comply with MCP protocol (JSON-RPC 2.0)

---

## Connection Flow (Conceptual)

### How an External Agent Acts as an MCP Client

1. **Agent Initialization**:
   - External agent starts up in its native environment (GCP, AWS, Databricks, Salesforce)
   - Agent loads configuration (Azure APIM endpoint, OAuth credentials)

2. **Token Acquisition**:
   - Agent requests OAuth 2.0 access token from Azure Entra ID
   - Token includes scopes for specific MCP tools (e.g., `next_best_action`)
   - Token cached for reuse during token lifetime

3. **MCP Connection Establishment**:
   - Agent initiates HTTPS connection to Azure APIM `/sse` endpoint
   - `Authorization: Bearer <token>` header included
   - APIM validates token, applies policies, and routes to MCP server
   - MCP server returns 200 OK with `Content-Type: text/event-stream`

4. **Tool Discovery**:
   - Agent sends `tools/list` JSON-RPC request via `/message` endpoint
   - MCP server returns available tool catalog with schemas
   - Agent caches tool definitions for subsequent calls

5. **Tool Invocation**:
   - Agent sends `tools/call` JSON-RPC request with tool name and parameters
   - MCP server validates request, executes tool logic, and returns result
   - Responses may stream via SSE for long-running operations

6. **Connection Lifecycle**:
   - For long-lived agents, SSE connection remains open
   - For ephemeral agents (Lambda, Databricks jobs), connection opens per request
   - Token refresh handled automatically before expiration

### How Azure Exposes MCP Servers via APIM

1. **APIM Configuration**:
   - APIM acts as reverse proxy for MCP server endpoints
   - Backend service points to AKS LoadBalancer (ClusterIP + LoadBalancer service) or Azure AI Foundry endpoint
   - API definition includes:
     - `/sse` operation (GET) for Server-Sent Events
     - `/message` operation (POST) for JSON-RPC requests

2. **Policy Pipeline**:
   ```xml
   <policies>
       <inbound>
           <!-- Validate OAuth token -->
           <validate-jwt header-name="Authorization" failed-validation-httpcode="401">
               <openid-config url="https://login.microsoftonline.com/{tenant}/.well-known/openid-configuration" />
               <required-claims>
                   <claim name="scp" match="any">
                       <value>next_best_action</value>
                   </claim>
               </required-claims>
           </validate-jwt>
           
           <!-- Rate limiting per agent identity -->
           <rate-limit-by-key calls="100" renewal-period="60" counter-key="@(context.Request.Headers.GetValueOrDefault("Authorization"))" />
           
           <!-- Route to backend -->
           <set-backend-service base-url="http://mcp-service.default.svc.cluster.local" />
       </inbound>
       
       <outbound>
           <!-- Add observability headers -->
           <set-header name="X-Request-ID" exists-action="override">
               <value>@(context.RequestId)</value>
           </set-header>
       </outbound>
   </policies>
   ```

3. **Backend Routing**:
   - APIM forwards validated requests to MCP server
   - MCP server (FastAPI on AKS) handles MCP protocol
   - Server uses workload identity to access Azure services (Cosmos, Search, Storage)

### Request/Response Lifecycle and Governance Checkpoints

**Inbound Request Path**:
```
External Agent → APIM Gateway → APIM Policy (Auth) → APIM Policy (Rate Limit) 
                → APIM Backend Routing → AKS LoadBalancer → MCP Server Pod
                → Azure Services (Cosmos, Search, etc.)
```

**Governance Applied**:
- **APIM Gateway**: TLS termination, DDoS protection
- **APIM Policy (Auth)**: Token validation, scope verification
- **APIM Policy (Rate Limit)**: Per-identity quota enforcement
- **MCP Server**: RBAC checks, input validation, business logic
- **Azure Services**: Resource-level RBAC, network policies

**Outbound Response Path**:
```
Azure Services → MCP Server Pod → AKS LoadBalancer → APIM Backend 
               → APIM Policy (Transform) → APIM Gateway → External Agent
```

**Observability Injected**:
- **APIM**: Request/response logs, latency metrics, error rates
- **App Insights**: Distributed tracing with correlation IDs
- **Azure Monitor**: Resource-level telemetry, alerts

---

## Provider-Specific Notes

### GCP (Google Cloud Platform)

#### Use Case: Vertex AI-Hosted Agent Calling Azure MCP

**Scenario**: A Vertex AI agent orchestrates a workflow that requires access to enterprise data managed by Azure Cosmos DB and Azure AI Search.

**Architecture**:
```
Vertex AI Agent (Python) 
    ↓ MCP Client Library
    ↓ Workload Identity Federation (GCP Service Account ↔ Azure Entra ID)
    ↓ HTTPS + OAuth 2.0
Azure APIM (MCP Endpoint)
    ↓ Backend: AKS MCP Server
Azure Cosmos DB + AI Search
```

**Identity Setup**:
1. Create GCP Service Account: `vertex-agent@project.iam.gserviceaccount.com`
2. Create Azure Entra ID app registration: `gcp-vertex-agent`
3. Configure workload identity federation:
   - Azure: Add federated credential for GCP issuer
   - Issuer: `https://accounts.google.com`
   - Subject: `vertex-agent@project.iam.gserviceaccount.com`
4. Grant Azure app registration access to APIM scopes

**Code Pattern**:
```python
from google.auth import default, impersonated_credentials
from google.oauth2 import service_account
from mcp import ClientSession

# Use GCP service account to obtain Azure token
# NOTE: Replace 'apim-xyz.azure-api.net' and 'vertex-agent@project.iam.gserviceaccount.com' with your values
credentials, project = default()
target_credentials = impersonated_credentials.Credentials(
    source_credentials=credentials,
    target_principal='vertex-agent@project.iam.gserviceaccount.com',
    target_scopes=['https://apim-xyz.azure-api.net/.default']
)

# Get Azure token
azure_token = target_credentials.token

# Connect to Azure MCP
# NOTE: Replace with your actual APIM endpoint URL
async with ClientSession(
    "https://apim-xyz.azure-api.net/mcp/sse",
    headers={"Authorization": f"Bearer {azure_token}"}
) as session:
    result = await session.call_tool("query_cosmos", {"query": "..."})
```

**Networking Considerations**:
- Use Cloud NAT for consistent outbound IP (for APIM IP filtering)
- Configure VPC Service Controls for data governance
- Consider GCP Interconnect for private connectivity

**Monitoring**:
- GCP Cloud Logging captures agent-side logs
- Azure App Insights captures MCP server-side logs
- Correlation via `X-Request-ID` header

---

### AWS (Amazon Web Services)

#### Use Case: Container-Based Agent (ECS/Fargate)

**Scenario**: An ECS task running a containerized agent needs to invoke Azure MCP tools for data enrichment.

**Architecture**:
```
ECS Task (Container) 
    ↓ MCP Client Library
    ↓ IAM Role with Federated Access to Azure
    ↓ HTTPS + OAuth 2.0
Azure APIM (MCP Endpoint)
    ↓ Backend: AKS MCP Server
Azure Services
```

**Identity Setup**:
1. Create IAM role: `ecs-agent-role`
2. Create Azure Entra ID app registration: `aws-ecs-agent`
3. Configure OIDC federation:
   - Azure: Add federated credential for AWS issuer
   - Issuer: AWS account-specific OIDC provider
   - Subject: `system:serviceaccount:default:ecs-agent-role`
4. Attach IAM policy to ECS task role allowing `sts:AssumeRoleWithWebIdentity`

**Code Pattern**:
```python
import os
from mcp import ClientSession
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import requests

# Retrieve Azure credentials from environment variables
# NOTE: This is a simplified example. In production, environment variables should be
# populated by ECS task definition from AWS Secrets Manager using the 'secrets' field.
# See AWS documentation: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/specifying-sensitive-data-secrets.html
# 
# For a complete example of retrieving secrets directly from Secrets Manager at runtime,
# see the AWS Lambda example which demonstrates the proper SDK pattern (get_azure_token function).
import requests

# Obtain Azure token via federated credentials
# In production, use proper OIDC federation between AWS IAM and Azure Entra ID
token_url = f"https://login.microsoftonline.com/{os.environ['AZURE_TENANT_ID']}/oauth2/v2.0/token"
token_data = {
    'client_id': os.environ['AZURE_CLIENT_ID'],
    'client_secret': os.environ['AZURE_CLIENT_SECRET'],  # Injected from ECS task definition secret
    'scope': f"api://{os.environ['AZURE_APIM_APP_ID']}/.default",
    'grant_type': 'client_credentials'
}
token_response = requests.post(token_url, data=token_data)
azure_token = token_response.json()['access_token']

# Connect to Azure MCP
async with ClientSession(
    os.environ['AZURE_MCP_ENDPOINT'],
    headers={"Authorization": f"Bearer {azure_token}"}
) as session:
    result = await session.call_tool("fetch_user_profile", {"user_id": "123"})
```

**Networking Considerations**:
- Use VPC endpoints for AWS Secrets Manager (to store fallback credentials)
- Configure NAT Gateway for private subnet ECS tasks
- Use AWS PrivateLink + Azure Private Link for end-to-end private connectivity

**Monitoring**:
- CloudWatch Logs for agent-side telemetry
- X-Ray for distributed tracing (integrate with App Insights)

#### Use Case: Lambda-Based Agent

**Scenario**: AWS Lambda function invokes Azure MCP endpoint for real-time data analysis.

**Considerations**:
- Lambda cold starts: Cache OAuth token in Lambda environment or use Lambda Layers
- Connection pooling: Reuse SSE connections across invocations (if Lambda execution environment persists)
- Timeout: Lambda max 15 minutes; ensure MCP tool execution completes within window

**Code Pattern**:
```python
import os
import json
import boto3
import requests
from mcp import ClientSession

# Cache token and expiration across Lambda invocations
_cached_token = None
_token_expiry = 0

def is_token_expired(expiry_time):
    """Check if token is expired with 5 minute buffer"""
    import time
    return time.time() >= expiry_time - 300

def get_azure_token():
    """Retrieve Azure OAuth token using client credentials"""
    # Get credentials from Secrets Manager
    secrets = boto3.client('secretsmanager')
    secret_value = secrets.get_secret_value(SecretId='azure-agent-credentials')
    credentials = json.loads(secret_value['SecretString'])
    
    # Obtain token
    token_url = f"https://login.microsoftonline.com/{credentials['tenant_id']}/oauth2/v2.0/token"
    token_data = {
        'client_id': credentials['client_id'],
        'client_secret': credentials['client_secret'],
        'scope': f"api://{credentials['apim_app_id']}/.default",
        'grant_type': 'client_credentials'
    }
    response = requests.post(token_url, data=token_data)
    token_info = response.json()
    return token_info['access_token'], token_info['expires_in']

async def lambda_handler(event, context):
    global _cached_token, _token_expiry
    
    # Reuse token if valid
    if _cached_token is None or is_token_expired(_token_expiry):
        _cached_token, expires_in = get_azure_token()
        import time
        _token_expiry = time.time() + expires_in
    
    # Connect to Azure MCP
    async with ClientSession(
        os.environ['AZURE_MCP_ENDPOINT'],
        headers={"Authorization": f"Bearer {_cached_token}"}
    ) as session:
        result = await session.call_tool("process_event", event)
        return {"statusCode": 200, "body": json.dumps(result)}
```

---

### Databricks

#### Use Case: Notebook-Based Agent

**Scenario**: Data scientist develops agent in Databricks notebook that queries Azure-managed data sources.

**Architecture**:
```
Databricks Notebook (Python) 
    ↓ MCP Client Library
    ↓ Service Principal (stored in Databricks Secrets)
    ↓ HTTPS + OAuth 2.0
Azure APIM (MCP Endpoint)
    ↓ Backend: AKS MCP Server
Azure Cosmos DB / AI Search / Fabric IQ
```

**Identity Setup**:
1. Create Databricks service principal in Azure Entra ID
2. Store credentials in Databricks secrets:
   ```bash
   databricks secrets create-scope --scope azure-creds
   databricks secrets put --scope azure-creds --key client-id
   databricks secrets put --scope azure-creds --key client-secret
   databricks secrets put --scope azure-creds --key tenant-id
   ```
3. Grant service principal access to APIM scopes

**Code Pattern** (see earlier Databricks example):
- Use `msal` library for token acquisition
- Use `mcp-client` for protocol implementation
- Store results in Delta Lake for downstream consumption

**Integration with Databricks Features**:
- **Databricks Workflows**: Schedule notebook runs that invoke MCP tools
- **Delta Live Tables**: Use MCP tools as data sources in DLT pipelines
- **MLflow**: Log MCP tool invocations as part of model training experiments

**Networking Considerations**:
- Databricks workspace must have outbound HTTPS access to Azure APIM
- Use VNet peering for private connectivity (Databricks VNet ↔ Azure VNet)

#### Use Case: Databricks Job-Based Agent

**Scenario**: Scheduled Databricks job runs agent logic that requires Azure MCP tools.

**Considerations**:
- Token refresh: Job may run for hours; implement token refresh logic
- Error handling: Retry logic for transient MCP failures
- Results storage: Write outputs to Delta Lake or Azure Storage

**Code Pattern**:
```python
# Databricks Job script
from mcp import ClientSession
from msal import ConfidentialClientApplication
import asyncio
import time

def get_azure_token():
    """Obtain Azure OAuth token from Entra ID"""
    app = ConfidentialClientApplication(
        client_id=dbutils.secrets.get("azure-creds", "client-id"),
        authority=f"https://login.microsoftonline.com/{dbutils.secrets.get('azure-creds', 'tenant-id')}",
        client_credential=dbutils.secrets.get("azure-creds", "client-secret")
    )
    result = app.acquire_token_for_client(scopes=["api://azure-mcp/.default"])
    return result['access_token'], result['expires_in']

def process_result(result):
    """Process MCP tool result - implement your business logic here"""
    print(f"Processing result: {result}")
    # Add your processing logic here

async def run_agent_job():
    """Long-running agent job with token refresh
    
    NOTE: This is a simplified example. Production implementations should include:
    - Proper termination logic (e.g., checking for shutdown signal)
    - Error handling with retries for transient failures
    - Graceful shutdown on exceptions
    - Health check reporting
    """
    # Long-running job loop with token refresh
    while True:  # In production, add termination condition
        try:
            # Get fresh token
            token, expires_in = get_azure_token()
            token_expiry = time.time() + expires_in
            
            # Create session with current token
            async with ClientSession(
                "https://apim-xyz.azure-api.net/mcp/sse",
                headers={"Authorization": f"Bearer {token}"}
            ) as session:
                
                # Use session until token needs refresh
                while time.time() < token_expiry - 300:  # 5 min buffer
                    # Invoke MCP tool (replace {...} with actual parameters)
                    result = await session.call_tool("analyze_data", {"param": "value"})
                    
                    # Process result
                    process_result(result)
                    
                    # Wait before next iteration
                    await asyncio.sleep(60)
                
                # Token is about to expire, loop will exit and recreate session
                # with new token on next outer loop iteration
        
        except Exception as e:
            # In production, implement proper error handling and retry logic
            print(f"Error in agent job: {e}")
            await asyncio.sleep(60)  # Wait before retry

# Run the agent job
asyncio.run(run_agent_job())
```

---

### Salesforce

#### Use Case: Apex-Based Agent

**Scenario**: Salesforce org needs to invoke Azure MCP tools as part of Apex trigger or batch process.

**Architecture**:
```
Salesforce (Apex Code) 
    ↓ HttpRequest (Named Credential)
    ↓ OAuth 2.0 JWT Bearer Flow
Azure APIM (MCP Endpoint)
    ↓ Backend: AKS MCP Server
Azure Services
```

**Identity Setup**:
1. Create Salesforce Connected App:
   - Enable OAuth settings
   - Enable JWT Bearer flow
   - Upload certificate for JWT signing
2. Create Azure Entra ID app registration for Salesforce federation
3. Configure Named Credential in Salesforce:
   - Type: External Service
   - Authentication: OAuth 2.0
   - Token Endpoint: `https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token`

**Code Pattern** (see earlier Salesforce example):
- Use `HttpRequest` to call `/message` endpoint
- Construct JSON-RPC 2.0 payload
- Parse JSON-RPC response

**Governor Limit Considerations**:
- **Callout Limits**: Max 100 callouts per transaction
- **Timeout**: 120 seconds max for synchronous callouts
- **Heap Size**: 6 MB for synchronous, 12 MB for asynchronous
- **Solution**: Use `@future(callout=true)` for async MCP calls

**Code Pattern for Async Callout**:
```apex
public class AzureMCPService {
    
    @future(callout=true)
    public static void callMCPToolAsync(String toolName, String paramsJson) {
        HttpRequest req = new HttpRequest();
        req.setEndpoint('callout:AzureMCP/message');
        req.setMethod('POST');
        req.setHeader('Content-Type', 'application/json');
        
        Map<String, Object> rpcRequest = new Map<String, Object>{
            'jsonrpc' => '2.0',
            'method' => 'tools/call',
            'params' => new Map<String, Object>{
                'name' => toolName,
                'arguments' => (Map<String, Object>)JSON.deserializeUntyped(paramsJson)
            },
            'id' => String.valueOf(DateTime.now().getTime())
        };
        req.setBody(JSON.serialize(rpcRequest));
        
        Http http = new Http();
        HttpResponse res = http.send(req);
        
        // Store result in custom object for later processing
        MCP_Result__c result = new MCP_Result__c(
            Tool_Name__c = toolName,
            Response__c = res.getBody(),
            Status_Code__c = res.getStatusCode()
        );
        insert result;
    }
}
```

#### Use Case: Salesforce Flow + External Service

**Scenario**: Business users configure Salesforce Flow that invokes Azure MCP tools without code.

**Setup**:
1. Register Azure MCP endpoint as External Service in Salesforce
2. Import OpenAPI schema for MCP `/message` endpoint
3. Create Flow with External Service action
4. Map Flow variables to MCP tool parameters

**Flow Configuration**:
- **Trigger**: Record-triggered (e.g., Opportunity closes)
- **Action**: External Service → AzureMCP → call_tool
- **Parameters**: Tool name and arguments from Flow variables
- **Output**: Store result in Opportunity custom field

**Example Flow**:
```
Start: When Opportunity.StageName = "Closed Won"
    ↓
Action: External Service "AzureMCP.call_tool"
    - Tool Name: "generate_contract"
    - Arguments: {"opportunity_id": "{!Opportunity.Id}"}
    ↓
Assignment: Opportunity.Contract_URL__c = {!AzureMCP.result.url}
    ↓
End
```

---

## Security & Governance Considerations

### Identity Federation / Token-Based Auth

**Preferred Pattern**: Federated Workload Identity
- External cloud identity (service account, IAM role, service principal) federates to Azure Entra ID
- No secrets leave the external cloud environment
- Token exchange happens at runtime via standard OIDC flows

**Implementation**:
1. **Azure Side**: Configure federated credential on Entra ID app registration
   - Add issuer URL (GCP: `https://accounts.google.com`, AWS: account-specific OIDC provider)
   - Add subject identifier (service account email, IAM role ARN, etc.)
2. **External Cloud Side**: Configure workload identity to request tokens for Azure scope
   - Scope: `api://{azure-app-id}/.default` or custom scopes

**Token Lifecycle Management**:
- Tokens expire after configured lifetime (recommended: 1 hour)
- Refresh tokens used for long-lived agents
- Token caching reduces authentication overhead

**Fallback Pattern**: Client Credentials Flow
- For environments without federation support, use client ID + secret
- Store secrets in cloud-native secrets managers (never in code)
- Rotate secrets regularly (recommended: every 90 days)

### Rate Limiting, Policy Enforcement, and Observability

#### Rate Limiting

**APIM Policies**:
```xml
<!-- Extract agent identity from JWT token for rate limiting -->
<rate-limit-by-key calls="100" renewal-period="60" 
    counter-key="@(context.Request.Headers.GetValueOrDefault("Authorization","").AsJwt()?.Subject ?? "anonymous")" />
```

**Strategies**:
- **Per-Agent Identity**: Limit based on OAuth token subject (agent ID) extracted from JWT
- **Per-Tenant**: Limit based on tenant ID claim in token
- **Per-Tool**: Different rate limits for different MCP tools (expensive vs. cheap operations)

**Quota Exhaustion Handling**:
- Return HTTP 429 (Too Many Requests) with `Retry-After` header
- External agent implements exponential backoff

#### Policy Enforcement

**Common APIM Policies**:
1. **Authentication**: `validate-jwt` policy verifies OAuth token signature and claims
2. **Authorization**: `check-header` or custom policy validates scopes/roles
3. **Request Validation**: `validate-content` ensures JSON-RPC schema compliance
4. **Response Transformation**: `set-body` or `json-to-xml` for format conversion
5. **IP Filtering**: `ip-filter` restricts access to known external IPs (optional)

**Custom Policy Example** (Content Safety):
```xml
<inbound>
    <!-- Call Azure Content Safety API before tool execution -->
    <send-request mode="new" response-variable-name="safety-check">
        <set-url>https://contentsafety.cognitiveservices.azure.com/contentsafety/text:analyze</set-url>
        <set-method>POST</set-method>
        <set-header name="Ocp-Apim-Subscription-Key" exists-action="override">
            <value>{{content-safety-key}}</value>
        </set-header>
        <set-body>@{
            var requestBody = context.Request.Body.As<JObject>();
            return JsonConvert.SerializeObject(new { text = requestBody["params"]["arguments"]["input"].ToString() });
        }</set-body>
    </send-request>
    
    <choose>
        <when condition="@(((IResponse)context.Variables["safety-check"]).Body.As<JObject>()["categoriesAnalysis"].Any(c => c["severity"].Value<int>() > 2))">
            <return-response>
                <set-status code="400" reason="Content Policy Violation" />
            </return-response>
        </when>
    </choose>
</inbound>
```

#### Observability

**Telemetry Collection**:
- **APIM**: All requests logged to Application Insights with custom dimensions:
  - `agent_id`: OAuth token subject
  - `tool_name`: MCP tool invoked
  - `latency_ms`: End-to-end request duration
  - `status_code`: HTTP status code
- **MCP Server**: Structured logging with OpenTelemetry:
  - Tool execution time
  - Azure service call latency (Cosmos query time, AI Search query time)
  - Error details with stack traces
- **Azure Services**: Resource-level metrics and logs

**Distributed Tracing**:
- Correlation ID (`X-Request-ID`) propagated from external agent → APIM → MCP server → Azure services
- OpenTelemetry spans created at each hop
- Visualize in Application Insights Application Map

**Alerting**:
- Alert on rate limit violations (potential DDoS or misconfiguration)
- Alert on authentication failures (credential issues)
- Alert on high latency (performance degradation)
- Alert on error rate spikes (service issues)

**Dashboard Example**:
```kusto
// App Insights query: MCP tool invocation metrics by external agent
requests
| where url has "/message"
| extend agent_id = tostring(customDimensions.agent_id)
| extend tool_name = tostring(customDimensions.tool_name)
| summarize 
    count(), 
    avg(duration), 
    percentile(duration, 95), 
    countif(resultCode >= 400) 
  by agent_id, tool_name
| order by count_ desc
```

### Why Azure Remains the System of Record

**Governance Enforcement**:
- All agent operations audited in Azure Monitor (immutable logs)
- Compliance requirements satisfied via Azure Policy and Defender for Cloud
- Data residency enforced via Azure region selection

**Identity Authority**:
- Azure Entra ID is the authoritative identity provider for all agents
- Conditional access policies enforced at the control plane
- Privileged identity management (PIM) for high-risk operations

**Data Sovereignty**:
- Enterprise data remains in Azure (Cosmos DB, AI Search, Storage)
- External agents invoke tools; data never leaves Azure boundary
- Data processing logs available for regulatory audit

**Policy Consistency**:
- APIM policies apply uniformly to all agents (Azure, GCP, AWS, Databricks, Salesforce)
- No "shadow IT" execution paths
- Single source of truth for tool definitions and access control

---

## Validation & Troubleshooting

### How to Verify a Successful MCP Handshake

#### Step 1: Verify OAuth Token

**Tool**: Use `jwt.io` or `az` CLI to decode token

```bash
# Decode token (do not share decoded tokens publicly)
az account get-access-token --resource api://azure-mcp | jq -r '.accessToken' | cut -d'.' -f2 | base64 -d | jq
```

**Validation Checklist**:
- [ ] `aud` (audience) matches Azure APIM app ID
- [ ] `iss` (issuer) is `https://login.microsoftonline.com/{tenant}/v2.0`
- [ ] `scp` (scopes) includes required scope (e.g., `next_best_action`)
- [ ] `exp` (expiration) is in the future
- [ ] `sub` (subject) matches external agent identity

#### Step 2: Test APIM Connectivity

**Tool**: `curl` or Postman

```bash
# Test /sse endpoint
curl -i -X GET "https://apim-xyz.azure-api.net/mcp/sse" \
  -H "Authorization: Bearer <token>" \
  -H "Accept: text/event-stream"

# Expected response:
# HTTP/1.1 200 OK
# Content-Type: text/event-stream
# ...
```

**Success Indicators**:
- Status code: `200 OK`
- Content-Type: `text/event-stream`
- SSE connection established (connection stays open)

#### Step 3: Test Tool Discovery

**Tool**: MCP client or `curl`

```bash
# Send tools/list request
curl -X POST "https://apim-xyz.azure-api.net/mcp/message" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
  }'

# Expected response:
# {
#   "jsonrpc": "2.0",
#   "id": 1,
#   "result": {
#     "tools": [
#       {"name": "query_data", "description": "...", "inputSchema": {...}},
#       {"name": "analyze_sales", "description": "...", "inputSchema": {...}}
#     ]
#   }
# }
```

**Success Indicators**:
- Status code: `200 OK`
- JSON-RPC response with `result.tools` array
- Tool schemas include `name`, `description`, `inputSchema`

#### Step 4: Test Tool Invocation

**Tool**: MCP client or `curl`

```bash
# Send tools/call request
curl -X POST "https://apim-xyz.azure-api.net/mcp/message" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "query_data",
      "arguments": {"query": "SELECT * FROM customers LIMIT 10"}
    },
    "id": 2
  }'

# Expected response:
# {
#   "jsonrpc": "2.0",
#   "id": 2,
#   "result": {
#     "content": [{"type": "text", "text": "Query results: ..."}]
#   }
# }
```

**Success Indicators**:
- Status code: `200 OK`
- JSON-RPC response with `result.content` array
- Tool execution completed without errors

### Common Failure Modes and Misconfigurations

#### Failure Mode 1: 401 Unauthorized

**Symptoms**:
```json
{
  "statusCode": 401,
  "message": "Access token is missing or invalid"
}
```

**Possible Causes**:
1. Token not included in `Authorization` header
2. Token expired
3. Token audience (`aud`) doesn't match APIM app ID
4. Token issuer (`iss`) not trusted by Azure

**Resolution**:
- Verify token is sent: `Authorization: Bearer <token>`
- Check token expiration: decode token and verify `exp` claim
- Verify token audience matches APIM app registration ID
- Verify issuer URL matches Azure Entra ID tenant

#### Failure Mode 2: 403 Forbidden

**Symptoms**:
```json
{
  "statusCode": 403,
  "message": "Insufficient permissions to access this resource"
}
```

**Possible Causes**:
1. Token missing required scope (e.g., `next_best_action`)
2. Agent identity not granted access to APIM API
3. IP address not in allowed list (if IP filtering enabled)

**Resolution**:
- Decode token and verify `scp` claim includes required scope
- In Azure Portal, verify agent app registration has API permissions for APIM API
- Check APIM IP filter policy (if enabled)

#### Failure Mode 3: 429 Too Many Requests

**Symptoms**:
```json
{
  "statusCode": 429,
  "message": "Rate limit exceeded. Retry after 60 seconds."
}
```

**Possible Causes**:
1. Agent exceeded rate limit quota
2. Shared token (multiple agents using same identity)

**Resolution**:
- Implement exponential backoff in agent code
- Reduce request frequency
- Request rate limit increase from Azure admin
- Use separate agent identities for different workloads

#### Failure Mode 4: 500 Internal Server Error

**Symptoms**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32603,
    "message": "Internal error"
  }
}
```

**Possible Causes**:
1. MCP server error (bug in tool implementation)
2. Azure service dependency failure (Cosmos, Search, etc.)
3. Network connectivity issue between APIM and MCP server

**Resolution**:
- Check APIM backend health: Azure Portal → APIM → Backends
- Check MCP server logs: `kubectl logs -n default deployment/mcp-agents`
- Check Application Insights for error details with stack traces
- Verify Azure service health status

#### Failure Mode 5: Connection Timeout

**Symptoms**:
- SSE connection never established
- Request hangs indefinitely

**Possible Causes**:
1. Network firewall blocking outbound HTTPS from external cloud
2. APIM backend unreachable (AKS cluster down)
3. Client-side timeout too short

**Resolution**:
- Verify outbound HTTPS allowed from external cloud to Azure (port 443)
- Check AKS cluster status: `kubectl get nodes`
- Increase client-side timeout (recommended: 30 seconds for SSE, 60 seconds for tool calls)
- Test connectivity with `curl` from external cloud environment

#### Failure Mode 6: JSON-RPC Protocol Error

**Symptoms**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32600,
    "message": "Invalid Request"
  }
}
```

**Possible Causes**:
1. Malformed JSON-RPC request (missing required fields)
2. Incorrect HTTP method (GET instead of POST for `/message`)
3. Wrong Content-Type header

**Resolution**:
- Verify JSON-RPC request includes: `jsonrpc`, `method`, `id`
- Use POST method for `/message` endpoint
- Set `Content-Type: application/json` header
- Validate JSON syntax with `jq` or JSON linter

---

## References

### Azure AI Foundry

- [Azure AI Foundry Overview](https://learn.microsoft.com/azure/ai-foundry/what-is-foundry)
- [Azure AI Foundry Agent Service](https://learn.microsoft.com/azure/ai-foundry/agents/overview)
- [Agent Service Quickstart](https://learn.microsoft.com/azure/ai-foundry/agents/quickstart)

### Model Context Protocol (MCP)

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP for Azure](https://learn.microsoft.com/azure/developer/azure-mcp-server/tools/azure-foundry)
- [MCP Python Client Library](https://github.com/modelcontextprotocol/python-sdk)

### Azure API Management

- [APIM Overview](https://learn.microsoft.com/azure/api-management/api-management-key-concepts)
- [OAuth 2.0 Authorization in APIM](https://learn.microsoft.com/azure/api-management/api-management-howto-oauth2)
- [APIM Policies Reference](https://learn.microsoft.com/azure/api-management/api-management-policies)
- [Rate Limiting in APIM](https://learn.microsoft.com/azure/api-management/rate-limit-policy)

### Microsoft Entra ID (Azure AD)

- [Entra ID Agent Identities](https://learn.microsoft.com/entra/agent-id/identity-professional/microsoft-entra-agent-identities-for-ai-agents)
- [Workload Identity Federation](https://learn.microsoft.com/entra/workload-id/workload-identity-federation)
- [OAuth 2.0 and OpenID Connect](https://learn.microsoft.com/entra/identity-platform/v2-protocols)

### Related Architecture Documentation

- [Azure Agents Control Plane Constitution](../.speckit/constitution.md)
- [System Architecture](AGENTS_ARCHITECTURE.md)
- [Identity Design for MCP Agents](AGENTS_IDENTITY_DESIGN.md)
- [Agent Deployment Notes](AGENTS_DEPLOYMENT_NOTES.md)

### External Cloud Documentation

#### Google Cloud Platform
- [Workload Identity Federation for GCP](https://cloud.google.com/iam/docs/workload-identity-federation)
- [Vertex AI Agent Builder](https://cloud.google.com/vertex-ai/docs/agent-builder/overview)

#### Amazon Web Services
- [IAM Roles for ECS Tasks](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-iam-roles.html)
- [AWS Lambda with OIDC](https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html)

#### Databricks
- [Databricks Secrets Management](https://docs.databricks.com/security/secrets/index.html)
- [Databricks Azure Integration](https://docs.databricks.com/administration-guide/cloud-configurations/azure/index.html)

#### Salesforce
- [Salesforce External Services](https://help.salesforce.com/s/articleView?id=sf.external_services_about.htm)
- [Salesforce Named Credentials](https://help.salesforce.com/s/articleView?id=sf.named_credentials_about.htm)
- [Apex Callouts](https://developer.salesforce.com/docs/atlas.en-us.apexcode.meta/apexcode/apex_callouts.htm)

---

## Conclusion

This guide provides the foundation for connecting external agents (GCP, AWS, Databricks, Salesforce) to the Azure Agents Control Plane via MCP. By centralizing governance, identity, and observability in Azure while allowing execution flexibility across clouds, organizations achieve:

- **Enterprise-grade agent governance** regardless of execution location
- **Consistent policy enforcement** across all agents and platforms
- **Unified observability** for compliance and troubleshooting
- **Secure, keyless authentication** via workload identity federation
- **Multi-cloud flexibility** without sacrificing control

For hands-on implementation guidance, refer to the [Lab Manual](LAB_MANUAL_BUILD_YOUR_OWN_AGENT.md) and [Architecture Documentation](AGENTS_ARCHITECTURE.md).
