# System Architecture Diagrams

This document provides comprehensive Mermaid diagrams for the AI Agents with AKS and APIM solution.

## Table of Contents

1. [Component Architecture Diagram](#component-architecture-diagram)
2. [Detailed Component Diagram](#detailed-component-diagram)
3. [Deployment Architecture](#deployment-architecture)
4. [Sequence Diagrams](#sequence-diagrams)
   - [Agent Authentication Flow](#agent-authentication-flow)
   - [MCP Tool Discovery](#mcp-tool-discovery)
   - [MCP Tool Execution](#mcp-tool-execution)
5. [Activity Diagrams](#activity-diagrams)
   - [Deployment Process](#deployment-process-activity-diagram)
   - [Request Processing](#request-processing-activity-diagram)

---

## Component Architecture Diagram

High-level component view of the entire system:

```mermaid
graph TB
    subgraph "Client Layer"
        Agent[AI Agent<br/>Claude/ChatGPT/Custom]
        Inspector[MCP Inspector<br/>Testing Tool]
    end

    subgraph "Azure API Management"
        APIM[APIM Gateway<br/>OAuth + Routing]
        OAuthAPI[OAuth API<br/>/authorize, /token]
        MCPAPI[MCP API<br/>/sse, /message]
    end

    subgraph "Azure Kubernetes Service"
        subgraph "System Node Pool"
            MCPPod1[MCP Server Pod 1]
            MCPPod2[MCP Server Pod 2]
        end
        
        MCPSvc[MCP Server Service<br/>ClusterIP]
    end

    subgraph "Azure Services"
        ACR[Azure Container<br/>Registry]
        Storage[Azure Blob<br/>Storage]
        AppInsights[Application<br/>Insights]
        EntraID[Azure Entra ID<br/>OAuth Provider]
    end

    Agent --> APIM
    Inspector --> APIM
    APIM --> OAuthAPI
    APIM --> MCPAPI
    OAuthAPI --> EntraID
    MCPAPI --> MCPSvc
    MCPSvc --> MCPPod1
    MCPSvc --> MCPPod2
    MCPPod1 --> Storage
    MCPPod2 --> Storage
    MCPPod1 --> AppInsights
    MCPPod2 --> AppInsights

    style Agent fill:#e1f5ff
    style APIM fill:#fff4e6
    style MCPPod1 fill:#e8f5e9
    style MCPPod2 fill:#e8f5e9
```

---

## Detailed Component Diagram

Detailed view showing all components, their responsibilities, and interactions:

```mermaid
graph TB
    subgraph "AI Agent Client"
        AgentCore[Agent Core Engine]
        MCPClient[MCP Client Library]
        AgentCore --> MCPClient
    end

    subgraph "Azure API Management Layer"
        subgraph "OAuth Components"
            AuthEndpoint[/authorize Endpoint]
            TokenEndpoint[/token Endpoint]
            RegisterEndpoint[/register Endpoint]
            WellKnown[/.well-known<br/>OAuth Metadata]
        end
        
        subgraph "MCP Components"
            SSEEndpoint[/sse Endpoint<br/>Server-Sent Events]
            MessageEndpoint[/message Endpoint<br/>JSON-RPC 2.0]
        end
        
        subgraph "APIM Policies"
            AuthPolicy[Authentication Policy<br/>OAuth Token Validation]
            RateLimitPolicy[Rate Limiting Policy]
            CORSPolicy[CORS Policy]
            BackendPolicy[Backend Routing Policy]
        end
    end

    subgraph "AKS Cluster - System Node Pool"
        subgraph "MCP Server Deployment"
            subgraph "Pod 1"
                FastAPI1[FastAPI App]
                SSEHandler1[SSE Handler<br/>Async Streaming]
                MCPTools1[MCP Tool Executor<br/>hello, get, save]
                StorageClient1[Azure Storage SDK]
            end
            
            subgraph "Pod 2"
                FastAPI2[FastAPI App]
                SSEHandler2[SSE Handler]
                MCPTools2[MCP Tool Executor]
                StorageClient2[Azure Storage SDK]
            end
        end
        
        K8sService[Kubernetes Service<br/>Load Balancer]
        ServiceAccount[Service Account<br/>Workload Identity]
    end

    subgraph "Azure Container Registry"
        MCPImage[MCP Server Image<br/>mcp-server:latest]
    end

    subgraph "Azure Storage"
        SnippetsContainer[snippets Container<br/>Blob Storage]
        DeploymentContainer[deployment Container<br/>Code Packages]
    end

    subgraph "Monitoring & Identity"
        AppInsightsSDK[App Insights SDK<br/>Telemetry]
        ManagedIdentity[Managed Identity<br/>AAD Integration]
        LogAnalytics[Log Analytics<br/>Workspace]
    end

    MCPClient --> SSEEndpoint
    MCPClient --> MessageEndpoint
    SSEEndpoint --> AuthPolicy
    MessageEndpoint --> AuthPolicy
    AuthPolicy --> RateLimitPolicy
    RateLimitPolicy --> BackendPolicy
    BackendPolicy --> K8sService
    
    K8sService --> FastAPI1
    K8sService --> FastAPI2
    
    FastAPI1 --> SSEHandler1
    FastAPI1 --> MCPTools1
    MCPTools1 --> StorageClient1
    StorageClient1 --> SnippetsContainer
    
    FastAPI2 --> SSEHandler2
    FastAPI2 --> MCPTools2
    MCPTools2 --> StorageClient2
    StorageClient2 --> SnippetsContainer
    
    ServiceAccount --> ManagedIdentity
    ManagedIdentity --> SnippetsContainer
    
    ACR --> MCPImage
    
    FastAPI1 --> AppInsightsSDK
    FastAPI2 --> AppInsightsSDK
    AppInsightsSDK --> LogAnalytics
    
    AuthEndpoint --> ManagedIdentity

    style AgentCore fill:#e1f5ff
    style APIM fill:#fff4e6
    style FastAPI1 fill:#e8f5e9
    style FastAPI2 fill:#e8f5e9
```

---

## Deployment Architecture

Infrastructure and deployment view:

```mermaid
graph TB
    subgraph "Azure Subscription"
        subgraph "Resource Group"
            subgraph "Network Layer"
                VNet[Virtual Network<br/>Optional]
                SystemSubnet[System Subnet<br/>10.240.0.0/16]
            end
            
            subgraph "Compute Layer"
                AKS[AKS Cluster<br/>Managed Kubernetes]
                SystemNodePool[System Node Pool<br/>Standard_DS2_v2<br/>Min: 2, Max: 4]
            end
            
            subgraph "Container Services"
                ACR[Azure Container Registry<br/>Standard SKU]
            end
            
            subgraph "API Layer"
                APIM[API Management<br/>Developer/Standard Tier]
            end
            
            subgraph "Storage Layer"
                StorageAcct[Storage Account<br/>Standard_LRS]
                BlobService[Blob Service<br/>snippets container]
            end
            
            subgraph "Monitoring Layer"
                LogWorkspace[Log Analytics Workspace]
                AppInsights[Application Insights]
            end
            
            subgraph "Identity Layer"
                AKSIdentity[AKS Managed Identity]
                MCPIdentity[MCP Workload Identity]
                APIMIdentity[APIM Managed Identity]
            end
        end
    end

    AKS --> SystemNodePool
    SystemNodePool --> SystemSubnet
    
    AKS --> AKSIdentity
    AKS --> ACR
    
    APIM --> AKS
    APIM --> APIMIdentity
    
    SystemNodePool --> StorageAcct
    StorageAcct --> BlobService
    
    MCPIdentity --> BlobService
    
    AKS --> AppInsights
    AppInsights --> LogWorkspace
    
    ACR --> AKSIdentity

    style AKS fill:#326ce5,color:#fff
    style SystemNodePool fill:#326ce5,color:#fff
    style APIM fill:#0078d4,color:#fff
    style ACR fill:#0078d4,color:#fff
    style StorageAcct fill:#0078d4,color:#fff
```

---

## Sequence Diagrams

### Agent Authentication Flow

OAuth 2.0 PKCE flow for AI agent authentication:

```mermaid
sequenceDiagram
    participant Agent as AI Agent
    participant APIM as API Management
    participant EntraID as Azure Entra ID
    participant MCP as MCP Server

    Note over Agent,MCP: OAuth 2.0 Authorization Code Flow with PKCE
    
    Agent->>Agent: Generate code_verifier<br/>Generate code_challenge
    
    Agent->>APIM: GET /oauth/authorize<br/>?client_id=...&code_challenge=...
    APIM->>EntraID: Redirect to Entra ID login
    
    Note over EntraID: User authenticates
    
    EntraID-->>APIM: Authorization code
    APIM-->>Agent: Authorization code
    
    Agent->>APIM: POST /oauth/token<br/>code + code_verifier
    APIM->>EntraID: Exchange code for tokens
    EntraID-->>APIM: Access token + Refresh token
    APIM-->>Agent: Access token
    
    Agent->>APIM: GET /mcp/sse<br/>Authorization: Bearer {token}
    APIM->>APIM: Validate token
    APIM->>MCP: Forward request
    MCP-->>Agent: SSE Connection established
```

### MCP Tool Discovery

How AI agents discover available tools:

```mermaid
sequenceDiagram
    participant Agent as AI Agent
    participant APIM as API Management
    participant MCP as MCP Server
    participant Tools as Tool Registry

    Agent->>APIM: SSE Connect /mcp/sse<br/>Authorization: Bearer token
    APIM->>MCP: Forward connection
    MCP-->>Agent: SSE Stream opened
    
    Agent->>APIM: POST /mcp/message<br/>{"method": "tools/list"}
    APIM->>MCP: Forward request
    
    MCP->>Tools: Get registered tools
    Tools-->>MCP: Tool definitions
    
    MCP-->>Agent: {<br/>  "tools": [<br/>    {"name": "hello_mcp", ...},<br/>    {"name": "save_snippet", ...},<br/>    {"name": "get_snippet", ...}<br/>  ]<br/>}
    
    Agent->>Agent: Cache available tools<br/>for reasoning
```

### MCP Tool Execution

Executing a tool and handling the response:

```mermaid
sequenceDiagram
    participant Agent as AI Agent
    participant APIM as API Management
    participant MCP as MCP Server
    participant Storage as Azure Storage
    participant Insights as App Insights

    Agent->>APIM: POST /mcp/message<br/>{"method": "tools/call",<br/> "params": {<br/>   "name": "save_snippet",<br/>   "arguments": {<br/>     "snippetname": "test",<br/>     "snippet": "Hello World"<br/>   }<br/> }}
    
    APIM->>APIM: Validate token<br/>Check rate limits
    
    APIM->>MCP: Forward tool request
    
    MCP->>MCP: Parse tool call<br/>Validate arguments
    
    MCP->>Insights: Log: Tool called<br/>Tool: save_snippet
    
    MCP->>Storage: Upload blob<br/>Container: snippets<br/>Name: test
    
    Storage-->>MCP: Upload success
    
    MCP->>MCP: Build success response
    
    MCP->>Insights: Log: Tool execution completed<br/>Duration: 150ms
    
    MCP-->>APIM: {<br/>  content: [{<br/>    type: "text",<br/>    text: "Snippet saved successfully"<br/>  }],<br/>  isError: false<br/>}
    
    APIM-->>Agent: Tool execution result
    
    Agent->>Agent: Process result<br/>Continue reasoning
```

---

## Activity Diagrams

### Deployment Process Activity Diagram

Complete deployment workflow:

```mermaid
flowchart TD
    Start([Start Deployment]) --> PreReq{Prerequisites<br/>Installed?}
    
    PreReq -->|No| InstallTools[Install:<br/>- Azure CLI<br/>- azd<br/>- kubectl<br/>- Docker]
    InstallTools --> PreReq
    
    PreReq -->|Yes| AzdInit[Run: azd init]
    AzdInit --> SetEnv[Select/Create<br/>Environment]
    SetEnv --> AzdUp[Run: azd up]
    
    AzdUp --> ProvInfra{Infrastructure<br/>Provision Success?}
    
    ProvInfra -->|No| CheckErrors[Check Errors:<br/>- Quota limits<br/>- Region availability<br/>- Permissions]
    CheckErrors --> AzdUp
    
    ProvInfra -->|Yes| InfraReady[Infrastructure Ready:<br/>✓ AKS Cluster<br/>✓ ACR<br/>✓ APIM<br/>✓ Storage<br/>✓ Monitoring]
    
    InfraReady --> GetCreds[azd post-provision:<br/>Get AKS credentials]
    GetCreds --> BuildImage[Run:<br/>./scripts/build-and-push.sh]
    BuildImage --> ImagePush[Push to ACR]
    
    ImagePush --> DeployMCP[kubectl apply:<br/>mcp-server-deployment.yaml]
    DeployMCP --> MCPCheck{MCP Server<br/>Pods Ready?}
    
    MCPCheck -->|No| WaitMCP[Wait for pods<br/>Check logs]
    WaitMCP --> MCPCheck
    
    MCPCheck -->|Yes| RunTests[Run Tests:<br/>- test_apim_mcp_connection.py<br/>- test_apim_mcp_use_cases.py]
    
    RunTests --> TestsPass{All Tests<br/>Pass?}
    
    TestsPass -->|No| Debug[Debug Issues:<br/>- Check logs<br/>- Verify connectivity<br/>- Check auth]
    Debug --> RunTests
    
    TestsPass -->|Yes| Complete([Deployment Complete<br/>System Ready])
    
    style Start fill:#4caf50,color:#fff
    style Complete fill:#4caf50,color:#fff
    style PreReq fill:#ff9800
    style ProvInfra fill:#ff9800
    style MCPCheck fill:#ff9800
    style TestsPass fill:#ff9800
```

### Request Processing Activity Diagram

How a request flows through the system:

```mermaid
flowchart TD
    Start([AI Agent Request]) --> HasToken{Has Valid<br/>Access Token?}
    
    HasToken -->|No| StartAuth[Initiate OAuth Flow]
    StartAuth --> CodeChallenge[Generate PKCE<br/>code_challenge]
    CodeChallenge --> Authorize[Call /oauth/authorize]
    Authorize --> UserLogin[User authenticates<br/>with Entra ID]
    UserLogin --> GetCode[Receive auth code]
    GetCode --> ExchangeToken[POST /oauth/token<br/>with code_verifier]
    ExchangeToken --> ReceiveToken[Receive access_token]
    ReceiveToken --> HasToken
    
    HasToken -->|Yes| ConnectSSE[Connect to<br/>/mcp/sse endpoint]
    
    ConnectSSE --> ValidateToken{APIM Validates<br/>Token?}
    
    ValidateToken -->|Invalid| Return401[Return 401<br/>Unauthorized]
    Return401 --> End([End])
    
    ValidateToken -->|Valid| CheckRateLimit{Within Rate<br/>Limits?}
    
    CheckRateLimit -->|No| Return429[Return 429<br/>Too Many Requests]
    Return429 --> End
    
    CheckRateLimit -->|Yes| ForwardAPIM[APIM forwards to<br/>AKS Service]
    
    ForwardAPIM --> LoadBalance{Load Balancer<br/>Select Pod}
    
    LoadBalance -->|Pod 1| MCP1[MCP Server Pod 1]
    LoadBalance -->|Pod 2| MCP2[MCP Server Pod 2]
    
    MCP1 --> CreateSession[Create SSE Session]
    MCP2 --> CreateSession
    
    CreateSession --> StreamOpen[Stream Connection<br/>Established]
    
    StreamOpen --> WaitRequest[Wait for<br/>JSON-RPC Request]
    
    WaitRequest --> ReqType{Request Type?}
    
    ReqType -->|tools/list| GetTools[Retrieve Tool<br/>Definitions]
    GetTools --> ReturnTools[Return tool list<br/>via SSE]
    ReturnTools --> WaitRequest
    
    ReqType -->|tools/call| ParseTool[Parse tool name<br/>and arguments]
    
    ParseTool --> ToolType{Which Tool?}
    
    ToolType -->|hello_mcp| SimpleResp[Return greeting<br/>message]
    SimpleResp --> LogMetric
    
    ToolType -->|save_snippet| GetToken[Get Managed<br/>Identity token]
    GetToken --> WriteBlog[Write to Azure<br/>Blob Storage]
    WriteBlog --> ConfirmSave[Return success<br/>confirmation]
    ConfirmSave --> LogMetric
    
    ToolType -->|get_snippet| GetToken2[Get Managed<br/>Identity token]
    GetToken2 --> ReadBlob[Read from Azure<br/>Blob Storage]
    ReadBlob --> ReturnSnippet[Return snippet<br/>content]
    ReturnSnippet --> LogMetric
    
    LogMetric[Log to App Insights:<br/>- Duration<br/>- Success/Failure<br/>- Tool name] --> CheckMore{More Requests?}
    
    CheckMore -->|Yes| WaitRequest
    CheckMore -->|No| CloseSSE[Close SSE<br/>Connection]
    
    CloseSSE --> Cleanup[Cleanup session<br/>resources]
    Cleanup --> End
    
    style Start fill:#2196f3,color:#fff
    style End fill:#2196f3,color:#fff
    style ValidateToken fill:#ff9800
    style CheckRateLimit fill:#ff9800
    style ReqType fill:#ff9800
    style ToolType fill:#ff9800
    style CheckMore fill:#ff9800
```

---

## Legend

### Component Diagram Shapes

- **Rectangle**: Component or Service
- **Cylinder**: Database or Storage
- **Hexagon**: External Service
- **Diamond**: Decision Point
- **Rounded Rectangle**: Process or Function

### Sequence Diagram Notation

- **Solid Line →**: Synchronous Call
- **Dashed Line ⇢**: Response
- **Note**: Additional Information
- **Alt/Else**: Alternative Paths

### Activity Diagram Symbols

- **Rounded Rectangle**: Activity/Process
- **Diamond**: Decision Point
- **Circle**: Start/End Point
- **Rectangle**: Subprocess

---

## Viewing These Diagrams

These Mermaid diagrams can be rendered in:

1. **GitHub/GitLab**: Automatically renders in markdown files
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Mermaid Live Editor**: https://mermaid.live
4. **Documentation Sites**: Most support Mermaid natively

---

## Additional Resources

- [Mermaid Documentation](https://mermaid.js.org/)
- [Deployment Guide](./README.md)
- [Azure Kubernetes Service](https://learn.microsoft.com/azure/aks/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
