# System Architecture

This document provides comprehensive Mermaid diagrams for the AI Agents with AKS and APIM solution.

## Table of Contents

1. [Component Architecture Diagram](#component-architecture-diagram)
2. [Detailed Component Diagram](#detailed-component-diagram)
3. [Memory Architecture](#memory-architecture)
   - [Memory Components](#memory-components)
   - [Facts Memory - Fabric IQ Integration](#facts-memory---fabric-iq-integration)
   - [Domain Ontologies](#domain-ontologies)
   - [Cross-Domain Reasoning](#cross-domain-reasoning)
   - [Memory Flow](#memory-flow)
4. [Deployment Architecture](#deployment-architecture)
5. [Sequence Diagrams](#sequence-diagrams)
   - [Agent Authentication Flow](#agent-authentication-flow)
   - [MCP Tool Discovery](#mcp-tool-discovery)
   - [MCP Tool Execution](#mcp-tool-execution)
6. [Activity Diagrams](#activity-diagrams)
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
        MCPImage[MCP Server Image<br/>mcp-agents:latest]
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

## Memory Architecture

The MCP Agent uses a composite memory system combining short-term session memory, long-term persistent memory, and ontology-grounded facts memory for semantic reasoning:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CompositeMemory                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐ │
│  │  Short-Term Memory  │  │   Long-Term Memory  │  │     Facts Memory        │ │
│  │    (CosmosDB)       │  │   (AI Search)       │  │    (Fabric IQ)          │ │
│  │  - Session-based    │  │  - Persistent       │  │  - Ontology-grounded    │ │
│  │  - TTL support      │  │  - Cross-session    │  │  - Cross-domain         │ │
│  │  - Fast access      │  │  - Hybrid search    │  │  - Entity relationships │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Memory Components

```mermaid
graph TB
    subgraph "Agent Layer"
        Agent[MCP Agent<br/>next_best_action_agent.py]
        Tools[MCP Tools<br/>store_memory, recall_memory]
        FactsTools[Facts Tools<br/>search_facts, cross_domain_analysis]
    end

    subgraph "Memory Abstraction Layer"
        CompositeMemory[CompositeMemory<br/>Unified Memory Interface]
        
        subgraph "Short-Term Memory"
            CosmosMemory[ShortTermMemory<br/>Session-based storage]
            STMFeatures[Features:<br/>• TTL support<br/>• Partition by session_id<br/>• Vector similarity search]
        end
        
        subgraph "Long-Term Memory"
            AISearchMemory[LongTermMemory<br/>Persistent storage]
            FoundryIQ[FoundryIQMemory<br/>Knowledge graph]
            LTMFeatures[Features:<br/>• Cross-session retrieval<br/>• Hybrid search<br/>• Entity extraction]
        end
        
        subgraph "Facts Memory"
            FactsMemory[FactsMemory<br/>Ontology-grounded facts]
            FMFeatures[Features:<br/>• Domain ontologies<br/>• Entity relationships<br/>• Cross-domain reasoning]
        end
    end

    subgraph "Storage Layer"
        subgraph "Azure CosmosDB"
            ShortTermContainer[short_term_memory<br/>Container]
            TasksContainer[tasks Container]
            PlansContainer[plans Container]
        end
        
        subgraph "Azure AI Search"
            SearchIndex[long_term_memory<br/>Search Index]
        end
        
        subgraph "Microsoft Fabric IQ"
            Ontology[Ontology<br/>Entity Types & Relationships]
            OneLake[OneLake<br/>Unified Data Storage]
        end
        
        subgraph "Azure AI Foundry"
            EmbeddingModel[text-embedding-3-large<br/>3072 dimensions]
            ChatModel[gpt-5.2-chat<br/>Intent Analysis]
        end
    end

    Agent --> Tools
    Agent --> FactsTools
    Tools --> CompositeMemory
    FactsTools --> FactsMemory
    CompositeMemory --> CosmosMemory
    CompositeMemory --> AISearchMemory
    CompositeMemory --> FoundryIQ
    
    CosmosMemory --> ShortTermContainer
    CosmosMemory --> TasksContainer
    CosmosMemory --> PlansContainer
    
    AISearchMemory --> SearchIndex
    
    FactsMemory --> Ontology
    FactsMemory --> OneLake
    
    CosmosMemory --> EmbeddingModel
    AISearchMemory --> EmbeddingModel
    FactsMemory --> EmbeddingModel
    Agent --> ChatModel

    style Agent fill:#e1f5ff
    style CompositeMemory fill:#fff4e6
    style CosmosMemory fill:#e8f5e9
    style AISearchMemory fill:#fce4ec
    style FoundryIQ fill:#f3e5f5
    style FactsMemory fill:#e3f2fd
```

### Facts Memory - Fabric IQ Integration

The Facts Memory provider integrates with [Microsoft Fabric IQ](https://learn.microsoft.com/en-us/fabric/iq/overview) to provide ontology-grounded facts for AI agent reasoning. It organizes data according to business language and exposes it with consistent semantic meaning.

```mermaid
graph TB
    subgraph "Facts Memory Architecture"
        FactsMemory[FactsMemory Provider]
        
        subgraph "Domain Ontologies"
            CustomerOntology[Customer Domain<br/>Churn Analysis]
            DevOpsOntology[DevOps Domain<br/>CI/CD Pipelines]
            UserMgmtOntology[User Management<br/>Authentication]
        end
        
        subgraph "Entity Types"
            CustomerEntities[Customer<br/>Transaction<br/>Engagement<br/>ChurnPrediction]
            PipelineEntities[Pipeline<br/>PipelineRun<br/>Deployment<br/>Cluster]
            UserEntities[User<br/>Session<br/>AuthEvent<br/>AccessLog]
        end
        
        subgraph "Fact Types"
            Observations[Observations<br/>Current state facts]
            Predictions[Predictions<br/>ML-derived insights]
            Derived[Derived Facts<br/>Cross-domain reasoning]
        end
    end
    
    subgraph "Fabric IQ Components"
        OntologyItem[Ontology Item<br/>Entity & Relationship Definitions]
        GraphFabric[Graph in Fabric<br/>Relationship Traversal]
        DataAgent[Data Agent<br/>Conversational Q&A]
    end
    
    subgraph "OneLake"
        Lakehouse[Lakehouse Tables]
        Eventhouse[Eventhouse Streams]
        SemanticModel[Power BI Semantic Model]
    end

    FactsMemory --> CustomerOntology
    FactsMemory --> DevOpsOntology
    FactsMemory --> UserMgmtOntology
    
    CustomerOntology --> CustomerEntities
    DevOpsOntology --> PipelineEntities
    UserMgmtOntology --> UserEntities
    
    CustomerEntities --> Observations
    PipelineEntities --> Observations
    UserEntities --> Observations
    
    Observations --> Predictions
    Predictions --> Derived
    
    OntologyItem --> Lakehouse
    OntologyItem --> Eventhouse
    OntologyItem --> SemanticModel
    
    GraphFabric --> OntologyItem
    DataAgent --> OntologyItem

    style FactsMemory fill:#e3f2fd
    style CustomerOntology fill:#ffecb3
    style DevOpsOntology fill:#c8e6c9
    style UserMgmtOntology fill:#f3e5f5
```

### Domain Ontologies

| Domain | Entity Types | Use Cases |
|--------|--------------|-----------|
| **Customer** | Customer, Transaction, Engagement, ChurnPrediction | Customer churn analysis, predictive modeling, at-risk identification |
| **DevOps** | Pipeline, PipelineRun, Deployment, Cluster | CI/CD monitoring, failure analysis, deployment tracking |
| **User Management** | User, Session, AuthEvent, AccessLog | Security monitoring, access patterns, authentication analysis |

### Fact Types

| Fact Type | Description | Example |
|-----------|-------------|---------|
| `observation` | Current state of an entity | "Pipeline 'api-gateway' has 93% success rate" |
| `prediction` | ML-derived forecast | "Customer 'Liam' has 82% churn risk" |
| `derived` | Cross-domain insight | "Deployment failures correlate with high-risk customers" |
| `rule` | Business rule fact | "MFA required for admin access" |

### Cross-Domain Reasoning

Facts Memory enables cross-domain analysis to find connections between different business domains:

```mermaid
graph LR
    subgraph "Customer Domain"
        HighRiskCustomer[High-Risk Customer<br/>82% churn risk]
    end
    
    subgraph "DevOps Domain"
        FailedDeployment[Failed Deployment<br/>user-service outage]
    end
    
    subgraph "User Management"
        SecurityAlert[Security Alert<br/>Suspicious login]
    end
    
    HighRiskCustomer -->|"Impact Analysis"| FailedDeployment
    FailedDeployment -->|"Service Affected"| SecurityAlert
    SecurityAlert -->|"User Correlation"| HighRiskCustomer
    
    style HighRiskCustomer fill:#ffcdd2
    style FailedDeployment fill:#fff9c4
    style SecurityAlert fill:#f3e5f5
```

### Memory Flow

```mermaid
sequenceDiagram
    participant Agent as MCP Agent
    participant CM as CompositeMemory
    participant STM as Short-Term Memory<br/>(CosmosDB)
    participant LTM as Long-Term Memory<br/>(AI Search)
    participant FM as Facts Memory<br/>(Fabric IQ)
    participant Embed as Embedding Model

    Note over Agent,Embed: Store Memory Flow
    Agent->>CM: store(content, session_id)
    CM->>Embed: get_embedding(content)
    Embed-->>CM: embedding vector (3072 dims)
    CM->>STM: store(entry + embedding)
    STM-->>CM: entry_id
    
    alt Promote to Long-Term
        CM->>LTM: store(entry + embedding)
        LTM-->>CM: entry_id
    end
    CM-->>Agent: stored entry_id

    Note over Agent,Embed: Recall Memory Flow
    Agent->>CM: search(query, session_id)
    CM->>Embed: get_embedding(query)
    Embed-->>CM: query embedding
    
    par Search Short-Term
        CM->>STM: vector_search(embedding)
        STM-->>CM: short-term results
    and Search Long-Term
        CM->>LTM: hybrid_search(embedding, text)
        LTM-->>CM: long-term results
    end
    
    Note over Agent,Embed: Facts Retrieval Flow
    Agent->>FM: search_facts(query, domain)
    FM->>Embed: get_embedding(query)
    Embed-->>FM: query embedding
    FM->>FM: semantic_search(facts)
    FM-->>Agent: ontology-grounded facts
    
    Note over Agent,Embed: Cross-Domain Analysis
    Agent->>FM: cross_domain_query(query, source, target)
    FM->>FM: traverse_relationships()
    FM-->>Agent: cross-domain connections
    
    CM->>CM: merge & deduplicate results
    CM-->>Agent: ranked memory results
```

### Memory Types

| Type | Description | Use Case |
|------|-------------|----------|
| `TASK` | Task descriptions with embeddings | Semantic task matching |
| `PLAN` | Generated action plans | Plan reuse and learning |
| `CONVERSATION` | Chat messages | Session history |
| `CONTEXT` | General context | User preferences, state |
| `EMBEDDING` | Raw embeddings | Vector operations |

### Fact Types (Fabric IQ)

| Type | Description | Use Case |
|------|-------------|----------|
| `observation` | Current state facts | "Pipeline has 93% success rate" |
| `prediction` | ML-derived forecasts | "Customer has 82% churn risk" |
| `derived` | Cross-domain insights | Correlation analysis |
| `rule` | Business rules | Policy enforcement |

### CosmosDB Container Schema

**short_term_memory container:**
```json
{
  "id": "uuid",
  "content": "memory content text",
  "memory_type": "context|conversation|task|plan",
  "embedding": [0.123, ...],  // 3072 dimensions
  "session_id": "partition key",
  "user_id": "optional user identifier",
  "metadata": {},
  "created_at": "ISO timestamp",
  "updated_at": "ISO timestamp",
  "ttl": 3600  // Time-to-live in seconds
}
```

### Fabric IQ Fact Schema

**Facts in FactsMemory:**
```json
{
  "id": "fact-churn-cust-123",
  "fact_type": "prediction",
  "domain": "customer",
  "statement": "Customer 'Liam' has 82% churn risk",
  "confidence": 0.82,
  "evidence": ["cust-123", "tx-456"],
  "context": {
    "segment": "business",
    "risk_level": "critical",
    "tenure_months": 8
  },
  "embedding": [0.123, ...],
  "created_at": "ISO timestamp",
  "valid_until": "ISO timestamp"
}
```

### MCP Memory Tools

| Tool | Description |
|------|-------------|
| `store_memory` | Store content with embedding in short-term memory |
| `recall_memory` | Semantic search for relevant memories |
| `get_session_history` | Retrieve conversation history |
| `clear_session_memory` | Clear session memory |
| `next_best_action` | Analyze task, find similar tasks, generate plan |

### MCP Facts Memory Tools (Fabric IQ)

| Tool | Description |
|------|-------------|
| `search_facts` | Semantic search across all domain ontologies |
| `get_customer_churn_facts` | Customer churn analysis facts and predictions |
| `get_pipeline_health_facts` | CI/CD pipeline health and failure analysis |
| `get_user_security_facts` | User security alerts and authentication patterns |
| `cross_domain_analysis` | Cross-domain reasoning and correlation |
| `get_facts_memory_stats` | Facts memory statistics by domain |

### Environment Variables

**Facts Memory Configuration:**
```bash
FABRIC_ENDPOINT=https://<workspace>.fabric.microsoft.com
FABRIC_WORKSPACE_ID=<workspace-id>
FABRIC_ONTOLOGY_NAME=agent-ontology
```

---

## Deployment Architecture

Infrastructure and deployment view with Microsoft Fabric integration:

```mermaid
graph TB
    subgraph "Azure Subscription"
        subgraph "Resource Group"
            subgraph "Network Layer"
                VNet[Virtual Network<br/>10.0.0.0/16]
                PrivateSubnet[Private Endpoints Subnet<br/>10.0.1.0/24]
                AppSubnet[App Subnet<br/>10.0.2.0/24]
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
            
            subgraph "Memory Layer"
                CosmosDB[CosmosDB NoSQL<br/>Vector Search]
                AISearch[Azure AI Search<br/>Semantic Search]
            end
            
            subgraph "AI Layer"
                Foundry[Azure AI Foundry<br/>GPT + Embeddings]
            end
            
            subgraph "Fabric Layer"
                FabricCapacity[Fabric Capacity<br/>F2 SKU]
                OneLake[OneLake<br/>Ontology Storage]
                FabricIQ[Fabric IQ<br/>Facts Engine]
            end
            
            subgraph "Private Endpoints"
                PE_Storage[Storage PE]
                PE_Cosmos[CosmosDB PE]
                PE_Search[AI Search PE]
                PE_Foundry[Foundry PE]
                PE_Fabric[Fabric PE<br/>OneLake DFS/Blob]
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
    SystemNodePool --> AppSubnet
    
    AKS --> AKSIdentity
    AKS --> ACR
    
    APIM --> AKS
    APIM --> APIMIdentity
    
    %% Private Endpoints in dedicated subnet
    PE_Storage --> PrivateSubnet
    PE_Cosmos --> PrivateSubnet
    PE_Search --> PrivateSubnet
    PE_Foundry --> PrivateSubnet
    PE_Fabric --> PrivateSubnet
    
    %% MCP Pod connections via Private Endpoints
    MCPIdentity --> PE_Storage
    MCPIdentity --> PE_Cosmos
    MCPIdentity --> PE_Search
    MCPIdentity --> PE_Foundry
    MCPIdentity --> PE_Fabric
    
    %% Fabric connections
    PE_Fabric --> OneLake
    OneLake --> FabricCapacity
    FabricIQ --> FabricCapacity
    
    AKS --> AppInsights
    AppInsights --> LogWorkspace

    style AKS fill:#326ce5,color:#fff
    style SystemNodePool fill:#326ce5,color:#fff
    style APIM fill:#0078d4,color:#fff
    style FabricCapacity fill:#f25022,color:#fff
    style OneLake fill:#f25022,color:#fff
    style FabricIQ fill:#f25022,color:#fff
    style PE_Fabric fill:#ffb900,color:#000
```

### Fabric Infrastructure Details

The Microsoft Fabric integration provides ontology-grounded facts for AI agents:

| Component | Purpose | Private Link DNS |
|-----------|---------|-----------------|
| Fabric Capacity | Compute for Fabric workloads | N/A (management plane) |
| OneLake DFS | Data File System API for ontologies | `privatelink.dfs.fabric.microsoft.com` |
| OneLake Blob | Blob API for file access | `privatelink.blob.fabric.microsoft.com` |
| Fabric API | REST API access | `privatelink.api.fabric.microsoft.com` |

### Private Endpoint Architecture

```mermaid
graph LR
    subgraph "AKS Cluster"
        MCP[MCP Server Pod]
    end
    
    subgraph "Private DNS Zones"
        DNS1[privatelink.dfs.fabric.microsoft.com]
        DNS2[privatelink.blob.fabric.microsoft.com]
        DNS3[privatelink.api.fabric.microsoft.com]
    end
    
    subgraph "Private Endpoints"
        PE[Fabric Private Endpoint]
    end
    
    subgraph "Microsoft Fabric"
        OneLake[OneLake Storage]
        Workspace[Fabric Workspace]
    end
    
    MCP -->|DNS Query| DNS1
    MCP -->|DNS Query| DNS2
    MCP -->|DNS Query| DNS3
    
    DNS1 -->|Resolve to| PE
    DNS2 -->|Resolve to| PE
    DNS3 -->|Resolve to| PE
    
    PE -->|Private Link| OneLake
    PE -->|Private Link| Workspace
    
    style MCP fill:#326ce5,color:#fff
    style PE fill:#ffb900,color:#000
    style OneLake fill:#f25022,color:#fff
    style Workspace fill:#f25022,color:#fff
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
    
    ImagePush --> DeployMCP[kubectl apply:<br/>mcp-agents-deployment.yaml]
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

