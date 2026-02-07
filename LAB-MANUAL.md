# Azure Agents Control Plane - Enterprise Lab Manual

## Welcome

Welcome to the Azure Agents Control Plane lab! In this hands-on experience, you will learn how to build enterprise-grade AI agents using Azure as your central control plane for governance, security, observability, and lifecycle management.

This lab demonstrates the core value proposition:

> **"Copilot helped me move faster, but Microsoft Azure, Foundry, and Fabric made this enterprise-ready."**

You will experience firsthand how Azure provides centralized governance and observability for AI agents—regardless of where those agents execute. Whether your agents run on Azure Kubernetes Service, AWS Lambda, GCP Cloud Run, or even on-premises, Azure acts as the enterprise control plane that ensures compliance, security, and operational excellence.

---

## Introduction

### What is the Azure Agents Control Plane?

The Azure Agents Control Plane is a comprehensive solution accelerator that governs the complete lifecycle of enterprise AI agents:

- **Analysis** - Understanding business problems and requirements
- **Design** - Creating agent specifications and architecture
- **Development** - Building agents with Copilot and SpecKit methodology
- **Testing** - Validating agent behavior and compliance
- **Deployment** - Releasing agents to governed runtimes
- **Observation** - Monitoring agent behavior and performance
- **Evaluation** - Measuring agent quality and task adherence
- **Fine-Tuning** - Optimizing agent responses through reinforcement learning

### Why Azure as the Enterprise Control Plane?

Traditional AI agent demos focus on getting something working quickly. However, enterprise deployments require:

- **Centralized Governance** - Policy enforcement, rate limiting, and compliance tracking
- **Identity-First Security** - Every agent has a Microsoft Entra ID identity with RBAC
- **API-First Architecture** - All agent operations flow through Azure API Management
- **Multi-Cloud Capable** - Agents can execute anywhere while being governed by Azure
- **Continuous Improvement** - Built-in evaluation and fine-tuning pipelines
- **Human Oversight** - Agent 365 integration for human-in-the-loop workflows

### How This Lab Differs from Simple Agent Demos

This is not a "hello world" agent tutorial. This lab is designed for enterprise practitioners who need to understand:

- How to govern AI agents at scale
- How to secure agent identities and enforce least-privilege access
- How to observe agent behavior across distributed systems
- How to continuously improve agent quality
- How to enable human approval workflows for critical decisions

You will work with production-grade components including Azure API Management, Azure Kubernetes Service, Azure AI Foundry, Cosmos DB, AI Search, Microsoft Fabric, and Agent 365.

---

## Audience & Prerequisites

### Target Audience

This lab is designed for:

- **Senior Developers** - Building production AI agent systems
- **Solution Architects** - Designing enterprise AI governance architectures
- **Platform Engineers** - Operating and scaling AI agent infrastructure
- **DevOps Engineers** - Implementing CI/CD for AI agents
- **Security Engineers** - Ensuring compliance and least-privilege access

### Prerequisites

#### Required Knowledge

- Familiarity with Python programming
- Basic understanding of REST APIs and HTTP protocols
- Experience with Azure services (at a conceptual level)
- Understanding of Kubernetes concepts (pods, services, deployments)
- Familiarity with Git and GitHub

#### Environment Assumptions

This lab assumes you have access to a pre-deployed Azure Agents Control Plane environment including:

- Azure Subscription with deployed infrastructure
- Azure Kubernetes Service (AKS) cluster with MCP agents
- Azure API Management (APIM) instance configured with OAuth
- Azure AI Foundry workspace with model deployments
- Azure Cosmos DB and Azure AI Search instances
- Microsoft Fabric workspace (optional for facts memory)
- Development machine or VM with required tools

#### Required Tools

Your environment should include:

- **Python 3.11+** with pip and uv
- **Azure CLI** (az) - authenticated to your subscription
- **Azure Developer CLI** (azd) - authenticated
- **kubectl** - configured to access your AKS cluster
- **Visual Studio Code** - with GitHub Copilot extension
- **Docker Desktop** (optional, for local testing)
- **Git** - for version control

#### Validation

Before proceeding, verify your environment:

```bash
# Check Azure CLI authentication
az account show

# Check kubectl access
kubectl get pods -n mcp-agents

# Check Python and uv
python --version
uv --version

# Check GitHub Copilot in VS Code
# Open VS Code and verify Copilot icon is active
```

---

## Lab Objectives

By completing this lab, you will be able to:

### 1. Governance

- ✅ Understand how Azure API Management acts as the governance gateway
- ✅ Inspect APIM policies that enforce rate limits, quotas, and compliance
- ✅ Trace requests through the control plane using distributed tracing
- ✅ Apply the Model Context Protocol (MCP) for standardized agent interfaces

### 2. Security & Identity

- ✅ Create and manage Microsoft Entra ID Agent identities
- ✅ Configure workload identity federation for AKS pods
- ✅ Implement least-privilege RBAC for agent tool access
- ✅ Validate keyless authentication patterns

### 3. Observability

- ✅ Monitor agent behavior using Azure Monitor and Application Insights
- ✅ Query telemetry with Kusto Query Language (KQL)
- ✅ Create dashboards for agent performance metrics
- ✅ Set up alerts for anomalous agent behavior

### 4. Evaluation & Fine-Tuning

- ✅ Run structured evaluations measuring intent resolution, tool accuracy, and task adherence
- ✅ Capture agent episodes for training data collection
- ✅ Label episodes with rewards (human or automated)
- ✅ Fine-tune models using Agent Lightning
- ✅ Deploy tuned models and validate improvements

### 5. Cross-Cloud Agent Execution

- ✅ Understand how agents can execute on Azure, AWS, GCP, or on-premises
- ✅ Validate that Azure maintains governance regardless of execution location
- ✅ Apply consistent identity and policy enforcement across clouds

---

## Solution Architecture Overview

### High-Level Architecture

The Azure Agents Control Plane consists of eight pillars that provide enterprise-grade governance:

| Pillar | Azure Service | Purpose |
|--------|---------------|---------|
| **API Governance** | APIM + MCP | Centralized policies, rate limits, routing |
| **Secure Runtime** | AKS + VNet | Private networking, workload identity |
| **Enterprise Memory** | Cosmos DB + AI Search + Fabric IQ | Short/long-term memory, ontology facts |
| **Agent Orchestration** | Azure AI Foundry | Multi-agent coordination, tool catalog |
| **Identity & Access** | Microsoft Entra ID | Agent identities, RBAC, conditional access |
| **Observability** | Azure Monitor + App Insights | Telemetry, tracing, compliance proof |
| **Secrets & Artifacts** | Key Vault + Storage | Keyless auth, artifact governance |
| **Human Oversight** | Agent 365 | Human-in-the-loop, approval workflows |

### Runtime Architecture

The runtime architecture describes how requests flow through the system:

```
┌─────────────────────────────────────────────────────────────────┐
│                         AI Agent Client                         │
│                 (Claude, ChatGPT, Custom Agent)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Azure API Management (APIM)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ OAuth Endpoints (/authorize, /token)                     │   │
│  │ MCP Endpoints (/sse, /message)                           │   │
│  │ Policies: Rate Limit, Quotas, Auth, Routing             │   │
│  └────────────────────────┬─────────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────────┘
                            │ Private Link
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│           Azure Kubernetes Service (AKS)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ MCP Server Pods (mcp-agents namespace)                   │   │
│  │ - Workload Identity Enabled                              │   │
│  │ - FastAPI MCP Server                                     │   │
│  │ - Tools: ask_foundry, next_best_action, etc.            │   │
│  └────────┬─────────────┬─────────────┬──────────────────────┘  │
└───────────┼─────────────┼─────────────┼─────────────────────────┘
            │             │             │
            ▼             ▼             ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │ Cosmos  │   │   AI    │   │ Azure   │
      │   DB    │   │ Search  │   │ Foundry │
      └─────────┘   └─────────┘   └─────────┘
           │             │             │
           └─────────────┴─────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ App Insights    │
              │ (Observability) │
              └─────────────────┘
```

### Buildtime Architecture

The buildtime architecture describes the developer workflow:

```
┌────────────────────────────────────────────────────────────┐
│              Developer Workflow                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  VS Code + GitHub Copilot                                 │
│         │                                                  │
│         ▼                                                  │
│  SpecKit Methodology                                       │
│  - Analyze requirements                                    │
│  - Define specifications (.speckit/*.spec.md)             │
│  - Generate agent code with Copilot                       │
│  - Write tests                                            │
│         │                                                  │
│         ▼                                                  │
│  GitHub (Push to Branch)                                  │
│         │                                                  │
│         ▼                                                  │
│  GitHub Actions CI/CD                                     │
│  ┌──────────────────┬────────────────────┐                │
│  │   App Build      │   Infra Build      │                │
│  │   - Lint        │   - Bicep validate  │                │
│  │   - Test        │   - Deploy infra    │                │
│  │   - Build image │   - Update config   │                │
│  └────────┬─────────┴────────┬───────────┘                │
│           │                  │                             │
│           └────────┬─────────┘                             │
│                    ▼                                       │
│           Human Approval (Agent 365)                      │
│                    │                                       │
│                    ▼                                       │
│           Deploy to AKS                                   │
│                    │                                       │
│                    ▼                                       │
│           Validate + Monitor                              │
└────────────────────────────────────────────────────────────┘
```

### Key Patterns

#### 1. Model Context Protocol (MCP)

All agent tools are exposed via the Model Context Protocol:

- **Tool Discovery** - Agents query available tools via `/tools/list`
- **Tool Execution** - Agents call tools via `/tools/call`
- **Server-Sent Events** - Real-time streaming of results
- **Schema Validation** - Pydantic models ensure type safety

#### 2. Identity Federation

Agent pods use workload identity to authenticate:

```
AKS Pod Identity → Federated Token → Entra Agent ID → Azure Services
```

This eliminates secrets in code and enables fine-grained RBAC.

#### 3. Memory Architecture

Agents maintain three types of memory:

| Memory Type | Storage | Purpose |
|-------------|---------|---------|
| **Short-Term** | Cosmos DB | Session context, recent interactions |
| **Long-Term** | AI Search | Persistent knowledge, vector embeddings |
| **Facts** | Fabric IQ | Ontology-grounded domain facts |

#### 4. Agent Lightning Feedback Loop

Continuous improvement happens through:

1. **Capture** - Record agent episodes (input, output, metadata)
2. **Label** - Assign rewards (human or automated evaluators)
3. **Build** - Create training datasets from positive episodes
4. **Train** - Fine-tune models via Azure OpenAI
5. **Deploy** - Promote tuned models to production

---
## Lab Exercises

### Exercise 0: Lab Orientation

**Duration:** 20 minutes

**Objective:** Validate your environment and understand the lab flow.

#### Step 1: Validate Azure Connectivity

Confirm you can access Azure resources:

```bash
# Check your Azure subscription
az account show

# List resource groups
az group list --output table

# Verify AKS cluster access
kubectl cluster-info

# List MCP agent pods
kubectl get pods -n mcp-agents

# Check pod logs (optional)
kubectl logs -n mcp-agents -l app=mcp-agents --tail=20
```

**Expected Result:** You should see your subscription details, AKS cluster information, and running MCP agent pods.

#### Step 2: Review Architecture Diagrams

Open the repository and review:

- `runtime.svg` - Shows the complete runtime request flow
- `buildtime.svg` - Shows the developer and CI/CD workflow
- `docs/AGENTS_ARCHITECTURE.md` - Detailed architecture documentation

#### Step 3: Explore the Specification Catalog

Browse the agent specifications in `.speckit/specifications/`:

```bash
cd .speckit/specifications
ls -la
```

Open one or more spec files and observe the structure:

- **Business Framing** - Problem statement and value proposition
- **Agent Architecture** - Single-agent vs. multi-agent design
- **MCP Tool Catalog** - Available tools and their schemas
- **Workflow Specification** - Step-by-step agent behavior
- **Testing Strategy** - How to validate the agent

#### Step 4: Test APIM Connectivity

Verify that APIM is accessible:

```bash
# Get APIM endpoint
APIM_ENDPOINT=$(az apim show --resource-group <your-rg> --name <your-apim> --query gatewayUrl -o tsv)

echo $APIM_ENDPOINT

# Test OAuth well-known endpoint
curl "$APIM_ENDPOINT/.well-known/oauth-authorization-server"
```

**Expected Result:** You should see OAuth metadata including authorization and token endpoints.

#### Step 5: Review the Lab Flow

You will complete the following exercises:

1. **Specification-Driven Agent Development** - Use Copilot + SpecKit to create an agent
2. **Deploying and Running Agents** - Deploy to AKS and test via APIM
3. **End-to-End Governance Review** - Inspect policies, identity, memory, and telemetry
4. **Fine-Tuning with Agent Lightning** - Capture, label, train, and deploy
5. **Evaluations** - Run structured evaluations and review metrics

---

### Exercise 1: Specification-Driven Agent Development

**Duration:** 60 minutes

**Objective:** Use GitHub Copilot and SpecKit methodology to design, implement, and test an AI agent.

#### Context

In this exercise, you will create a **Product Recommendation Agent** that helps e-commerce customers discover products based on their preferences and purchase history. You will follow the SpecKit methodology to ensure your agent is well-defined, testable, and enterprise-ready.

#### Step 1: Analyze Requirements

Open VS Code and create a new specification file:

```bash
touch .speckit/specifications/product_recommendation_agent.spec.md
code .speckit/specifications/product_recommendation_agent.spec.md
```

Use GitHub Copilot to help you define the specification. Start by typing:

```markdown
# Product Recommendation Agent Specification

## Overview

| Property | Value |
|----------|-------|
| **Spec ID** | `PRA-001` |
| **Version** | `1.0.0` |
| **Status** | `Active` |
| **Domain** | E-Commerce |
| **Agent Type** | Single Agent with Tools |
| **Governance Model** | Autonomous |

## Business Framing
```

Then use Copilot Chat to generate the rest of the specification:

**Copilot Prompt:**
```
Generate a complete SpecKit specification for a Product Recommendation Agent that:
- Analyzes customer purchase history
- Considers customer preferences and browsing behavior
- Integrates with product catalog
- Provides personalized recommendations
- Logs recommendations for analytics

Follow the structure in customer_churn_agent.spec.md
```

#### Step 2: Define Agent Specifications

Review the generated specification and ensure it includes:

- **Business Framing** - Problem statement and value proposition
- **Target Problems Addressed** - Clear problem/impact/solution table
- **Agent Architecture** - Diagram showing tool orchestration
- **Control Plane Integration** - Mapping to Azure services
- **MCP Tool Catalog** - List of required tools with schemas
- **Workflow Specification** - Sequence diagram for primary flow
- **Success Criteria** - Measurable outcomes
- **Testing Strategy** - Unit, integration, and functional tests

#### Step 3: Generate Agent Code

Create the agent implementation file:

```bash
touch src/product_recommendation_agent.py
code src/product_recommendation_agent.py
```

Use Copilot to generate the FastAPI MCP server implementation:

**Copilot Prompt:**
```python
# Create a FastAPI MCP server for the Product Recommendation Agent
# Requirements:
# - Implement MCP protocol (SSE, tools/list, tools/call)
# - Tools: get_customer_profile, get_purchase_history, search_products, generate_recommendations, log_recommendation
# - Use Azure SDKs for Cosmos DB, AI Search, and AI Foundry
# - Include OpenTelemetry instrumentation
# - Use managed identity authentication
```

Let Copilot generate the implementation. Review the code and ensure:

- ✅ FastAPI app with proper initialization
- ✅ MCP endpoints (`/sse`, `/message`, `/tools/list`, `/tools/call`)
- ✅ Tool functions with proper type hints and docstrings
- ✅ Azure SDK integration with DefaultAzureCredential
- ✅ Error handling and logging
- ✅ OpenTelemetry spans for distributed tracing

#### Step 4: Write Unit Tests

Create a test file:

```bash
touch tests/test_product_recommendation_agent.py
code tests/test_product_recommendation_agent.py
```

Use Copilot to generate tests:

**Copilot Prompt:**
```python
# Generate pytest unit tests for the Product Recommendation Agent
# Test:
# - Tool discovery (get_customer_profile, get_purchase_history, etc.)
# - Tool execution with valid inputs
# - Error handling for invalid inputs
# - MCP protocol compliance
# - Mock Azure SDK calls
```

Run the tests locally:

```bash
pytest tests/test_product_recommendation_agent.py -v
```

**Expected Result:** All tests should pass.

#### Step 5: Update Documentation

Update the main README to include your new agent:

```bash
code README.md
```

Add an entry in the agent catalog section:

```markdown
### Product Recommendation Agent

**Spec:** `.speckit/specifications/product_recommendation_agent.spec.md`
**Implementation:** `src/product_recommendation_agent.py`
**Domain:** E-Commerce
**Type:** Single Agent with Tools

Provides personalized product recommendations based on customer preferences and purchase history. Integrates with Fabric IQ for customer facts and Azure AI Search for product catalog.
```

#### Validation

✅ **Specification Completed** - All sections filled out with clear, actionable content  
✅ **Code Generated** - FastAPI MCP server with all required tools  
✅ **Tests Passing** - Unit tests validate tool behavior  
✅ **Documentation Updated** - Agent is cataloged in README  

---

### Exercise 2: Deploying and Running Agents

**Duration:** 45 minutes

**Objective:** Deploy your agent to AKS and validate it works through the APIM gateway.

#### Step 1: Build Container Image

Create a Dockerfile for your agent (if not already present):

```bash
code Dockerfile
```

Ensure it follows best practices:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Run the server
CMD ["uv", "run", "uvicorn", "src.product_recommendation_agent:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build the image:

```bash
# Authenticate to ACR
az acr login --name <your-acr-name>

# Build and push image
docker build -t <your-acr-name>.azurecr.io/product-recommendation-agent:v1 .
docker push <your-acr-name>.azurecr.io/product-recommendation-agent:v1
```

#### Step 2: Create Kubernetes Deployment

Create a Kubernetes deployment manifest:

```bash
touch k8s/product-recommendation-agent.yaml
code k8s/product-recommendation-agent.yaml
```

Define the deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-recommendation-agent
  namespace: mcp-agents
spec:
  replicas: 2
  selector:
    matchLabels:
      app: product-recommendation-agent
  template:
    metadata:
      labels:
        app: product-recommendation-agent
        azure.workload.identity/use: "true"
    spec:
      serviceAccountName: mcp-agents-sa
      containers:
      - name: agent
        image: <your-acr-name>.azurecr.io/product-recommendation-agent:v1
        ports:
        - containerPort: 8000
        env:
        - name: AZURE_CLIENT_ID
          value: "<agent-managed-identity-client-id>"
        - name: APPLICATIONINSIGHTS_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: app-insights-secret
              key: connection-string
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: product-recommendation-agent-svc
  namespace: mcp-agents
spec:
  selector:
    app: product-recommendation-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

Deploy to AKS:

```bash
kubectl apply -f k8s/product-recommendation-agent.yaml

# Verify pods are running
kubectl get pods -n mcp-agents -l app=product-recommendation-agent

# Check logs
kubectl logs -n mcp-agents -l app=product-recommendation-agent --tail=50
```

#### Step 3: Configure APIM Backend

Add your agent as a backend in APIM:

```bash
az apim backend create   --resource-group <your-rg>   --service-name <your-apim>   --backend-id product-recommendation-backend   --url "http://product-recommendation-agent-svc.mcp-agents.svc.cluster.local"   --protocol http   --description "Product Recommendation Agent"
```

Update APIM policy to route to your agent (manually in Azure Portal or via Bicep).

#### Step 4: Test via MCP Inspector

Use the MCP Inspector tool to test your agent:

```bash
# Install MCP Inspector (if not already installed)
npm install -g @modelcontextprotocol/inspector

# Get APIM OAuth token
TOKEN=$(az account get-access-token --resource <your-apim-client-id> --query accessToken -o tsv)

# Start inspector
mcp-inspector   --url "https://<your-apim>.azure-api.net/mcp/sse"   --header "Authorization: Bearer $TOKEN"
```

In the inspector UI:

1. **Discover Tools** - Click "List Tools" and verify you see all your agent tools
2. **Call a Tool** - Select `get_customer_profile` and provide a customer ID
3. **Inspect Response** - Verify the tool returns expected data
4. **Test Workflow** - Call tools in sequence to simulate a full recommendation workflow

#### Step 5: Validate MCP Protocol Compliance

Test the MCP endpoints directly:

```bash
# Test tool discovery
curl -X POST "https://<your-apim>.azure-api.net/mcp/message"   -H "Authorization: Bearer $TOKEN"   -H "Content-Type: application/json"   -d '{
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 1
  }'

# Expected: JSON array of tool definitions

# Test tool execution
curl -X POST "https://<your-apim>.azure-api.net/mcp/message"   -H "Authorization: Bearer $TOKEN"   -H "Content-Type: application/json"   -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "get_customer_profile",
      "arguments": {
        "customer_id": "CUST123"
      }
    },
    "id": 2
  }'

# Expected: Tool result with customer profile data
```

#### Validation

✅ **Container Image Built and Pushed** - Image available in ACR  
✅ **Pods Running in AKS** - `kubectl get pods` shows healthy pods  
✅ **APIM Backend Configured** - Backend exists and routes correctly  
✅ **MCP Tools Discoverable** - Inspector shows all tools  
✅ **MCP Tools Executable** - Tool calls return expected results  

---
### Exercise 3: End-to-End Governance Review

**Duration:** 45 minutes

**Objective:** Inspect and understand the governance mechanisms across APIM policies, identity, memory, and observability.

#### Step 1: Inspect APIM Policies

Navigate to Azure Portal → API Management → APIs → MCP API → Policies.

Review the inbound policies:

```xml
<policies>
    <inbound>
        <!-- OAuth validation -->
        <validate-jwt header-name="Authorization" failed-validation-httpcode="401">
            <openid-config url="https://login.microsoftonline.com/{tenant-id}/v2.0/.well-known/openid-configuration" />
            <audiences>
                <audience>{client-id}</audience>
            </audiences>
        </validate-jwt>
        
        <!-- Rate limiting -->
        <rate-limit-by-key calls="100" renewal-period="60" 
                          counter-key="@(context.Request.Headers.GetValueOrDefault("Authorization",""))" />
        
        <!-- Quota enforcement -->
        <quota-by-key calls="10000" renewal-period="86400" 
                     counter-key="@(context.Request.Headers.GetValueOrDefault("Authorization",""))" />
        
        <!-- Routing based on agent -->
        <set-backend-service backend-id="mcp-agents-backend" />
        
        <!-- Add correlation ID for tracing -->
        <set-header name="X-Correlation-ID" exists-action="skip">
            <value>@(Guid.NewGuid().ToString())</value>
        </set-header>
    </inbound>
</policies>
```

**Questions to Answer:**

- What happens if an unauthenticated request is made?
- What is the rate limit per minute?
- How are requests traced across services?

#### Step 2: Review Memory in Cosmos DB and AI Search

**Cosmos DB (Short-Term Memory):**

```bash
# Query recent agent sessions
az cosmosdb sql query   --account-name <your-cosmosdb>   --database-name agents   --container-name sessions   --query "SELECT * FROM c WHERE c.agent_id = 'product-recommendation-agent' ORDER BY c._ts DESC OFFSET 0 LIMIT 5"
```

Inspect a session document:

```json
{
  "id": "session-abc123",
  "agent_id": "product-recommendation-agent",
  "customer_id": "CUST123",
  "interactions": [
    {
      "timestamp": "2026-02-07T10:30:00Z",
      "tool": "get_customer_profile",
      "result": "..."
    },
    {
      "timestamp": "2026-02-07T10:30:15Z",
      "tool": "generate_recommendations",
      "result": "..."
    }
  ],
  "created_at": "2026-02-07T10:30:00Z",
  "_ts": 1707303000
}
```

**AI Search (Long-Term Memory):**

```bash
# Search for product recommendations
az search query   --service-name <your-search-service>   --index-name recommendations   --search-text "CUST123"   --select "customer_id,products,timestamp"
```

**Questions to Answer:**

- How long is short-term memory retained?
- What triggers moving memory to long-term storage?
- How are vector embeddings used for similarity search?

#### Step 3: Validate Identity and RBAC

Inspect the managed identity assigned to your agent pods:

```bash
# Get service account
kubectl get serviceaccount mcp-agents-sa -n mcp-agents -o yaml

# Verify workload identity annotation
# Expected: azure.workload.identity/client-id: <agent-managed-identity-id>

# Check federated credentials
az identity federated-credential list   --identity-name <agent-managed-identity>   --resource-group <your-rg>
```

Review RBAC assignments:

```bash
# List role assignments for the agent identity
az role assignment list   --assignee <agent-managed-identity-client-id>   --all
```

**Expected Roles:**

- `Cognitive Services User` - For Azure AI Foundry access
- `Storage Blob Data Contributor` - For Azure Storage access
- `Cosmos DB Account Reader Role` - For Cosmos DB access
- `Search Index Data Reader` - For AI Search access

**Questions to Answer:**

- What is the difference between a Service Principal and a Managed Identity?
- How does workload identity eliminate secrets?
- What principle guides the RBAC assignments (hint: least privilege)?

#### Step 4: Examine Telemetry in Application Insights

Navigate to Azure Portal → Application Insights → Transaction Search.

Filter by:
- **Operation Name:** `POST /tools/call`
- **Time Range:** Last 1 hour

Click on a transaction and review:

- **End-to-End Transaction** - Full request trace from APIM → AKS → Azure AI Foundry
- **Dependencies** - Calls to Cosmos DB, AI Search, Azure OpenAI
- **Performance** - Duration of each span
- **Custom Properties** - `agent_id`, `tool_name`, `customer_id`

Run a KQL query to analyze tool usage:

```kusto
dependencies
| where name contains "tools/call"
| extend tool_name = tostring(customDimensions["tool_name"])
| summarize count() by tool_name
| order by count_ desc
```

**Questions to Answer:**

- Which tool is called most frequently?
- What is the P95 latency for tool calls?
- Are there any failed requests?

#### Validation

✅ **APIM Policies Understood** - You can explain rate limiting, OAuth, and routing  
✅ **Memory Inspected** - You've queried short-term (Cosmos) and long-term (AI Search) memory  
✅ **Identity Validated** - You understand workload identity and RBAC assignments  
✅ **Telemetry Reviewed** - You've traced requests and analyzed performance  

---

### Exercise 4: Fine-Tuning with Agent Lightning

**Duration:** 60 minutes

**Objective:** Identify sub-optimal agent behavior, capture episodes, label with rewards, fine-tune a model, and validate improvements.

#### Context

Agent Lightning implements a reinforcement learning feedback loop:

1. **Capture** - Record agent interactions (episodes)
2. **Label** - Assign rewards (human or automated)
3. **Build** - Create training datasets
4. **Train** - Fine-tune models
5. **Deploy** - Promote tuned models to production

#### Step 1: Enable Episode Capture

Update your agent configuration to enable episode capture:

```bash
kubectl set env deployment/product-recommendation-agent   -n mcp-agents   ENABLE_EPISODE_CAPTURE=true
```

Verify the environment variable:

```bash
kubectl get deployment product-recommendation-agent -n mcp-agents -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="ENABLE_EPISODE_CAPTURE")].value}'
```

Make several agent requests to generate episodes:

```bash
# Use MCP Inspector or curl to call tools multiple times
for i in {1..10}; do
  curl -X POST "https://<your-apim>.azure-api.net/mcp/message"     -H "Authorization: Bearer $TOKEN"     -H "Content-Type: application/json"     -d '{
      "jsonrpc": "2.0",
      "method": "tools/call",
      "params": {
        "name": "generate_recommendations",
        "arguments": {
          "customer_id": "CUST'$i'"
        }
      },
      "id": '$i'
    }'
  sleep 2
done
```

#### Step 2: Review Captured Episodes

Query Cosmos DB for captured episodes:

```bash
az cosmosdb sql query   --account-name <your-cosmosdb>   --database-name agents   --container-name rl_episodes   --query "SELECT * FROM c WHERE c.agent_id = 'product-recommendation-agent' ORDER BY c._ts DESC OFFSET 0 LIMIT 5"
```

Inspect an episode document:

```json
{
  "id": "episode-xyz789",
  "agent_id": "product-recommendation-agent",
  "tool": "generate_recommendations",
  "input": {
    "customer_id": "CUST1"
  },
  "output": {
    "recommendations": [
      {"product_id": "PROD123", "score": 0.92},
      {"product_id": "PROD456", "score": 0.87}
    ]
  },
  "metadata": {
    "model": "gpt-4o-mini",
    "timestamp": "2026-02-07T11:00:00Z",
    "latency_ms": 324
  },
  "reward": null,
  "labeled_at": null,
  "_ts": 1707306000
}
```

#### Step 3: Label Episodes with Rewards

Label episodes using the CLI tool:

```bash
# List unlabeled episodes
python -m lightning.cli list-episodes   --agent-id product-recommendation-agent   --unlabeled-only

# Review an episode and assign reward
python -m lightning.cli label-episode   --episode-id episode-xyz789   --reward 0.9   --reason "High-quality recommendations matched customer preferences"

# Label multiple episodes
python -m lightning.cli label-batch   --agent-id product-recommendation-agent   --auto-label   --evaluator task_adherence   --min-score 0.7
```

The `auto-label` option uses the evaluation framework to automatically score episodes.

#### Step 4: Build Fine-Tuning Dataset

Create a training dataset from labeled episodes:

```bash
python -m lightning.cli build-dataset   --agent-id product-recommendation-agent   --name recommendations-v1   --min-reward 0.7   --output-format jsonl
```

This creates a JSONL file in the format required by Azure OpenAI fine-tuning:

```json
{"messages": [{"role": "system", "content": "You are a product recommendation agent..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are a product recommendation agent..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Review the dataset:

```bash
cat datasets/recommendations-v1.jsonl | head -5 | jq .
```

#### Step 5: Start Fine-Tuning Job

Submit a fine-tuning job to Azure OpenAI:

```bash
python -m lightning.cli start-training   --agent-id product-recommendation-agent   --dataset-name recommendations-v1   --base-model gpt-4o-mini   --training-epochs 3   --learning-rate-multiplier 0.1
```

Monitor the job status:

```bash
# Check training status
python -m lightning.cli get-training-status   --job-id <training-job-id>

# Sample output:
# {
#   "job_id": "ftjob-abc123",
#   "status": "running",
#   "progress": "45%",
#   "estimated_completion": "2026-02-07T12:30:00Z"
# }
```

Once the job completes (status: `succeeded`), note the fine-tuned model name (e.g., `ft:gpt-4o-mini:product-recommendation:abc123`).

#### Step 6: Deploy Tuned Model

Promote the tuned model to active:

```bash
python -m lightning.cli promote-model   --agent-id product-recommendation-agent   --model-name ft:gpt-4o-mini:product-recommendation:abc123   --mark-active
```

Update agent deployment to use the tuned model:

```bash
kubectl set env deployment/product-recommendation-agent   -n mcp-agents   USE_TUNED_MODEL=true
```

Restart pods to pick up the change:

```bash
kubectl rollout restart deployment/product-recommendation-agent -n mcp-agents
kubectl rollout status deployment/product-recommendation-agent -n mcp-agents
```

#### Step 7: Validate Improvements

Test the agent with the tuned model:

```bash
curl -X POST "https://<your-apim>.azure-api.net/mcp/message"   -H "Authorization: Bearer $TOKEN"   -H "Content-Type: application/json"   -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "generate_recommendations",
      "arguments": {
        "customer_id": "CUST1"
      }
    },
    "id": 1
  }'
```

Compare the results to the baseline model. Look for:

- **Higher relevance scores** for recommended products
- **Better alignment** with customer preferences
- **More contextual** recommendations

Run evaluations (covered in Exercise 5) to measure quantitative improvements.

#### Validation

✅ **Episodes Captured** - Multiple agent interactions recorded in Cosmos DB  
✅ **Episodes Labeled** - Rewards assigned (manual or automated)  
✅ **Dataset Built** - JSONL file created for fine-tuning  
✅ **Model Fine-Tuned** - Azure OpenAI job completed successfully  
✅ **Tuned Model Deployed** - Agent using new model in production  
✅ **Improvements Validated** - Better recommendations observed  

---
### Exercise 5: Evaluations

**Duration:** 60 minutes

**Objective:** Run structured evaluations to measure agent quality across intent resolution, tool accuracy, and task adherence.

#### Context

The Azure Agents Control Plane includes a comprehensive evaluation framework based on the Azure AI Evaluation SDK. Evaluations measure:

1. **Intent Resolution** - How well the agent understands user intent
2. **Tool Call Accuracy** - How accurately the agent uses available tools
3. **Task Adherence** - How well the agent follows specified workflows

#### Step 1: Prepare Evaluation Dataset

Create a JSONL file with test cases:

```bash
touch evals/product_recommendation_eval.jsonl
code evals/product_recommendation_eval.jsonl
```

Define test cases in JSONL format:

```json
{"query": "I'm looking for outdoor gear for a camping trip", "expected_intent": "product_search", "expected_tools": ["get_customer_profile", "search_products", "generate_recommendations"], "customer_id": "CUST1"}
{"query": "Show me products similar to my last purchase", "expected_intent": "product_similarity", "expected_tools": ["get_purchase_history", "search_similar_products", "generate_recommendations"], "customer_id": "CUST2"}
{"query": "What did I buy last month?", "expected_intent": "purchase_history", "expected_tools": ["get_purchase_history"], "customer_id": "CUST3"}
```

#### Step 2: Run Intent Resolution Evaluation

Test how well the agent identifies user intent:

```bash
python -m evals.evaluate_intent_resolution   --agent-id product-recommendation-agent   --eval-data evals/product_recommendation_eval.jsonl   --output evals/results/intent_resolution.json
```

Review the results:

```bash
cat evals/results/intent_resolution.json | jq .
```

Sample output:

```json
{
  "evaluator": "IntentResolutionEvaluator",
  "total_cases": 3,
  "passed": 3,
  "failed": 0,
  "average_score": 4.67,
  "threshold": 3,
  "cases": [
    {
      "query": "I'm looking for outdoor gear for a camping trip",
      "expected_intent": "product_search",
      "actual_intent": "product_search",
      "score": 5,
      "explanation": "Agent correctly identified product search intent",
      "passed": true
    }
  ]
}
```

#### Step 3: Run Tool Call Accuracy Evaluation

Test how accurately the agent calls tools:

```bash
python -m evals.evaluate_tool_accuracy   --agent-id product-recommendation-agent   --eval-data evals/product_recommendation_eval.jsonl   --output evals/results/tool_accuracy.json
```

Review the results:

```json
{
  "evaluator": "ToolCallAccuracyEvaluator",
  "total_cases": 3,
  "passed": 2,
  "failed": 1,
  "average_score": 3.67,
  "threshold": 3,
  "cases": [
    {
      "query": "I'm looking for outdoor gear for a camping trip",
      "expected_tools": ["get_customer_profile", "search_products", "generate_recommendations"],
      "actual_tools": ["get_customer_profile", "search_products", "generate_recommendations"],
      "score": 5,
      "explanation": "All expected tools were called in the correct order",
      "passed": true
    },
    {
      "query": "Show me products similar to my last purchase",
      "expected_tools": ["get_purchase_history", "search_similar_products", "generate_recommendations"],
      "actual_tools": ["get_purchase_history", "generate_recommendations"],
      "score": 3,
      "explanation": "Agent missed calling search_similar_products",
      "passed": true
    }
  ]
}
```

#### Step 4: Run Task Adherence Evaluation

Test how well the agent follows specified workflows:

```bash
python -m evals.evaluate_task_adherence   --agent-id product-recommendation-agent   --eval-data evals/product_recommendation_eval.jsonl   --output evals/results/task_adherence.json
```

Review the results:

```json
{
  "evaluator": "TaskAdherenceEvaluator",
  "total_cases": 3,
  "passed": 3,
  "failed": 0,
  "average_score": 4.33,
  "threshold": 3,
  "cases": [
    {
      "query": "I'm looking for outdoor gear for a camping trip",
      "score": 5,
      "explanation": "Agent followed the recommendation workflow perfectly",
      "passed": true
    }
  ]
}
```

#### Step 5: Run Batch Evaluation

Run all evaluations in a single command:

```bash
python -m evals.run_batch_evaluation   --agent-id product-recommendation-agent   --eval-data evals/product_recommendation_eval.jsonl   --evaluators intent_resolution,tool_accuracy,task_adherence   --output evals/results/batch_evaluation.json
```

#### Step 6: Generate Evaluation Report

Create a summary report:

```bash
python -m evals.generate_report   --input evals/results/batch_evaluation.json   --output evals/results/evaluation_report.md
```

Open the report:

```bash
code evals/results/evaluation_report.md
```

Sample report:

```markdown
# Product Recommendation Agent - Evaluation Report

**Date:** 2026-02-07  
**Agent ID:** product-recommendation-agent  
**Total Test Cases:** 3  

## Summary

| Evaluator | Avg Score | Passed | Failed | Pass Rate |
|-----------|-----------|--------|--------|-----------|
| Intent Resolution | 4.67 | 3 | 0 | 100% |
| Tool Accuracy | 3.67 | 2 | 1 | 67% |
| Task Adherence | 4.33 | 3 | 0 | 100% |

## Recommendations

1. **Tool Accuracy** - The agent occasionally misses calling intermediate tools (e.g., search_similar_products). Consider fine-tuning to improve tool orchestration.
2. **Intent Resolution** - Performing well across all test cases.
3. **Task Adherence** - Workflows are being followed correctly.

## Next Steps

- Capture more episodes with correct tool sequences
- Label episodes with high rewards
- Fine-tune model to improve tool orchestration
- Re-run evaluations to validate improvements
```

#### Step 7: Track Evaluation Metrics Over Time

Store evaluation results in Cosmos DB for historical tracking:

```bash
python -m evals.store_results   --input evals/results/batch_evaluation.json   --agent-id product-recommendation-agent   --version v1.0.0
```

Query historical results:

```bash
az cosmosdb sql query   --account-name <your-cosmosdb>   --database-name agents   --container-name evaluation_results   --query "SELECT c.agent_id, c.version, c.average_score FROM c WHERE c.agent_id = 'product-recommendation-agent' ORDER BY c._ts DESC"
```

#### Validation

✅ **Evaluation Dataset Created** - Test cases defined in JSONL format  
✅ **Intent Resolution Evaluated** - Score > 4.0 (target: 3.0+)  
✅ **Tool Accuracy Evaluated** - Score > 3.5 (target: 3.0+)  
✅ **Task Adherence Evaluated** - Score > 4.0 (target: 3.0+)  
✅ **Report Generated** - Markdown summary with recommendations  
✅ **Results Tracked** - Historical data stored for regression testing  

---

## Success Criteria

By completing this lab, you have demonstrated enterprise-grade AI agent governance. You should now be able to:

### 1. Governance ✅

- **Understand** how Azure API Management acts as the central governance gateway
- **Inspect** APIM policies including rate limits, quotas, OAuth validation, and routing
- **Trace** requests through the entire control plane using correlation IDs
- **Apply** the Model Context Protocol (MCP) for standardized agent interfaces

### 2. Security & Identity ✅

- **Create** Microsoft Entra ID Agent identities with least-privilege RBAC
- **Configure** workload identity federation for AKS pods
- **Eliminate** secrets in code through managed identities
- **Validate** keyless authentication to Azure services

### 3. Observability ✅

- **Monitor** agent behavior using Azure Monitor and Application Insights
- **Query** telemetry with Kusto Query Language (KQL)
- **Analyze** request traces and performance metrics
- **Identify** bottlenecks and failure patterns

### 4. Evaluation & Fine-Tuning ✅

- **Run** structured evaluations (intent resolution, tool accuracy, task adherence)
- **Capture** agent episodes for training data collection
- **Label** episodes with rewards (human or automated)
- **Fine-tune** models using Agent Lightning
- **Deploy** tuned models and validate improvements

### 5. Cross-Cloud Agent Execution ✅

- **Understand** how Azure governs agents regardless of execution location
- **Apply** consistent identity and policy enforcement
- **Validate** that governance flows through Azure even for non-Azure runtimes

### The Core Value Proposition

You have proven:

> **"Copilot helped me move faster, but Microsoft Azure, Foundry, and Fabric made this enterprise-ready."**

- **Copilot** accelerated your agent development through specification-driven code generation
- **Azure API Management** provided centralized governance and policy enforcement
- **Azure Kubernetes Service** offered a secure, scalable runtime with workload identity
- **Azure AI Foundry** enabled model hosting, fine-tuning, and evaluation
- **Cosmos DB, AI Search, and Fabric IQ** delivered enterprise-grade memory and facts
- **Azure Monitor and App Insights** ensured full observability and compliance
- **Agent 365** enabled human-in-the-loop approvals for critical decisions

---

## Optional Extensions (Stretch Goals)

### Extension 1: Cross-Cloud Agent Execution

**Objective:** Deploy an agent to AWS Lambda and maintain Azure governance.

**Steps:**

1. Package your agent as an AWS Lambda function
2. Deploy to AWS using AWS CDK or Terraform
3. Configure the Lambda to call Azure APIM for tool execution
4. Use Azure Managed Identity via OIDC federation for authentication
5. Validate that telemetry flows to Azure Application Insights
6. Confirm APIM policies are enforced for Lambda-originated requests

**Key Insight:** Azure controls governance even when agents execute outside Azure.

### Extension 2: Multi-Agent Orchestration

**Objective:** Build a multi-agent system where specialized agents collaborate.

**Agents:**

- **Intent Agent** - Classifies user intent
- **Product Agent** - Handles product catalog queries
- **Recommendation Agent** - Generates personalized recommendations
- **Orchestrator Agent** - Coordinates the workflow

**Steps:**

1. Define specifications for each agent
2. Implement agents as separate MCP servers
3. Deploy to AKS with service mesh (Istio or Linkerd)
4. Configure the orchestrator to route to specialized agents
5. Test end-to-end orchestration
6. Review distributed traces showing multi-agent coordination

### Extension 3: Advanced Human-in-the-Loop with Agent 365

**Objective:** Implement human approval for high-value recommendations.

**Steps:**

1. Register your agent in the Entra Agent Registry
2. Configure Agent 365 approval workflows
3. Integrate with Microsoft Teams for approval cards
4. Trigger approval requests when recommendation value exceeds $1000
5. Track approval decisions in Cosmos DB
6. Review audit trail for compliance

### Extension 4: Real-Time Streaming with SSE

**Objective:** Implement server-sent events for real-time agent responses.

**Steps:**

1. Enhance your agent to support SSE streaming
2. Stream partial results as they're generated
3. Test with MCP Inspector in streaming mode
4. Measure latency improvements (TTFB vs. full response)
5. Implement cancellation support

### Extension 5: Custom Evaluators

**Objective:** Create domain-specific evaluators for your use case.

**Steps:**

1. Define custom evaluation metrics (e.g., "Recommendation Diversity Score")
2. Implement custom evaluator using Azure AI Evaluation SDK
3. Run evaluations on historical episodes
4. Incorporate scores into Agent Lightning reward labeling
5. Track custom metrics over time

---

## References

### Azure Documentation

- [Azure AI Foundry](https://learn.microsoft.com/azure/ai-foundry/what-is-foundry)
- [Azure AI Foundry Agent Service](https://learn.microsoft.com/azure/ai-foundry/agents/overview)
- [Azure API Management](https://learn.microsoft.com/azure/api-management/)
- [Azure Kubernetes Service (AKS)](https://learn.microsoft.com/azure/aks/)
- [Microsoft Entra ID](https://learn.microsoft.com/entra/identity/)
- [Microsoft Entra Agent Identity](https://learn.microsoft.com/entra/workload-id/)
- [Azure Cosmos DB](https://learn.microsoft.com/azure/cosmos-db/)
- [Azure AI Search](https://learn.microsoft.com/azure/search/)
- [Microsoft Fabric](https://learn.microsoft.com/fabric/)
- [Azure Monitor & App Insights](https://learn.microsoft.com/azure/azure-monitor/overview)

### Agent Frameworks & Protocols

- [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
- [SpecKit Methodology](https://speckit.dev)
- [Agents 365](https://learn.microsoft.com/microsoft-365-copilot/extensibility/)
- [Azure AI Evaluation SDK](https://learn.microsoft.com/azure/ai-studio/how-to/develop/evaluate-sdk)

### Related Labs & Samples

- [Contoso Creative Writer Lab](https://github.com/Azure-Samples/contoso-creative-writer)
- [Azure Agents Control Plane Repository](https://github.com/ctava-msft/apim-mcp-aks)

### Architecture & Design Docs

- [AGENTS_ARCHITECTURE.md](docs/AGENTS_ARCHITECTURE.md) - System architecture diagrams
- [AGENTS_IDENTITY_DESIGN.md](docs/AGENTS_IDENTITY_DESIGN.md) - Identity architecture
- [AGENTS_AGENT_LIGHTNING_DESIGN.md](docs/AGENTS_AGENT_LIGHTNING_DESIGN.md) - Fine-tuning design
- [AGENTS_EVALUATIONS.md](docs/AGENTS_EVALUATIONS.md) - Evaluation framework
- [AGENTS_APPROVALS.md](docs/AGENTS_APPROVALS.md) - Agent 365 approvals

---

## Conclusion

Congratulations! You have completed the Azure Agents Control Plane lab. You now understand how to:

- **Build agents faster** with GitHub Copilot and SpecKit
- **Govern agents at scale** with Azure API Management and MCP
- **Secure agents properly** with Microsoft Entra Agent Identity
- **Observe agent behavior** with Azure Monitor and App Insights
- **Improve agents continuously** with Agent Lightning fine-tuning
- **Evaluate agent quality** with structured evaluation frameworks
- **Enable human oversight** with Agent 365 approvals

You are now equipped to build enterprise-grade AI agents that are secure, observable, compliant, and continuously improving—all governed by Azure as your control plane.

### Next Steps

1. **Apply to Your Use Case** - Adapt the patterns to your specific business problems
2. **Contribute Back** - Share your agent specifications and learnings
3. **Stay Updated** - Follow the Azure AI Foundry and Agent Service updates
4. **Join the Community** - Engage with the Azure AI community

### Feedback

We'd love to hear your feedback on this lab. Please share:

- What worked well?
- What was confusing?
- What would you like to see added?

Open an issue or discussion in the [GitHub repository](https://github.com/ctava-msft/apim-mcp-aks).

---

**Thank you for participating in this lab!** 🎉
