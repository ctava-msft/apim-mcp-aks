# Exercise 1: Lab Review

**Duration:** 30 minutes

## Overview

This exercise introduces the lab environment, reviews the solution architecture, and validates that your environment is properly configured.

---

## Step 1.1: Review Lab Objectives

By completing this lab, you will learn how to:

### Building Your Agent
- ✅ Review the SpecKit constitution and understand governance principles
- ✅ Write a structured agent specification following the SpecKit methodology
- ✅ Implement an MCP-compliant agent using FastAPI
- ✅ Containerize your agent with Docker
- ✅ Deploy your agent to AKS as a new pod
- ✅ Integrate with the Azure Agents Control Plane infrastructure

### Enterprise Governance
- ✅ Understand how Azure API Management acts as the governance gateway
- ✅ Inspect APIM policies that enforce rate limits, quotas, and compliance
- ✅ Trace requests through the control plane using distributed tracing

### Security & Identity
- ✅ Configure workload identity federation for AKS pods
- ✅ Implement least-privilege RBAC for agent tool access
- ✅ Validate keyless authentication patterns

### Observability
- ✅ Monitor agent behavior using Azure Monitor and Application Insights
- ✅ Query telemetry with Kusto Query Language (KQL)

### Fine-Tuning & Evaluation
- ✅ Capture agent episodes for training data collection
- ✅ Label episodes with rewards (human or automated)
- ✅ Fine-tune models using Agent Lightning
- ✅ Run structured evaluations measuring intent resolution, tool accuracy, and task adherence

---

## Step 1.2: Review Solution Architecture

### Azure Agents Control Plane

The Azure Agents Control Plane governs the complete software development lifecycle of enterprise AI agents:

![Agent Software Development Lifecycle](../buildtime.svg)

- **Analysis** - Understanding business problems and requirements
- **Design** - Creating agent specifications and architecture
- **Development** - Building agents with Copilot and SpecKit methodology
- **Testing** - Validating agent behavior and compliance
- **Deployment** - Releasing agents to governed runtimes
- **Observation** - Monitoring agent behavior and performance
- **Fine-Tuning** - Optimizing agent responses through reinforcement learning
- **Evaluation** - Measuring agent quality and task adherence

### Architecture Components

| Component | Azure Service | Purpose |
|-----------|---------------|---------|
| API Gateway | Azure API Management | Governance, rate limiting, OAuth validation |
| Agent Runtime | Azure Kubernetes Service | Workload identity, container orchestration |
| Short-Term Memory | Azure Cosmos DB | Session context, episode storage |
| Long-Term Memory | Azure AI Search | Historical data, vector search |
| Facts/Ontology | Microsoft Fabric / Storage Account | Domain knowledge, grounded facts |
| AI Models | Azure AI Foundry | LLM inference, embeddings |
| Identity | Microsoft Entra ID | Agent identity, RBAC |
| Observability | Azure Monitor + App Insights | Metrics, traces, logs |
| Human Oversight | Agent 365 | Approval workflows, Teams integration |

### SpecKit Methodology

SpecKit is a **specification-driven development methodology** that ensures all agents are properly defined before implementation:

```
.speckit/
├── constitution.md           # Core principles and governance framework
└── specifications/           # Individual agent specifications
    ├── customer_churn_agent.spec.md
    ├── devops_cicd_pipeline_agent.spec.md
    └── your_agent.spec.md    # Your agent specification
```

---

## Step 1.3: Review Exercises

| Exercise | Duration | Description |
|----------|----------|-------------|
| **Exercise 1: Lab Review** | 30 min | Review objectives, architecture, validate environment |
| **Exercise 2: Build Agents** | 2 hr | Use GitHub Copilot and SpecKit to specify, create, test, and deploy agents |
| **Exercise 3: Review End-to-End** | 30 min | Inspect governance, memory, observability, and identify problems |
| **Exercise 4: Fine-Tune Agent** | 1 hr | Use Agent Lightning to fine-tune and correct agent behavior |
| **Exercise 5: Evaluations** | 1 hr | Use evaluation framework to measure task adherence |

**Total Lab Duration:** 5 hours

---

## Step 1.4: Validate Environment

Run the environment validation script to confirm everything is properly configured.

### Prerequisites Check

```powershell
# Check Python version (3.11+ required)
python --version

# Check Docker
docker --version

# Check Azure CLI
az --version

# Check kubectl
kubectl version --client

# Check you're logged in to Azure
az account show
```

### AKS Cluster Access

```powershell
# Verify AKS connection
kubectl get nodes

# Verify mcp-agents namespace exists
kubectl get namespace mcp-agents
```

### Environment Variables

Ensure these are set in your terminal session:

```powershell
# Required environment variables
$env:CONTAINER_REGISTRY      # e.g., "youracr.azurecr.io"
$env:AZURE_TENANT_ID         # Your Azure tenant ID
$env:FOUNDRY_PROJECT_ENDPOINT # Azure AI Foundry endpoint
$env:COSMOSDB_ENDPOINT        # CosmosDB endpoint
```

### Execute Environment Validation Test

Run the test script to confirm the MCP connection to AKS works:

```powershell
# Navigate to tests directory
cd tests

# Run the MCP connection test
python test_mcp_connection.py
```

**Expected Output:**

```
============================================================
MCP Connection Test
============================================================
✅ AKS cluster connection successful
✅ MCP agents namespace found
✅ MCP server pods running
✅ MCP health endpoint responding
✅ MCP initialize method successful
✅ MCP tools/list method successful
============================================================
All tests passed! Environment is ready.
============================================================
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| AKS connection failed | Run: `az aks get-credentials --resource-group <rg> --name <cluster>` |
| Namespace not found | Deploy base infrastructure: `azd provision` |
| Pods not running | Check logs: `kubectl logs -n mcp-agents -l app=mcp-server` |
| Health endpoint failing | Verify service: `kubectl get svc -n mcp-agents` |

---

## Verification Checklist

Before proceeding to Exercise 2, confirm:

- [ ] Reviewed lab objectives and understand what you will build
- [ ] Reviewed solution architecture and key Azure services
- [ ] Reviewed all 5 exercises and understand the lab flow
- [ ] Environment validation script executed successfully
- [ ] All required tools installed (Python, Docker, Azure CLI, kubectl)
- [ ] Connected to AKS cluster
- [ ] Environment variables configured

---

**Next:** [Exercise 2: Build Agents using GitHub Copilot and SpecKit](exercise_02_build_agents.md)
