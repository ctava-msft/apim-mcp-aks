# Exercise 2: Build Agents using GitHub Copilot and SpecKit

**Duration:** 2 hours

## Overview

In this exercise, you will use GitHub Copilot and the SpecKit methodology to specify, create, unit test, and deploy agents. You will build two agents:

1. **Autonomous Agent** - Operates independently without human intervention
2. **Human-in-the-Loop Agent** - Requires administrator approval via Agent 365 for critical actions

---

## Part A: Review the Project Constitution

Before creating agents, you must understand the governance framework defined in the constitution.

### Step A.1: Open the Constitution

```powershell
# View the constitution
code .speckit/constitution.md
```

### Step A.2: Identify Key Elements

As you review, document:

| Element | Your Findings |
|---------|---------------|
| Project Mission | |
| Core Principles | |
| Security Requirements | |
| Quality Standards | |
| Development Phases | |

### Step A.3: Review Example Specifications

```powershell
# View example specifications
code .speckit/specifications/customer_churn_agent.spec.md
code .speckit/specifications/devops_cicd_pipeline_agent.spec.md
```

---

## Part B: Build Autonomous Agent

### Step B.1: Create Agent Specification with Copilot

Use GitHub Copilot to help create your agent specification:

```powershell
# Create specifications directory if needed
New-Item -ItemType Directory -Path ".speckit/specifications" -Force

# Create your autonomous agent specification
code .speckit/specifications/autonomous_agent.spec.md
```

**Prompt Copilot with:**
> "Create an MCP agent specification for an autonomous analytics agent that can query data, generate reports, and send notifications without human intervention. Follow the SpecKit template format."

Your specification should include:
- Overview (Spec ID, Version, Domain, Governance Model: Autonomous)
- Business Framing
- MCP Tool Catalog
- Workflow Specification
- Success Metrics
- Security Requirements

### Step B.2: Implement Autonomous Agent with Copilot

Create the agent implementation:

```powershell
# Create agent directory
New-Item -ItemType Directory -Path "src/agents/autonomous_agent" -Force

# Create agent files
code src/agents/autonomous_agent/agent.py
```

**Prompt Copilot with:**
> "Implement an MCP-compliant FastAPI agent based on the autonomous_agent.spec.md specification. Include health endpoint, SSE endpoint, and message endpoint with tools/list and tools/call handlers."

### Step B.3: Create Unit Tests with Copilot

```powershell
# Create test file
code tests/test_autonomous_agent.py
```

**Prompt Copilot with:**
> "Create pytest unit tests for the autonomous agent including tests for health endpoint, MCP initialize, tools/list, and tools/call methods."

### Step B.4: Run Unit Tests

```powershell
# Navigate to project root
cd c:\Users\christava\Documents\src\github.com\ctava-msft\agents-top\apim-mcp-aks

# Run tests
pytest tests/test_autonomous_agent.py -v
```

### Step B.5: Deploy Autonomous Agent

```powershell
# Build Docker image
cd src/agents/autonomous_agent
docker build -t autonomous-agent:latest .

# Tag and push to ACR
$REGISTRY = $env:CONTAINER_REGISTRY
docker tag autonomous-agent:latest "$REGISTRY/autonomous-agent:latest"
docker push "$REGISTRY/autonomous-agent:latest"

# Deploy to AKS
kubectl apply -f k8s/autonomous-agent-deployment.yaml

# Verify deployment
kubectl get pods -n mcp-agents -l app=autonomous-agent
```

### Step B.6: Functional Test with Copilot

Use Copilot to test the deployed agent:

```powershell
# Port forward
kubectl port-forward -n mcp-agents svc/autonomous-agent 8080:80

# Test health
Invoke-RestMethod -Uri "http://localhost:8080/health"

# Test tools/list
$body = @{
    jsonrpc = "2.0"
    id = 1
    method = "tools/list"
    params = @{}
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8080/message" -Method Post -Body $body -ContentType "application/json"
```

---

## Part C: Build Human-in-the-Loop Agent with Agent 365 Approval

### Step C.1: Create Agent Specification

```powershell
# Create specification for human-in-the-loop agent
code .speckit/specifications/approval_agent.spec.md
```

**Prompt Copilot with:**
> "Create an MCP agent specification for an agent that requires administrator approval via Agent 365 for high-value actions like deployments or data deletions. Governance Model should be Semi-Autonomous."

Key differences from autonomous agent:
- Governance Model: Semi-Autonomous
- Approval workflow integration
- Risk classification for actions

### Step C.2: Implement Approval Agent with Copilot

```powershell
# Create agent directory
New-Item -ItemType Directory -Path "src/agents/approval_agent" -Force

# Create agent files
code src/agents/approval_agent/agent.py
```

**Prompt Copilot with:**
> "Implement an MCP agent that integrates with Agent 365 for human approval. Include a requires_approval() function that classifies actions by risk level, and a request_approval() function that sends approval requests to a Logic App endpoint."

Example approval logic:

```python
def requires_approval(action: str, params: dict) -> bool:
    """Determine if action requires human approval."""
    HIGH_RISK_ACTIONS = [
        "deploy_production",
        "delete_data",
        "modify_permissions",
        "update_configuration"
    ]
    HIGH_VALUE_THRESHOLD = 10000
    
    return (
        action in HIGH_RISK_ACTIONS or
        params.get("value", 0) > HIGH_VALUE_THRESHOLD
    )

async def execute_with_approval(action: str, params: dict) -> dict:
    """Execute action with approval check."""
    if requires_approval(action, params):
        approval = await request_approval(
            agent_id="approval-agent",
            action=action,
            context=params
        )
        if not approval.get("approved"):
            return {"status": "rejected", "reason": approval.get("reason")}
    
    return await execute_action(action, params)
```

### Step C.3: Configure Agent 365 Integration

Review the Agent 365 approval workflow:

```powershell
# View approval Logic App definition
code agent365/workflows/agent_approval_logic_app.json

# View Teams adaptive card templates
code agent365/teams/agent_approval_card.json
code agent365/teams/agent_approval_result_card.json
```

### Step C.4: Create Unit Tests

```powershell
# Create test file
code tests/test_approval_agent.py
```

**Prompt Copilot with:**
> "Create pytest tests for the approval agent including tests for requires_approval() classification, request_approval() integration, and end-to-end approval flow with mock responses."

### Step C.5: Run Unit Tests

```powershell
pytest tests/test_approval_agent.py -v
```

### Step C.6: Deploy Approval Agent

```powershell
# Build and push
cd src/agents/approval_agent
docker build -t approval-agent:latest .
docker tag approval-agent:latest "$REGISTRY/approval-agent:latest"
docker push "$REGISTRY/approval-agent:latest"

# Deploy
kubectl apply -f k8s/approval-agent-deployment.yaml

# Verify
kubectl get pods -n mcp-agents -l app=approval-agent
```

### Step C.7: Functional Test Approval Flow

```powershell
# Port forward
kubectl port-forward -n mcp-agents svc/approval-agent 8081:80

# Test action requiring approval
$body = @{
    jsonrpc = "2.0"
    id = 1
    method = "tools/call"
    params = @{
        name = "deploy_production"
        arguments = @{
            environment = "production"
            version = "v2.0.0"
        }
    }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:8081/message" -Method Post -Body $body -ContentType "application/json"
```

**Expected Behavior:**
1. Agent identifies action as high-risk
2. Approval request sent to Logic App
3. Teams adaptive card posted to approval channel
4. Agent waits for approval response
5. Action proceeds or is rejected based on response

---

## Verification Checklist

### Autonomous Agent
- [ ] Specification created following SpecKit template
- [ ] Agent implemented with MCP endpoints
- [ ] Unit tests passing
- [ ] Docker image built and pushed to ACR
- [ ] Agent deployed to AKS and running
- [ ] Functional tests passing

### Approval Agent
- [ ] Specification created with Semi-Autonomous governance
- [ ] Agent implemented with approval workflow
- [ ] requires_approval() correctly classifies actions
- [ ] Agent 365 integration configured
- [ ] Unit tests passing
- [ ] Agent deployed to AKS and running
- [ ] Approval flow working end-to-end

---

## Summary

You have now built two agents using GitHub Copilot and SpecKit:

| Agent | Governance Model | Approval Required |
|-------|------------------|-------------------|
| Autonomous Agent | Autonomous | No |
| Approval Agent | Semi-Autonomous | Yes (for high-risk actions) |

Both agents follow the same MCP protocol and integrate with the Azure Agents Control Plane for governance, observability, and security.

---

## Part D: Advanced - Multi-Agent Orchestration (Optional)

For organizations requiring specialized agents that collaborate, consider the orchestrator pattern:

### Orchestrator Pattern Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator Agent                          │
│            Routes requests to specialized agents                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Analytics    │   │ DevOps       │   │ Customer     │
│ Agent        │   │ Agent        │   │ Service Agent│
└──────────────┘   └──────────────┘   └──────────────┘
```

### Key Concepts

1. **Intent Classification** - Route requests to appropriate specialized agents:

```python
INTENT_PATTERNS = {
    "analytics": ["analyze", "report", "metrics", "dashboard"],
    "devops": ["deploy", "pipeline", "kubernetes", "container"],
    "customer": ["support", "ticket", "complaint", "feedback"],
}

def classify_intent(query: str) -> str:
    """Classify user intent based on query keywords."""
    query_lower = query.lower()
    for intent, keywords in INTENT_PATTERNS.items():
        if any(keyword in query_lower for keyword in keywords):
            return intent
    return "general"
```

2. **Agent-to-Agent Communication** - Use internal service discovery:

```python
AGENT_REGISTRY = {
    "analytics": "http://analytics-agent.mcp-agents.svc.cluster.local",
    "devops": "http://devops-agent.mcp-agents.svc.cluster.local",
    "customer": "http://customer-agent.mcp-agents.svc.cluster.local",
}

async def route_to_agent(intent: str, request: dict) -> dict:
    """Route request to the appropriate specialized agent."""
    agent_url = AGENT_REGISTRY.get(intent)
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{agent_url}/message", json=request)
        return response.json()
```

3. **Service Mesh (Optional)** - Add Istio for advanced traffic management:
   - Mutual TLS between agents
   - Distributed tracing across agents  
   - Circuit breaking for fault tolerance

4. **Parallel Execution** - Query multiple agents simultaneously:

```python
async def parallel_query(queries: Dict[str, str]) -> Dict[str, Any]:
    """Execute queries across multiple agents in parallel."""
    tasks = [
        route_to_agent(intent, {"query": query})
        for intent, query in queries.items()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(zip(queries.keys(), results))
```

For the full multi-agent orchestration exercise, see **Extension 2** in the [main lab manual](../LAB_MANUAL_BUILD_YOUR_OWN_AGENT.md#extension-2-multi-agent-orchestration).

---

**Next:** [Exercise 3: Review Agents End-to-End](exercise_03_review_end_to_end.md)
