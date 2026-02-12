# Optional Exercise: Build Human-in-the-Loop Agent with Agent 365 Approval

**Duration:** 1 hour

## Overview

In this exercise, you will build a Human-in-the-Loop agent that requires administrator approval via Agent 365 for critical actions. This builds on the Autonomous Agent from [Exercise 2](../exercises/exercise_02_build_agents.md).

---

## Step 1: Create Agent Specification

In VS Code Explorer, navigate to and open `.speckit/specifications/approval_agent.spec.md`.

**Prompt Copilot with:**
> "Create an MCP agent specification for an agent that requires administrator approval via Agent 365 for high-value actions like deployments or data deletions. Governance Model should be Semi-Autonomous."

Key differences from autonomous agent:
- Governance Model: Semi-Autonomous
- Approval workflow integration
- Risk classification for actions

## Step 2: Implement Approval Agent with Copilot

In VS Code Explorer, navigate to and open `src/approval_agent.py`.

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

## Step 3: Configure Agent 365 Integration

Review the Agent 365 approval workflow:

In VS Code Explorer, navigate to and open:
- `agent365/workflows/agent_approval_logic_app.json`
- `agent365/teams/agent_approval_card.json`
- `agent365/teams/agent_approval_result_card.json`

## Step 4: Create Unit Tests

In VS Code Explorer, navigate to and open `tests/test_approval_agent.py`.

**Prompt Copilot with:**
> "Create pytest tests for the approval agent including tests for requires_approval() classification, request_approval() integration, and end-to-end approval flow with mock responses."

## Step 5: Run Unit Tests

```powershell
pytest tests/test_approval_agent.py -v
```

## Step 6: Deploy Approval Agent

```powershell
# Build and push
cd src
docker build -t approval-agent:latest .
docker tag approval-agent:latest "$REGISTRY/approval-agent:latest"
docker push "$REGISTRY/approval-agent:latest"

# Deploy
kubectl apply -f k8s/approval-agent-deployment.yaml

# Verify
kubectl get pods -n mcp-agents -l app=approval-agent
```

## Step 7: Functional Test Approval Flow

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

- [ ] Specification created with Semi-Autonomous governance
- [ ] Agent implemented with approval workflow
- [ ] requires_approval() correctly classifies actions
- [ ] Agent 365 integration configured
- [ ] Unit tests passing
- [ ] Agent deployed to AKS and running
- [ ] Approval flow working end-to-end

---

**Back to:** [Exercise 2: Build Agents](../exercises/exercise_02_build_agents.md)
