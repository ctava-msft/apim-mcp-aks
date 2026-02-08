# Exercise 3: Review Agents End-to-End

**Duration:** 30 minutes

## Overview

In this exercise, you will inspect the complete Azure Agents Control Plane to understand how governance, memory, observability, identity, and human oversight work together. You will also identify any problems with your deployed agents.

---

## Step 3.1: Check APIM – Policies (Management/Governance)

Azure API Management enforces governance policies for all agent traffic.

### Navigate to APIM

1. Open Azure Portal
2. Navigate to **API Management** → Your APIM instance
3. Go to **APIs** → **MCP API** → **Design** → **Inbound processing**

### Review Policies

Inspect the inbound policies:

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
                          counter-key="@(context.Request.Headers.GetValueOrDefault('Authorization',''))" />
        
        <!-- Quota enforcement -->
        <quota-by-key calls="10000" renewal-period="86400" 
                     counter-key="@(context.Request.Headers.GetValueOrDefault('Authorization',''))" />
        
        <!-- Correlation ID for tracing -->
        <set-header name="X-Correlation-ID" exists-action="skip">
            <value>@(Guid.NewGuid().ToString())</value>
        </set-header>
    </inbound>
</policies>
```

### Questions to Answer

| Question | Your Answer |
|----------|-------------|
| What happens if an unauthenticated request is made? | |
| What is the rate limit per minute? | |
| What is the daily quota? | |
| How are requests traced across services? | |

---

## Step 3.2: Check Cosmos DB (Short-Term Memory)

Cosmos DB stores session context and agent episodes.

### Navigate to Cosmos DB

1. Open Azure Portal
2. Navigate to **Azure Cosmos DB** → Your account
3. Go to **Data Explorer**

### Query Agent Sessions

```sql
SELECT * FROM c 
WHERE c.agent_id = 'autonomous-agent' 
ORDER BY c._ts DESC 
OFFSET 0 LIMIT 10
```

### Query Reinforcement Learning Episodes

```sql
SELECT * FROM c 
WHERE c.type = 'episode' 
ORDER BY c._ts DESC 
OFFSET 0 LIMIT 10
```

### Review Data Structure

| Container | Purpose | Key Fields |
|-----------|---------|------------|
| sessions | Session context | agent_id, user_id, context, timestamp |
| rl_episodes | Training episodes | agent_id, state, action, reward, next_state |
| approvals | Approval decisions | agent_id, action, approved, approver |

---

## Step 3.3: Check Azure AI Foundry / AI Search (Long-Term Memory)

Azure AI Search provides vector search for long-term memory retrieval.

### Navigate to AI Search

1. Open Azure Portal
2. Navigate to **Azure AI Search** → Your service
3. Go to **Indexes** → **agent-memory**

### Run a Search Query

```http
POST https://{search-service}.search.windows.net/indexes/agent-memory/docs/search?api-version=2024-07-01
Content-Type: application/json
api-key: {your-key}

{
  "search": "autonomous-agent",
  "select": "agent_id, content, timestamp",
  "top": 10,
  "orderby": "timestamp desc"
}
```

### Review AI Foundry Connection

```powershell
# Check Foundry endpoint
$env:FOUNDRY_PROJECT_ENDPOINT

# Test connection
az ai foundry project show --name <project-name> --resource-group <rg>
```

---

## Step 3.4: Check Fabric IQ / Storage Account (Facts/Ontology)

Ontologies provide grounded facts for agent reasoning.

### Navigate to Storage Account

1. Open Azure Portal
2. Navigate to **Storage Account** → Your account
3. Go to **Containers** → **ontologies**

### Review Ontology Files

```powershell
# List ontology files
az storage blob list --account-name <storage> --container-name ontologies --output table

# Download and review an ontology
az storage blob download --account-name <storage> --container-name ontologies --name customer_churn_ontology.json --file ./ontology.json

# View the ontology structure
Get-Content ./ontology.json | ConvertFrom-Json | Format-List
```

### Ontology Structure

```json
{
  "name": "customer_churn",
  "version": "1.0.0",
  "entities": ["Customer", "Product", "Subscription"],
  "relationships": ["purchases", "subscribes_to"],
  "facts": [
    {"subject": "Customer", "predicate": "has_churn_risk", "object": "ChurnScore"}
  ]
}
```

---

## Step 3.5: Check Log Analytics (Observability)

Azure Monitor collects logs, metrics, and traces from all agents.

### Navigate to Log Analytics

1. Open Azure Portal
2. Navigate to **Log Analytics** → Your workspace
3. Go to **Logs**

### Query Agent Logs

```kusto
// Agent container logs
ContainerLogV2
| where ContainerName contains "agent"
| where TimeGenerated > ago(1h)
| project TimeGenerated, ContainerName, LogMessage
| order by TimeGenerated desc
| take 100
```

### Query API Requests

```kusto
// APIM request logs
ApiManagementGatewayLogs
| where TimeGenerated > ago(1h)
| where OperationName contains "tools"
| project TimeGenerated, OperationName, ResponseCode, DurationMs
| order by TimeGenerated desc
```

---

## Step 3.6: Review Metrics and Traces via Azure Monitor and App Insights

### Navigate to Application Insights

1. Open Azure Portal
2. Navigate to **Application Insights** → Your instance
3. Go to **Transaction Search**

### View Distributed Traces

Filter by:
- **Operation Name:** `POST /message`
- **Time Range:** Last 1 hour

Click on a transaction to view the end-to-end trace.

### Query Custom Metrics

```kusto
// Tool call latency
customMetrics
| where name == "tool_call_duration_ms"
| summarize avg(value), percentile(value, 95), percentile(value, 99) by bin(timestamp, 5m)
| render timechart
```

### Query Error Rates

```kusto
// Error rate by agent
requests
| where timestamp > ago(1h)
| summarize 
    TotalRequests = count(),
    FailedRequests = countif(success == false),
    ErrorRate = round(countif(success == false) * 100.0 / count(), 2)
  by cloud_RoleName
| order by ErrorRate desc
```

---

## Step 3.7: Check Entra ID / RBAC

### Navigate to Entra ID

1. Open Azure Portal
2. Navigate to **Microsoft Entra ID** → **App registrations**
3. Find your agent's managed identity

### Review Role Assignments

```powershell
# List role assignments for agent identity
az role assignment list --assignee <agent-managed-identity-client-id> --all --output table
```

### Expected Roles

| Role | Resource | Purpose |
|------|----------|---------|
| Cognitive Services User | AI Foundry | LLM inference |
| Cosmos DB Data Contributor | Cosmos DB | Read/write sessions |
| Storage Blob Data Reader | Storage Account | Read ontologies |
| Search Index Data Reader | AI Search | Query long-term memory |

### Verify Workload Identity

```powershell
# Check service account annotation
kubectl get serviceaccount autonomous-agent-sa -n mcp-agents -o yaml

# Expected annotation:
# azure.workload.identity/client-id: <managed-identity-client-id>
```

---

## Step 3.8: Check Agent 365 (Human-in-the-Loop Approvals)

Agent 365 enables human oversight for critical agent decisions through approval workflows.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Your Agent                                   │
│                 Identifies high-value decision                        │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Agent 365 Approval Flow                            │
│           Logic App → Teams Adaptive Card → Human Response            │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                Approval Decision Recorded                             │
│                     Cosmos DB Audit Log                               │
└──────────────────────────────────────────────────────────────────────┘
```

### Review Approval Workflow Definition

```powershell
# Review the Logic App workflow definition
code agent365/workflows/agent_approval_logic_app.json

# Review Teams adaptive card templates
code agent365/teams/agent_approval_card.json
code agent365/teams/agent_approval_result_card.json
```

### Verify Logic App Execution

1. Open Azure Portal
2. Navigate to **Logic Apps** → **agent-approval-workflow**
3. Go to **Run history**
4. Review successful and failed runs

**For each run, verify:**
- Trigger received the approval request payload
- Teams adaptive card was posted successfully
- Approval response was received
- Callback to agent completed

### Check Teams Channel

1. Navigate to Microsoft Teams
2. Find the Agents Approval channel
3. Review recent approval cards:
   - Agent ID and action requested
   - Risk level classification
   - Context summary
   - Approve/Reject buttons

**Verify adaptive card contains:**
- Clear description of the action
- Risk level (high/medium/low)
- Sufficient context for decision-making
- Response buttons functioning

### Review Approval History in Cosmos DB

1. Navigate to Cosmos DB → **approvals** container
2. Query recent approvals:

```sql
SELECT * FROM c 
WHERE c.agent_id = 'approval-agent' 
ORDER BY c.timestamp DESC 
OFFSET 0 LIMIT 10
```

**Verify approval document structure:**

```json
{
  "id": "approval-request-id",
  "agent_id": "approval-agent",
  "action": "deploy_production",
  "approved": true,
  "approver": "user@contoso.com",
  "context": { "environment": "production", "version": "v2.0.0" },
  "timestamp": "2026-02-07T10:30:00Z"
}
```

### Generate Compliance Report

Query approval metrics in Application Insights:

```kusto
// Approval request metrics
customEvents
| where name == "ApprovalRequest"
| where timestamp > ago(7d)
| summarize 
    TotalRequests = count(),
    Approved = countif(customDimensions["approved"] == "true"),
    Rejected = countif(customDimensions["approved"] == "false"),
    AvgResponseTime = avg(todouble(customDimensions["response_time_seconds"]))
| extend ApprovalRate = round(Approved * 100.0 / TotalRequests, 2)
```

### Approval Best Practices

| Practice | Description |
|----------|-------------|
| **Set appropriate timeouts** | Don't block indefinitely; define escalation paths |
| **Provide rich context** | Include enough info for informed decisions |
| **Log everything** | Record all requests and decisions for audit |
| **Handle rejection gracefully** | Provide clear feedback when actions are rejected |
| **Define escalation paths** | What happens when approval times out? |

---

## Step 3.9: Identify Problems

Based on your review, identify any issues with your agents:

### Checklist

| Component | Status | Issue Found | Notes |
|-----------|--------|-------------|-------|
| APIM Policies | ✅ / ❌ | | |
| Cosmos DB Sessions | ✅ / ❌ | | |
| AI Search Memory | ✅ / ❌ | | |
| Ontology Storage | ✅ / ❌ | | |
| Log Analytics | ✅ / ❌ | | |
| App Insights Traces | ✅ / ❌ | | |
| Entra ID RBAC | ✅ / ❌ | | |
| Agent 365 Approvals | ✅ / ❌ | | |

### Common Problems

| Problem | Symptom | Solution |
|---------|---------|----------|
| High latency | P95 > 2s | Check AI Foundry throttling, scale pods |
| Failed tool calls | Error rate > 5% | Review logs for exceptions |
| Missing traces | No transactions in App Insights | Verify OpenTelemetry configuration |
| RBAC errors | 403 responses | Add missing role assignments |
| Approval timeouts | Approvals not completing | Check Logic App run history |

### Document Findings

Record any problems found for resolution in Exercise 4:

```
Problem 1: ___________________________________________
Symptom: ____________________________________________
Suspected Cause: ____________________________________

Problem 2: ___________________________________________
Symptom: ____________________________________________
Suspected Cause: ____________________________________
```

---

## Verification Checklist

- [ ] Reviewed APIM policies and understand governance controls
- [ ] Verified Cosmos DB is storing sessions and episodes
- [ ] Confirmed AI Search has agent memory data
- [ ] Reviewed ontology files in storage
- [ ] Queried Log Analytics for agent logs
- [ ] Viewed distributed traces in App Insights
- [ ] Verified RBAC role assignments
- [ ] Checked Agent 365 approval history
- [ ] Documented any problems found

---

**Next:** [Exercise 4: Fine-Tune Agent](exercise_04_fine_tune_agent.md)
