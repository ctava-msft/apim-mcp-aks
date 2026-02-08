# Exercise 4: Fine-Tune Agent

**Duration:** 1 hour

## Overview

In this exercise, you will use Agent Lightning to identify sub-optimal agent behavior captured in Exercise 3, fine-tune the model to correct it, deploy the fine-tuned model, and retest to validate improvements.

---

## Agent Lightning Overview

Agent Lightning implements a reinforcement learning feedback loop:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Capture   │ ──▶ │    Label    │ ──▶ │    Build    │
│  Episodes   │     │   Rewards   │     │   Dataset   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Deploy    │ ◀── │    Train    │ ◀── │   Upload    │
│   Model     │     │  Fine-Tune  │     │  to Azure   │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Step 4.1: Enable Episode Capture

Ensure episode capture is enabled for your agents:

```powershell
# Enable episode capture for autonomous agent
kubectl set env deployment/autonomous-agent -n mcp-agents ENABLE_EPISODE_CAPTURE=true

# Enable for approval agent
kubectl set env deployment/approval-agent -n mcp-agents ENABLE_EPISODE_CAPTURE=true

# Restart deployments
kubectl rollout restart deployment/autonomous-agent -n mcp-agents
kubectl rollout restart deployment/approval-agent -n mcp-agents
```

---

## Step 4.2: Generate Agent Interactions

Make several requests to generate episodes:

```powershell
# Port forward to agent
kubectl port-forward -n mcp-agents svc/autonomous-agent 8080:80

# Make varied test requests
$queries = @(
    "Analyze customer churn for Q4 2025",
    "Generate a sales report for the west region",
    "What are the top 10 products by revenue?",
    "Show me customer segments at risk",
    "Predict next month's sales forecast"
)

foreach ($query in $queries) {
    $body = @{
        jsonrpc = "2.0"
        id = [guid]::NewGuid().ToString()
        method = "tools/call"
        params = @{
            name = "analyze_query"
            arguments = @{ query = $query }
        }
    } | ConvertTo-Json -Depth 5
    
    $response = Invoke-RestMethod -Uri "http://localhost:8080/message" -Method Post -Body $body -ContentType "application/json"
    Write-Host "Query: $query"
    Write-Host "Response: $($response | ConvertTo-Json -Depth 5)"
    Write-Host "---"
    Start-Sleep -Seconds 2
}
```

---

## Step 4.3: Review Captured Episodes

Query Cosmos DB for captured episodes:

```powershell
# Using Azure CLI
az cosmosdb sql query `
  --account-name $env:COSMOSDB_ACCOUNT `
  --database-name agents `
  --container-name rl_episodes `
  --query "SELECT * FROM c WHERE c.agent_id = 'autonomous-agent' ORDER BY c._ts DESC OFFSET 0 LIMIT 10"
```

Alternatively, use the Lightning CLI:

```powershell
# List recent episodes
python -m src.lightning.cli list-episodes `
  --agent-id autonomous-agent `
  --limit 10
```

### Episode Structure

```json
{
  "id": "episode-abc123",
  "agent_id": "autonomous-agent",
  "timestamp": "2026-02-07T10:30:00Z",
  "state": {
    "query": "Analyze customer churn for Q4 2025",
    "context": { "user": "analyst@contoso.com" }
  },
  "action": {
    "tool": "analyze_churn",
    "arguments": { "period": "Q4 2025" }
  },
  "result": {
    "churn_rate": 0.12,
    "at_risk_customers": 450
  },
  "reward": null,
  "labeled": false
}
```

---

## Step 4.4: Label Episodes with Rewards

Review episodes and assign rewards based on quality:

### Manual Labeling

```powershell
# List unlabeled episodes
python -m src.lightning.cli list-episodes `
  --agent-id autonomous-agent `
  --unlabeled-only

# Review and label a specific episode
python -m src.lightning.cli label-episode `
  --episode-id episode-abc123 `
  --reward 0.9 `
  --reason "Correct tool selection, accurate response"

# Label a poor quality episode
python -m src.lightning.cli label-episode `
  --episode-id episode-def456 `
  --reward 0.2 `
  --reason "Wrong tool selected, incomplete response"
```

### Automated Labeling with Evaluators

Use the task adherence evaluator for batch labeling:

```powershell
# Auto-label using task adherence scores
python -m src.lightning.cli label-batch `
  --agent-id autonomous-agent `
  --auto-label `
  --evaluator task_adherence `
  --min-score 0.7

# Review labeling results
python -m src.lightning.cli list-episodes `
  --agent-id autonomous-agent `
  --labeled-only `
  --limit 20
```

### Reward Guidelines

| Reward Score | Quality Level | Criteria |
|--------------|---------------|----------|
| 0.9 - 1.0 | Excellent | Correct tool, complete response, accurate data |
| 0.7 - 0.89 | Good | Correct tool, mostly complete response |
| 0.5 - 0.69 | Acceptable | Partially correct, minor issues |
| 0.3 - 0.49 | Poor | Wrong approach but recoverable |
| 0.0 - 0.29 | Failed | Wrong tool, incorrect response, errors |

---

## Step 4.5: Build Fine-Tuning Dataset

Create a training dataset from labeled episodes:

```powershell
# Build dataset from high-quality episodes
python -m src.lightning.cli build-dataset `
  --agent-id autonomous-agent `
  --name autonomous-agent-v1 `
  --min-reward 0.7 `
  --output-format jsonl
```

This creates a JSONL file in the format expected by Azure OpenAI:

```json
{"messages": [{"role": "system", "content": "You are an analytics agent..."}, {"role": "user", "content": "Analyze customer churn for Q4 2025"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Review Dataset Statistics

```powershell
# View dataset stats
python -m src.lightning.cli dataset-info `
  --name autonomous-agent-v1
```

Expected output:

```
Dataset: autonomous-agent-v1
Episodes: 150
Avg Reward: 0.82
Min Reward: 0.70
Max Reward: 0.98
Token Count: 45,230
```

---

## Step 4.6: Start Fine-Tuning Job

Submit a fine-tuning job to Azure OpenAI:

```powershell
# Start training
python -m src.lightning.cli start-training `
  --agent-id autonomous-agent `
  --dataset-name autonomous-agent-v1 `
  --base-model gpt-4o-mini `
  --training-epochs 3 `
  --learning-rate-multiplier 0.1

# Note the job ID returned
# Job ID: ftjob-abc123xyz
```

### Monitor Training Progress

```powershell
# Check training status
python -m src.lightning.cli get-training-status `
  --job-id ftjob-abc123xyz
```

Expected output during training:

```
Job ID: ftjob-abc123xyz
Status: running
Progress: 45%
Current Epoch: 2/3
Training Loss: 0.412
Validation Loss: 0.438
Estimated Time Remaining: 12 minutes
```

Wait for training to complete (typically 15-30 minutes):

```
Job ID: ftjob-abc123xyz
Status: succeeded
Fine-tuned Model: ft:gpt-4o-mini:autonomous-agent:abc123
Training Loss: 0.318
Validation Loss: 0.342
Total Tokens: 45,230
Duration: 23 minutes
```

---

## Step 4.7: Deploy Fine-Tuned Model

### Promote Model to Active

```powershell
# Promote the fine-tuned model
python -m src.lightning.cli promote-model `
  --agent-id autonomous-agent `
  --model-name ft:gpt-4o-mini:autonomous-agent:abc123 `
  --mark-active
```

### Update Agent Deployment

```powershell
# Update agent to use fine-tuned model
kubectl set env deployment/autonomous-agent `
  -n mcp-agents `
  USE_TUNED_MODEL=true `
  TUNED_MODEL_NAME=ft:gpt-4o-mini:autonomous-agent:abc123

# Restart to apply changes
kubectl rollout restart deployment/autonomous-agent -n mcp-agents

# Wait for rollout
kubectl rollout status deployment/autonomous-agent -n mcp-agents
```

---

## Step 4.8: Retest Agent

### Run Same Test Queries

```powershell
# Port forward
kubectl port-forward -n mcp-agents svc/autonomous-agent 8080:80

# Run the same queries used earlier
foreach ($query in $queries) {
    $body = @{
        jsonrpc = "2.0"
        id = [guid]::NewGuid().ToString()
        method = "tools/call"
        params = @{
            name = "analyze_query"
            arguments = @{ query = $query }
        }
    } | ConvertTo-Json -Depth 5
    
    $response = Invoke-RestMethod -Uri "http://localhost:8080/message" -Method Post -Body $body -ContentType "application/json"
    Write-Host "Query: $query"
    Write-Host "Response: $($response | ConvertTo-Json -Depth 5)"
    Write-Host "---"
}
```

### Compare Before/After

| Metric | Before Fine-Tuning | After Fine-Tuning |
|--------|-------------------|-------------------|
| Tool Selection Accuracy | % | % |
| Response Completeness | % | % |
| Average Reward Score | | |
| Error Rate | % | % |

### Run Automated Comparison

```powershell
# Compare episode quality before and after
python -m src.lightning.cli compare-versions `
  --agent-id autonomous-agent `
  --before-date 2026-02-07T00:00:00Z `
  --after-date 2026-02-07T12:00:00Z
```

---

## Step 4.9: Validate Improvement in App Insights

Query Application Insights to validate improvements:

```kusto
// Compare error rates before and after fine-tuning
requests
| where timestamp > ago(24h)
| extend model_version = tostring(customDimensions["model_version"])
| summarize 
    TotalRequests = count(),
    ErrorRate = round(countif(success == false) * 100.0 / count(), 2),
    AvgDuration = round(avg(duration), 0)
  by model_version, bin(timestamp, 1h)
| order by timestamp desc
```

---

## Verification Checklist

- [ ] Episode capture enabled for agents
- [ ] Episodes generated through test interactions
- [ ] Episodes reviewed and labeled with rewards
- [ ] Fine-tuning dataset created with min reward threshold
- [ ] Fine-tuning job submitted and completed successfully
- [ ] Fine-tuned model deployed to agent
- [ ] Agent retest shows improvement
- [ ] Improvements validated in App Insights metrics

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No episodes captured | Verify ENABLE_EPISODE_CAPTURE=true, check Cosmos DB connection |
| Training job failed | Check dataset format, ensure sufficient high-quality episodes |
| Model not improving | Increase epochs, adjust learning rate, add more diverse episodes |
| Deployment issues | Verify model name, check agent environment variables |

---

**Next:** [Exercise 5: Evaluations](exercise_05_evaluations.md)
