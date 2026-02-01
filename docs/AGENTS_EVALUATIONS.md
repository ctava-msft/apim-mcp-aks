# NBA Agent Evaluation System

## Overview

This document describes the evaluation system for the Next Best Action (NBA) Agent deployed on Azure Kubernetes Service (AKS). The system uses the **Azure AI Evaluation SDK** to assess agent performance across three key dimensions:

1. **Intent Resolution** - How well the agent understands user intent
2. **Tool Call Accuracy** - How accurately the agent uses available tools  
3. **Task Adherence** - How well the agent's response adheres to assigned tasks

## Architecture


### MCP Evaluation Tools

Six **(6)** new MCP tools were added to the AKS FastAPI server that run evaluations on the cluster:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Developer Machine                                │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  evaluate_next_best_action_agent.py                                 │ │
│  │  - Loads evaluation data (JSONL)                                    │ │
│  │  - Connects via SSE to MCP server                                   │ │
│  │  - Calls evaluation MCP tools                                       │ │
│  │  - Aggregates and saves results                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP (LoadBalancer IP)
┌────────────────────────────────────────────────────────────────────────────┐
│                            Azure VNet                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  AKS Cluster (mcp-agents namespace)                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │  FastAPI MCP Server (next_best_action_agent.py)              │   │ │
│  │  │                                                                   │   │ │
│  │  │  Evaluation MCP Tools:                                            │   │ │
│  │  │  • get_evaluation_status                                          │   │ │
│  │  │  • evaluate_intent_resolution                                     │   │ │
│  │  │  • evaluate_tool_call_accuracy                                    │   │ │
│  │  │  • evaluate_task_adherence                                        │   │ │
│  │  │  • run_agent_evaluation                                           │   │ │
│  │  │  • run_batch_evaluation                                           │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼ Private Endpoint                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Azure OpenAI (cog-<redacted>)                                         │ │
│  │  - Model: gpt-5.2-chat                                                  │ │
│  │  - Public access: DISABLED                                              │ │
│  │  - Private endpoint: ENABLED                                            │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

## Evaluation MCP Tools

### 1. `get_evaluation_status`
Checks if evaluation tools are properly configured.

**Input:** None

**Output:**
```json
{
  "evaluation_available": true,
  "model_deployment": "gpt-5.2-chat",
  "evaluators": [
    "IntentResolutionEvaluator",
    "ToolCallAccuracyEvaluator", 
    "TaskAdherenceEvaluator"
  ]
}
```

### 2. `evaluate_intent_resolution`
Evaluates how well the agent identified user intent.

**Input:**
```json
{
  "query": "The user's question or request",
  "response": "The agent's response to evaluate"
}
```

**Output:**
```json
{
  "evaluator": "IntentResolutionEvaluator",
  "score": 4,
  "explanation": "The agent correctly identified the user's intent to...",
  "threshold_recommendation": 3,
  "passed": true
}
```

### 3. `evaluate_tool_call_accuracy`
Evaluates the accuracy of tool calls made by the agent.

**Input:**
```json
{
  "query": "User request",
  "tool_calls": [
    {
      "type": "tool_call",
      "tool_call_id": "call_001",
      "name": "next_best_action",
      "arguments": {"task": "..."}
    }
  ],
  "tool_definitions": [/* optional tool schemas */]
}
```

**Output:**
```json
{
  "evaluator": "ToolCallAccuracyEvaluator",
  "score": 5,
  "explanation": "Tool calls were appropriate for the request",
  "passed": true
}
```

### 4. `evaluate_task_adherence`
Evaluates if the response adheres to the assigned task.

**Input:**
```json
{
  "query": "The task/request",
  "response": "Agent's response",
  "tool_calls": [/* optional */],
  "system_message": "Optional system prompt"
}
```

**Output:**
```json
{
  "evaluator": "TaskAdherenceEvaluator",
  "flagged": false,
  "reasoning": "Response appropriately addresses the task",
  "passed": true
}
```

### 5. `run_agent_evaluation`
Runs all three evaluators on a single query/response pair.

**Input:**
```json
{
  "query": "User request",
  "response": "Agent response",
  "tool_calls": [/* optional */],
  "thresholds": {
    "intent_resolution": 3,
    "tool_call_accuracy": 3,
    "task_adherence": 3
  }
}
```

**Output:**
```json
{
  "query": "User request...",
  "evaluations": {
    "intent_resolution": {"score": 4, "passed": true},
    "tool_call_accuracy": {"score": 5, "passed": true},
    "task_adherence": {"flagged": false, "passed": true}
  },
  "all_passed": true
}
```

### 6. `run_batch_evaluation`
Runs evaluations on multiple items at once.

**Input:**
```json
{
  "evaluation_data": [
    {"query": "...", "response": "...", "tool_calls": [...]},
    {"query": "...", "response": "...", "tool_calls": [...]}
  ],
  "thresholds": {"intent_resolution": 3, "tool_call_accuracy": 3, "task_adherence": 3}
}
```

**Output:**
```json
{
  "summary": {
    "total_evaluated": 10,
    "metrics": {
      "intent_resolution": {"average_score": 4.2, "pass_rate": 90.0},
      "tool_call_accuracy": {"average_score": 4.5, "pass_rate": 100.0},
      "task_adherence": {"pass_rate": 80.0}
    }
  },
  "per_row_results": [...]
}
```

## Usage

### Running Evaluations

The evaluation script `evals/evaluate_next_best_action.py` supports three modes:

#### 1. Batch Mode (JSONL file)
```bash
python evaluate_next_best_action.py --data next_best_action_eval_data.jsonl --out ./results --direct
```

#### 2. Single Evaluation
```bash
python evaluate_next_best_action.py \
  --query "Analyze customer churn data" \
  --response "Here is my analysis plan..." \
  --out ./results --direct
```

#### 3. Live Agent Mode
```bash
python evaluate_next_best_action.py --target --out ./results --direct
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--data FILE` | JSONL file with evaluation data |
| `--query TEXT` | Single query to evaluate |
| `--response TEXT` | Single response to evaluate |
| `--target` | Run agent live for each query |
| `--out DIR` | Output directory for results |
| `--direct` | Use LoadBalancer IP (not APIM) |
| `--intent-threshold N` | Min passing score for intent (default: 3) |
| `--tool-threshold N` | Min passing score for tool calls (default: 3) |
| `--task-threshold N` | Min passing score for task adherence (default: 3) |
| `--strict` | Exit with code 1 if thresholds not met |
| `-v, --verbose` | Verbose logging |

### JSONL Data Format

Each line in the evaluation JSONL file should contain:

```json
{
  "query": "The user's question or task",
  "response": "The agent's response to evaluate",
  "tool_calls": [
    {
      "type": "tool_call",
      "tool_call_id": "call_001",
      "name": "next_best_action",
      "arguments": {"task": "..."}
    }
  ],
  "ground_truth_intent": "expected_intent_label",
  "system_message": "Optional system prompt context"
}
```

## Sample Evaluation Data

See `evals/next_best_action_eval_data.jsonl` for example evaluation data covering:
- Customer churn analysis
- CI/CD pipeline setup
- REST API design
- Data pipeline engineering
- Recommendation engine development
- Microservices migration
- Disaster recovery planning
- Database optimization
- Security implementation

## Output Files

The evaluation script generates two output files:

### 1. Detailed Results (`eval_results_YYYYMMDD_HHMMSS.json`)
Contains per-row evaluation scores and explanations.

### 2. Summary (`eval_summary_YYYYMMDD_HHMMSS.json`)
Contains aggregate metrics:
```json
{
  "total_evaluated": 10,
  "timestamp": "2026-02-01T12:30:00Z",
  "thresholds": {...},
  "metrics": {
    "intent_resolution": {
      "average_score": 4.2,
      "pass_rate": 90.0,
      "min": 3,
      "max": 5
    },
    ...
  },
  "overall_pass": true
}
```

## Dependencies

The evaluation system requires:

### On AKS (Container)
```
azure-ai-evaluation>=1.14.0
azure-identity>=1.19.0
```

### On Client (Developer Machine)
```
aiohttp
python-dotenv
```

## Known Limitations

### Azure AI Foundry vs Standard Azure OpenAI - CRITICAL

**Current Status:** The evaluation tools are implemented but **not fully functional** with Azure AI Foundry endpoints.

The Azure AI Evaluation SDK is designed for **standard Azure OpenAI endpoints** (`xxx.openai.azure.com`), not Azure AI Foundry project endpoints (`xxx.services.ai.azure.com/api/projects/...`).

**Issue:** The SDK internally constructs API paths like:
```
/api/projects/{project}/openai/deployments/{model}/chat/completions?api-version=xxx
```

This path structure and API versions may not be supported by the current Foundry deployment.

**Symptoms:**
- `400 Bad Request: API version not supported`
- `max_tokens is not supported with this model` errors
- Evaluation requests timing out
- Server disconnections during batch operations

**Resolution Applied:**

The `max_tokens` vs `max_completion_tokens` issue has been resolved by adding `is_reasoning_model=True` to all evaluator initializations. This is required for gpt-5.x series models.

```python
# Example fix applied to all 9 evaluator initializations:
evaluator = IntentResolutionEvaluator(
    model_config=model_config, 
    credential=credential, 
    is_reasoning_model=True  # Required for gpt-5.x models
)
```

### Current Working Components

| Component | Status | Notes |
|-----------|--------|-------|
| `ask_foundry` MCP tool | ✅ Working | Direct API calls work |
| `next_best_action` tool | ✅ Working | Uses same endpoint |
| Evaluation MCP tools defined | ✅ Ready | 6 tools in TOOLS list |
| Evaluation SDK initialized | ✅ Ready | Dependencies installed |
| `get_evaluation_status` | ✅ Working | Reports tools available |
| `evaluate_intent_resolution` | ✅ Working | Returns scores (1-5) |
| `evaluate_tool_call_accuracy` | ✅ Working | Requires tool_calls data |
| `evaluate_task_adherence` | ✅ Working | Returns flagged/passed status |
| Single evaluations | ✅ Working | ~10 seconds per evaluation |
| Batch evaluations (10 items) | ⚠️ Timeout | SSE connection times out after ~2 min |

### Role Assignments Required

The AKS workload identity needs these roles on the Azure AI Services resource:
- `Cognitive Services OpenAI Contributor`
- `Cognitive Services User`
- `Azure AI Developer` (for evaluation SDK agent operations)

## Troubleshooting

### "API version not supported"
The Azure AI Evaluation SDK may use an API version not supported by your deployment. Check the model's supported API versions and update `api_version` in the evaluator configurations.

### "PermissionDenied"
Ensure the AKS managed identity has the correct RBAC roles assigned to the Azure OpenAI resource.

### "Public access disabled"
This is expected - evaluations must run on the AKS cluster which has VNet access to Azure OpenAI via private endpoint.

## Related Documentation

- [AGENTS_ARCHITECTURE.md](AGENTS_ARCHITECTURE.md) - Overall system architecture
- [AGENTS_DEPLOYMENT_NOTES.md](AGENTS_DEPLOYMENT_NOTES.md) - Deployment procedures
- [AGENTS_IDENTITY_DESIGN.md](AGENTS_IDENTITY_DESIGN.md) - Workload identity configuration
