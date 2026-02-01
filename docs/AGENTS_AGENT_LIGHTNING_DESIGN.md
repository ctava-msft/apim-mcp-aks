# Agent Lightning - Fine-Tuning and Behavior Optimization

Agent Lightning is the fine-tuning and behavior optimization system for MCP agents. It captures agent interactions, labels them with rewards, builds training datasets, and manages fine-tuned model deployments—all with Cosmos DB as the authoritative system of record.

## 🔄 The Feedback Loop (How It Improves Responses)

Agent Lightning implements a reinforcement learning feedback loop that continuously improves agent responses:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     AGENT LIGHTNING FEEDBACK LOOP                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. CAPTURE EPISODES              2. LABEL WITH REWARDS                  │
│  ┌─────────────────────┐          ┌─────────────────────┐                │
│  │ User asks question  │          │ Human reviews       │                │
│  │ Agent responds      │ ───────► │ OR eval scores      │                │
│  │ Episode stored      │          │ Reward: +1 or -1    │                │
│  │ in Cosmos DB        │          │                     │                │
│  └─────────────────────┘          └──────────┬──────────┘                │
│                                              │                           │
│  5. USE TUNED MODEL               4. TRAIN   ▼   3. BUILD DATASET        │
│  ┌─────────────────────┐          ┌─────────────────────┐                │
│  │ get_model_deployment│ ◄─────── │ Filter episodes     │                │
│  │ returns tuned model │   Fine-  │ with reward > 0.5   │                │
│  │ Better responses!   │   Tune   │ Create JSONL        │                │
│  └─────────────────────┘          └─────────────────────┘                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### How Each Step Works

| Step | Component | What Happens |
|------|-----------|--------------|
| **1. Capture** | `EpisodeCaptureHook` | Every tool call (ask_foundry, next_best_action, etc.) is recorded with input, output, and metadata |
| **2. Label** | `RewardWriter` | Humans or automated evaluators assign rewards (-1 to +1) to episodes |
| **3. Build** | `DatasetBuilder` | Episodes with positive rewards are converted to JSONL format for fine-tuning |
| **4. Train** | `TrainingRunner` | Azure OpenAI fine-tuning creates a tuned model |
| **5. Deploy** | `DeploymentRegistry` | The tuned model is promoted to active; `get_model_deployment()` returns it |

### Tools Using Tuned Models

When `USE_TUNED_MODEL=true`, the following tools use tuned models via `get_model_deployment()`:

| Tool | Function | Uses Tuned Model |
|------|----------|------------------|
| `ask_foundry` | Direct LLM Q&A | ✅ Yes |
| `next_best_action` | Task analysis and planning | ✅ Yes |
| `analyze_intent` | Intent categorization | ✅ Yes |
| `generate_plan` | Plan generation | ✅ Yes |
| `generate_plan_with_instructions` | Plan with memory context | ✅ Yes |

## 🔧 How Fine-Tuning and Deployment Works

### The Complete Pipeline

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                         FINE-TUNING & DEPLOYMENT PIPELINE                           │
└────────────────────────────────────────────────────────────────────────────────────┘

  1. BUILD DATASET              2. START TRAINING           3. AZURE OPENAI FINE-TUNE
  ┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
  │ DatasetBuilder  │          │ TrainingRunner  │          │ Azure OpenAI    │
  │ ─────────────── │   CLI    │ ─────────────── │   API    │ ─────────────── │
  │ Query episodes  │ ───────► │ Upload JSONL    │ ───────► │ Fine-tune job   │
  │ with rewards    │          │ Create FT job   │          │ (gpt-4o-mini)   │
  │ Filter by score │          │ Track in Cosmos │          │ Creates model   │
  │ Output JSONL    │          │                 │          │                 │
  └────────┬────────┘          └────────┬────────┘          └────────┬────────┘
           │                            │                            │
           ▼                            ▼                            ▼
     rl_datasets                 rl_training_runs             ft:gpt-4o-mini:...
     (Cosmos DB)                 (Cosmos DB)                  (Azure OpenAI)


  4. PROMOTE MODEL              5. AGENT USES MODEL
  ┌─────────────────┐          ┌─────────────────┐
  │ DeploymentRegistry         │ get_model_deployment()
  │ ─────────────── │   CLI    │ ─────────────── │
  │ Mark as active  │ ───────► │ Check registry  │ ──► Uses tuned model!
  │ Store in Cosmos │          │ Return model    │
  │                 │          │ name            │
  └────────┬────────┘          └─────────────────┘
           │
           ▼
     rl_deployments
     (Cosmos DB)
```

### What Each Component Does

#### 1. DatasetBuilder (`dataset_builder.py`)

Builds fine-tuning datasets from rewarded episodes:

```bash
python -m lightning.cli build-dataset --agent-id mcp-agents --name v1 --min-reward 0.5
```

- Queries `rl_episodes` and `rl_rewards` from Cosmos DB
- Filters episodes where average reward ≥ threshold
- Converts to JSONL format: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
- Stores dataset manifest in `rl_datasets`

#### 2. TrainingRunner (`training_runner.py`)

Manages Azure OpenAI fine-tuning jobs:

```bash
python -m lightning.cli train --dataset-id <id> --agent-id mcp-agents
```

- Uploads JSONL files to Azure OpenAI
- Calls `client.fine_tuning.jobs.create()` to start fine-tuning
- Polls for completion (Azure does the actual training)
- On success, records the tuned model name (e.g., `ft:gpt-4o-mini-2024-07-18:org::abc123`)

#### 3. DeploymentRegistry (`deployment_registry.py`)

The bridge between training and runtime usage:

| Method | Purpose |
|--------|---------|
| `promote(agent_id, run_id)` | Make a tuned model the active deployment |
| `get_active_model(agent_id)` | Get the currently active tuned model name |
| `rollback(agent_id)` | Revert to previous deployment |
| `deactivate(agent_id)` | Stop using tuned model, use base model |
| `list_deployments(agent_id)` | Show deployment history |
| `get_deployment_lineage(agent_id)` | Full trace: deployment → training run → dataset |

#### 4. Model Selection (`get_model_deployment()`)

At runtime, the agent selects which model to use:

```python
def get_model_deployment() -> str:
    """Selection order:
    1. Active tuned deployment from Cosmos DB (via DeploymentRegistry)
    2. TUNED_MODEL_DEPLOYMENT_NAME env var (fallback)
    3. Base model (FOUNDRY_MODEL_DEPLOYMENT_NAME)
    """
    if not USE_TUNED_MODEL:
        return FOUNDRY_MODEL_DEPLOYMENT_NAME  # e.g., "gpt-5.2-chat"
    
    # Query Cosmos for active tuned model
    if deployment_registry:
        tuned_model = deployment_registry.get_active_model(LIGHTNING_AGENT_ID)
        if tuned_model:
            return tuned_model  # e.g., "ft:gpt-4o-mini-2024-07-18:org::abc123"
    
    return FOUNDRY_MODEL_DEPLOYMENT_NAME
```

### Example Workflow (CLI, API, and MCP Tools)

The Lightning workflow can be accessed via CLI, REST API, or MCP tools:

#### Option 1: CLI Commands

```bash
# Step 1: Build dataset from rewarded episodes
python -m lightning.cli build-dataset --agent-id mcp-agents --name v1 --min-reward 0.5
# Output: Dataset created: dataset-abc123

# Step 2: Start fine-tuning
python -m lightning.cli train --dataset-id dataset-abc123 --agent-id mcp-agents
# Output: Training run: run-xyz789, Status: succeeded
#         Tuned model: ft:gpt-4o-mini-2024-07-18:org::abc123

# Step 3: Promote to active
python -m lightning.cli promote --run-id run-xyz789 --agent-id mcp-agents
# Output: Model promoted! Agent will now use ft:gpt-4o-mini-2024-07-18:org::abc123

# Step 4: Enable in Kubernetes
kubectl set env deployment/mcp-agents -n mcp-agents USE_TUNED_MODEL=true

# Now ask_foundry and next_best_action use the tuned model!
```

#### Option 2: REST API

```bash
# Step 1: Build dataset from rewarded episodes
curl -X POST http://<endpoint>/api/lightning/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "mcp-agents",
    "name": "v1",
    "min_reward": 0.5
  }'
# Response: {"dataset_id": "dataset-abc123", "training_count": 800, "validation_count": 200}

# Step 2: Start fine-tuning
curl -X POST http://<endpoint>/api/lightning/training \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "mcp-agents",
    "dataset_id": "dataset-abc123",
    "base_model": "gpt-4o-mini-2024-07-18"
  }'
# Response: {"run_id": "run-xyz789", "status": "running"}

# Step 3: Check training status
curl http://<endpoint>/api/lightning/training/run-xyz789?agent_id=mcp-agents
# Response: {"run_id": "run-xyz789", "status": "succeeded", "tuned_model": "ft:gpt-4o-mini:..."}

# Step 4: Promote to active
curl -X POST http://<endpoint>/api/lightning/deployments \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "mcp-agents",
    "run_id": "run-xyz789",
    "promoted_by": "admin@example.com"
  }'
# Response: {"deployment_id": "dep-456", "is_active": true}
```

#### Option 3: MCP Tools (via SSE/JSON-RPC)

```json
// Step 1: Build dataset from rewarded episodes
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "tools/call",
  "params": {
    "name": "lightning_build_dataset",
    "arguments": {
      "agent_id": "mcp-agents",
      "name": "v1",
      "min_reward": 0.5
    }
  }
}
// Response: {"dataset_id": "dataset-abc123", "training_count": 800}

// Step 2: Start fine-tuning
{
  "jsonrpc": "2.0",
  "id": "2",
  "method": "tools/call",
  "params": {
    "name": "lightning_train",
    "arguments": {
      "agent_id": "mcp-agents",
      "dataset_id": "dataset-abc123"
    }
  }
}
// Response: {"run_id": "run-xyz789", "status": "running"}

// Step 3: Check training status
{
  "jsonrpc": "2.0",
  "id": "3",
  "method": "tools/call",
  "params": {
    "name": "lightning_training_status",
    "arguments": {
      "agent_id": "mcp-agents",
      "run_id": "run-xyz789"
    }
  }
}
// Response: {"status": "succeeded", "tuned_model": "ft:gpt-4o-mini:..."}

// Step 4: Promote to active
{
  "jsonrpc": "2.0",
  "id": "4",
  "method": "tools/call",
  "params": {
    "name": "lightning_promote",
    "arguments": {
      "agent_id": "mcp-agents",
      "run_id": "run-xyz789"
    }
  }
}
// Response: {"deployment_id": "dep-456", "is_active": true, "tuned_model": "ft:gpt-4o-mini:..."}
```

#### Python SDK

```python
from lightning import (
    get_dataset_builder,
    get_training_runner,
    get_deployment_registry,
)

# Step 1: Build dataset
builder = get_dataset_builder()
dataset = builder.build_dataset(
    agent_id="mcp-agents",
    name="v1",
    min_reward=0.5,
)
print(f"Dataset: {dataset.id}, Examples: {dataset.training_count}")

# Step 2: Start fine-tuning
runner = get_training_runner()
run = runner.run_training(
    dataset_id=dataset.id,
    agent_id="mcp-agents",
    wait=True,  # Block until complete
)
print(f"Tuned model: {run.tuned_model_name}")

# Step 3: Promote to active
registry = get_deployment_registry()
deployment = registry.promote(
    agent_id="mcp-agents",
    training_run_id=run.id,
    promoted_by="admin@example.com",
)
print(f"Deployment active: {deployment.is_active}")

# Step 4: Verify model selection
model = registry.get_active_model("mcp-agents")
print(f"Active model: {model}")  # ft:gpt-4o-mini:...
```

### Rollback and Management

```bash
# View deployment history
python -m lightning.cli list-deployments --agent-id mcp-agents

# Rollback to previous deployment
python -m lightning.cli rollback --agent-id mcp-agents --reason "Performance regression"

# Deactivate tuned model (use base model)
python -m lightning.cli deactivate --agent-id mcp-agents --reason "Testing base model"

# View full lineage (deployment → training → dataset)
python -m lightning.cli lineage --agent-id mcp-agents
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Runtime                            │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ MCP Tools    │───▶│ Episode     │───▶│ Cosmos DB    │       │
│  │ (ask_foundry │    │ Capture Hook │    │ rl_episodes  │       │
│  │  next_best   │    └──────────────┘    └──────────────┘       │
│  │  action...)  │                                               │
│  └──────────────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Model        │◀─-─│ Deployment   │◀───│ Cosmos DB    │      │
│  │ Selection    │    │ Registry     │    │ rl_deployments│      │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                           │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Reward       │───▶│ Dataset      │───▶│ Training     │      │
│  │ Writer       │    │ Builder      │    │ Runner       │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Cosmos DB    │    │ Cosmos DB    │    │ Cosmos DB    │       │
│  │ rl_rewards   │    │ rl_datasets  │    │ rl_training  │       │
│  └──────────────┘    └──────────────┘    │ _runs        │       │
│                                          └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Components

### 1. RL Ledger (Cosmos DB)
The authoritative system of record for all RL artifacts:

| Container | Purpose | Partition Key |
|-----------|---------|---------------|
| `rl_episodes` | Agent interactions (input, tools, output) | `agent_id` |
| `rl_rewards` | Labels and scores attached to episodes | `agent_id` |
| `rl_datasets` | Dataset manifests for fine-tuning | `agent_id` |
| `rl_training_runs` | Training job records | `agent_id` |
| `rl_deployments` | Active tuned model registry | `agent_id` |

### 2. Episode Capture Hook
Captures agent interactions as training data:
- User input
- Tool calls (names, arguments, results)
- Assistant output
- Metadata (latency, tokens, model used)

### 3. Reward Writer
Attaches labels/scores to episodes:
- Human approval (👍/👎)
- Evaluation scores (0-1)
- Test results (pass/fail)
- Safety checks
- Golden conversation matching

### 4. Dataset Builder
Builds fine-tuning datasets from rewarded episodes:
- Queries Cosmos for episodes with rewards
- Filters by reward threshold
- Produces JSONL for Azure OpenAI fine-tuning
- Splits into train/validation

### 5. Training Runner
Manages Azure OpenAI fine-tuning jobs:
- Uploads training files
- Creates fine-tuning jobs
- Monitors progress
- Records results in Cosmos

### 6. Deployment Registry
Manages which tuned model is active:
- Promote tuned models to active
- Rollback to previous deployments
- Cache active deployment for performance

## 🏗️ Infrastructure Setup

The Lightning system requires a separate Cosmos DB database (`agent_rl`) with 5 containers.
This is automatically provisioned via `infra/app/lightning-cosmos.bicep` when deploying with `azd up`.

### Containers Created

| Container | Purpose | Partition Key |
|-----------|---------|---------------|
| `rl_episodes` | Agent interactions (input, tools, output) | `/agent_id` |
| `rl_rewards` | Labels and scores attached to episodes | `/agent_id` |
| `rl_datasets` | Dataset manifests for fine-tuning | `/agent_id` |
| `rl_training_runs` | Training job records | `/agent_id` |
| `rl_deployments` | Active tuned model registry | `/agent_id` |

### Manual Deployment

If the containers don't exist (e.g., older deployment), re-run:

```bash
azd provision
```

Or apply the Bicep module manually:

```bash
az deployment group create \
  --resource-group <your-rg> \
  --template-file infra/app/lightning-cosmos.bicep \
  --parameters parentAccountName=<cosmos-account-name>
```

### Verify Containers

Run the test script to verify containers exist:

```bash
python tests/test_demo_lightning_loop.py --direct
```

## ⚙️ Environment Variables

### Required for Cosmos DB

| Variable | Description | Default |
|----------|-------------|---------|
| `COSMOS_ACCOUNT_URI` | Cosmos DB endpoint URL | *(required)* |
| `COSMOS_DATABASE_NAME` | Database name | `agent_rl` |
| `COSMOS_AUTH_MODE` | `aad` or `key` | `aad` |
| `COSMOS_ACCOUNT_KEY` | Account key (if auth_mode=key) | *(optional)* |

### Container Names (optional overrides)

| Variable | Default |
|----------|---------|
| `COSMOS_CONTAINER_EPISODES` | `rl_episodes` |
| `COSMOS_CONTAINER_REWARDS` | `rl_rewards` |
| `COSMOS_CONTAINER_DATASETS` | `rl_datasets` |
| `COSMOS_CONTAINER_RUNS` | `rl_training_runs` |
| `COSMOS_CONTAINER_DEPLOYMENTS` | `rl_deployments` |
| `COSMOS_PARTITION_KEY_FIELD` | `agent_id` |

### Agent Lightning Toggles

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_LIGHTNING_CAPTURE` | Enable episode capture | `false` |
| `USE_TUNED_MODEL` | Use tuned model if available | `false` |
| `TUNED_MODEL_DEPLOYMENT_NAME` | Fallback tuned model name | *(optional)* |
| `LIGHTNING_AGENT_ID` | Agent identifier | `mcp-agents` |
| `LIGHTNING_DATA_DIR` | Output directory for datasets | `./data/finetune` |

### Training Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LIGHTNING_BASE_MODEL` | Base model for fine-tuning | `gpt-4o-mini-2024-07-18` |
| `LIGHTNING_EPOCHS` | Number of training epochs | `3` |
| `LIGHTNING_LR_MULTIPLIER` | Learning rate multiplier | `1.0` |
| `LIGHTNING_MODEL_SUFFIX` | Custom suffix for model name | *(optional)* |
| `LIGHTNING_POLL_INTERVAL` | Seconds between status checks | `30` |
| `LIGHTNING_MAX_WAIT_MINUTES` | Max wait for training | `120` |

### Reward Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LIGHTNING_EVAL_THRESHOLD` | Eval score below this = negative | `0.7` |
| `LIGHTNING_LATENCY_THRESHOLD_MS` | Latency above this = penalty | `10000` |
| `LIGHTNING_LATENCY_PENALTY` | Penalty value for high latency | `-0.1` |
| `LIGHTNING_MIN_REWARD` | Min avg reward for dataset inclusion | `0.0` |
| `LIGHTNING_TRAIN_SPLIT` | Training/validation split ratio | `0.8` |

## 🚀 Quick Start

### 1. Enable Episode Capture

```bash
export ENABLE_LIGHTNING_CAPTURE=true
export COSMOS_ACCOUNT_URI="https://your-cosmos.documents.azure.com:443/"
export LIGHTNING_AGENT_ID="mcp-agents"
```

### 2. Run Your Agent
Episodes are automatically captured from tool calls.

### 3. Label Episodes with Rewards

Using CLI:
```bash
# Build dataset from all episodes (includes auto-rewards)
python -m lightning.cli build-dataset \
  --agent-id mcp-agents \
  --name customer-service-v1 \
  --min-reward 0.5
```

Programmatically:
```python
from lightning import get_reward_writer

writer = get_reward_writer()

# Human approval
writer.record_human_approval(
    episode_id="ep-123",
    agent_id="mcp-agents",
    approved=True,
    reviewer="admin@example.com"
)

# Evaluation score
writer.record_eval_score(
    episode_id="ep-123",
    agent_id="mcp-agents",
    score=0.85,
    rubric="accuracy"
)

# Test result
writer.record_test_result(
    episode_id="ep-123",
    agent_id="mcp-agents",
    passed=True,
    test_name="intent_match"
)
```

### 4. Build Training Dataset

```bash
python -m lightning.cli build-dataset \
  --agent-id mcp-agents \
  --name v1 \
  --min-reward 0.5 \
  --description "First production dataset"
```

Output:
```
✅ Dataset created successfully!
   ID: ds-abc123
   Training examples: 800
   Validation examples: 200
   Training file: ./data/finetune/v1_train_20240131.jsonl
```

### 5. Run Fine-Tuning

```bash
python -m lightning.cli train \
  --dataset-id ds-abc123 \
  --agent-id mcp-agents \
  --base-model gpt-4o-mini-2024-07-18
```

Output:
```
🚀 Starting fine-tuning job...
   Run ID: run-xyz789
   Status: running
   ...
✅ Training completed successfully!
   Tuned model: ft:gpt-4o-mini:org::abc123
```

### 6. Promote Tuned Model

```bash
python -m lightning.cli promote \
  --run-id run-xyz789 \
  --agent-id mcp-agents \
  --promoted-by admin@example.com
```

Output:
```
✅ Model promoted successfully!
   Deployment ID: dep-456
   Tuned model: ft:gpt-4o-mini:org::abc123
```

### 7. Enable Tuned Model

```bash
export USE_TUNED_MODEL=true
```

The agent will now use the tuned model automatically.

### 8. Rollback if Needed

```bash
python -m lightning.cli rollback \
  --agent-id mcp-agents \
  --reason "Performance regression detected"
```

## 📖 CLI Reference

```bash
# Dataset commands
python -m lightning.cli build-dataset --help
python -m lightning.cli build-golden --help
python -m lightning.cli list-datasets --agent-id mcp-agents

# Training commands
python -m lightning.cli train --help
python -m lightning.cli list-runs --agent-id mcp-agents
python -m lightning.cli status --run-id <id> --agent-id mcp-agents

# Deployment commands
python -m lightning.cli promote --help
python -m lightning.cli rollback --help
python -m lightning.cli deactivate --agent-id mcp-agents
python -m lightning.cli list-deployments --agent-id mcp-agents
python -m lightning.cli lineage --agent-id mcp-agents

# Health check
python -m lightning.cli health
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/test_lightning.py -v
```

Run the demo script:
```bash
python scripts/demo_lightning_loop.py
```

## 🔒 Security Considerations

### Secret Redaction
The Episode Capture Hook automatically redacts:
- Bearer tokens
- API keys
- Passwords
- Connection strings
- Azure account keys

### Cosmos DB Authentication
Prefer AAD authentication (`COSMOS_AUTH_MODE=aad`) over key-based auth.

### Data Retention
Configure Cosmos DB TTL policies for automatic cleanup of old episodes.

## 📊 Model Selection Logic

When `USE_TUNED_MODEL=true`, model selection follows this order:

1. **Cosmos Deployment Registry**: Check for active deployment for the agent
2. **Environment Variable**: Fall back to `TUNED_MODEL_DEPLOYMENT_NAME`
3. **Base Model**: Use `FOUNDRY_MODEL_DEPLOYMENT_NAME`

Cosmos lookups are cached with a configurable TTL (default 60 seconds) to avoid hot-loop reads.

## 🔄 Workflow: Golden Conversations

For curated training data:

1. Create a JSONL file with golden conversations:
```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

2. Build dataset from golden file:
```bash
python -m lightning.cli build-golden \
  --agent-id mcp-agents \
  --name golden-v1 \
  --golden-file ./data/golden_conversations.jsonl
```

## 🐛 Troubleshooting

### Episodes not being captured
- Check `ENABLE_LIGHTNING_CAPTURE=true`
- Check Cosmos DB connectivity: `python -m lightning.cli health`
- Check logs for capture errors

### Training job fails
- Check dataset format: ensure JSONL is valid
- Check base model availability
- Check Azure OpenAI quotas

### Tuned model not being used
- Check `USE_TUNED_MODEL=true`
- Check for active deployment: `python -m lightning.cli list-deployments --agent-id mcp-agents`
- Check Cosmos DB connectivity

## 📁 File Structure

```
src/lightning/
├── __init__.py           # Module exports
├── rl_ledger_cosmos.py   # Cosmos DB ledger for all RL artifacts
├── episode_capture.py    # Episode capture hook
├── reward_writer.py      # Reward/label writer
├── dataset_builder.py    # Dataset building from episodes
├── training_runner.py    # Azure OpenAI fine-tuning runner
├── deployment_registry.py # Tuned model deployment management
└── cli.py               # Command-line interface

tests/
└── test_lightning.py    # Test suite

scripts/
└── demo_lightning_loop.py # End-to-end demo script
```

## 🤝 Integration with next_best_action_agent.py

Agent Lightning is integrated into `next_best_action_agent.py` with minimal changes:

1. **Import**: Lightning modules are imported at startup
2. **Episode Capture**: Tool calls are captured via `execute_tool()` wrapper
3. **Model Selection**: `get_model_deployment()` selects base or tuned model
4. **Graceful Degradation**: All Lightning operations fail gracefully

## 📈 Metrics and Lineage

View complete deployment lineage:
```bash
python -m lightning.cli lineage --agent-id mcp-agents
```

Output shows:
- Deployment history
- Training runs that produced each model
- Datasets used for training
- Reward thresholds applied
- Who promoted/rolled back each deployment
