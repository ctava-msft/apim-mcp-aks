# Exercise 4: Fine-Tune and Evaluate Agent

**Duration:** 1 hour

## Overview

In this exercise, you will establish a **baseline evaluation** of your agent, use Agent Lightning to capture episodes and fine-tune the model, deploy the fine-tuned model, and then **re-evaluate to measure improvement**. This closed-loop approach ensures that reinforcement learning produces measurable, validated gains.

---

## Why Evaluate Before and After Fine-Tuning?

Fine-tuning without measurement is guesswork. The evaluation framework provides three complementary dimensions that together capture whether the agent actually improved:

| Evaluator | Scale | What it Measures | Why It Matters for RL |
|-----------|-------|------------------|----------------------|
| **Intent Resolution** | 1-5 | Does the agent correctly understand user intent? | Fine-tuning should sharpen intent classification |
| **Tool Call Accuracy** | 1-5 | Does the agent select the right tools with correct parameters? | This is the **primary behavior** trained by RLHF |
| **Task Adherence** | flagged true/false | Does the agent complete the assigned task correctly? | End-to-end quality gate on response quality |

Without before/after evaluation, you cannot distinguish a model that improved from one that regressed or stayed flat.

---

## Agent Lightning + Evaluation Loop

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  FINE-TUNING & EVALUATION CLOSED LOOP                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. BASELINE EVAL                 2. CAPTURE EPISODES                    │
│  ┌─────────────────────┐          ┌─────────────────────┐                │
│  │ Run evals on base   │          │ User asks question  │                │
│  │ model: intent,      │ ───────► │ Agent responds      │                │
│  │ tool, task scores   │          │ Episode stored      │                │
│  │ Record baseline     │          │ in Cosmos DB        │                │
│  └─────────────────────┘          └──────────┬──────────┘                │
│                                              │                           │
│  6. POST-TRAINING EVAL            3. LABEL   ▼   4. BUILD & TRAIN        │
│  ┌─────────────────────┐          ┌─────────────────────┐                │
│  │ Re-run SAME evals   │ ◄─────── │ Reward episodes     │                │
│  │ Compare to baseline │   Deploy │ Build JSONL dataset │                │
│  │ Gate: must improve! │   tuned  │ Fine-tune on Azure  │                │
│  └─────────────────────┘   model  │ OpenAI              │                │
│         │                         └─────────────────────┘                │
│         ▼                                                                │
│  ┌──────────────────────┐                                                │
│  │ 7. DECISION GATE     │                                                │
│  │ Improved? → Keep     │                                                │
│  │ Regressed? → Rollback│                                                │
│  │ Flat? → More data    │                                                │
│  └──────────────────────┘                                                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Step 4.1: Prepare Evaluation Dataset with GitHub Copilot

Before any fine-tuning, create a consistent evaluation dataset that will be used for both baseline and post-training measurement. Use GitHub Copilot in Agent Mode to generate evaluation data grounded in your agent's specification, task instructions, and ontology facts.

### Generate Evaluation Data with Copilot

**Prompt Copilot (Agent Mode) with:**
> "Generate an evaluation dataset for my autonomous agent. Review the SpecKit specification file at `.speckit/specifications/` that corresponds to my agent's use case. Also review the domain-specific task instruction documents in `task_instructions/` and the ontology fact files in `facts/ontology/` to understand the agent's domain, intents, tools, and expected behaviors. Additionally, review the existing evaluation data in `evals/next_best_action_eval_data.jsonl` as a reference for the expected JSONL format and structure. Using all of this context, generate a JSONL evaluation file at `evals/autonomous_agent_eval.jsonl`. Each line should be a JSON object with: query, expected_intent (derived from the spec's workflow intents), expected_tools (derived from the spec's MCP Tool Catalog), expected_response_contains (keywords from the task instructions and ontology), and context (user roles from the spec's security requirements). Generate at least 8 diverse test cases covering the full range of intents, tools, and edge cases defined in the specification."

### What Copilot Will Do

Copilot will:
1. Read your agent's SpecKit specification to extract defined intents, tools, and workflows
2. Read the task instruction files in `task_instructions/` to extract domain-specific queries and expected outcomes
3. Read the ontology facts in `facts/ontology/` to extract entity types, properties, and domain terminology
4. Read the existing `evals/next_best_action_eval_data.jsonl` to match the expected JSONL format
5. Synthesize all of this into a coherent evaluation dataset that covers your agent's capabilities

### Verify the Generated Dataset

**Prompt Copilot (Agent Mode) with:**
> "Read the generated `evals/autonomous_agent_eval.jsonl` and verify that: (1) each line is valid JSON, (2) the expected_intent values match intents from the specification, (3) the expected_tools reference tools from the spec's MCP Tool Catalog, and (4) there are at least 8 test cases covering different intents. Report a summary."

> **Important:** Use the **same evaluation dataset** for both baseline and post-training evaluation. This is the only way to produce a valid comparison.

---

## Step 4.2: Run Baseline Evaluation (Before Fine-Tuning) with Copilot

Establish baseline scores on the **base model** before any fine-tuning. These scores are your control group.

**Prompt Copilot (Agent Mode) with:**
> "Set up a kubectl port-forward from the autonomous-agent service in the mcp-agents namespace on port 8080:80. Then check if a .venv virtual environment exists in the project root — if it doesn't, create one. Activate it, install the dependencies from src/requirements.txt, and run the baseline evaluation using: `python -m evals.evaluate_next_best_action --data evals/next_best_action_eval_data.jsonl --out evals/eval_results --direct --strict`. After the evaluation completes, read the generated `evals/eval_results/eval_summary_*.json` file and display the baseline scores for Intent Resolution, Tool Call Accuracy, and Task Adherence."

### Review Baseline Results

Copilot will display your baseline scores. Record them here — you will compare against these after fine-tuning:

| Evaluator | Baseline Score | Threshold | Status |
|-----------|---------------|-----------|--------|
| Intent Resolution | ___ / 5 | ≥ 3 | |
| Tool Call Accuracy | ___ / 5 | ≥ 3 | |
| Task Adherence | pass / fail | not flagged | |
| **Overall** | ___ | all pass | |

> **Checkpoint:** If baseline scores are already very high (e.g., all 5/5), fine-tuning may yield diminishing returns. Focus on the dimensions where the agent scores lowest.

---

## Step 4.3: Enable Episode Capture with Copilot

Now enable episode capture to collect training data.

**Prompt Copilot (Agent Mode) with:**
> "Enable episode capture for the autonomous agent and approval agent deployments on AKS. Set the ENABLE_EPISODE_CAPTURE=true environment variable on both the `autonomous-agent` and `approval-agent` deployments in the `mcp-agents` namespace using kubectl set env. Then restart both deployments with kubectl rollout restart and verify they are running with the new environment variable."

---

## Step 4.4: Generate Agent Interactions with Copilot

Make several requests to generate episodes for training data.

**Prompt Copilot (Agent Mode) with:**
> "Set up a kubectl port-forward from the autonomous-agent service in the mcp-agents namespace on port 8080:80. Then activate the `.venv` virtual environment and run `python scripts/generate_episodes.py` to send each evaluation query from `evals/autonomous_agent_eval.jsonl` to the agent. The script connects to the MCP SSE endpoint at `http://localhost:8080/runtime/webhooks/mcp/sse`, establishes a session, and sends JSON-RPC tool calls for each query with a 2-second pause between requests. Display the output showing how many episodes were generated successfully."

> **Tip:** You can also run `python scripts/generate_episodes.py --list-tools` to see all available tools on the agent before generating episodes.

---

## Step 4.5: Review Captured Episodes with Copilot

Query captured episodes via the MCP API.

**Prompt Copilot (Agent Mode) with:**
> "Set up a kubectl port-forward from the mcp-agents service in the mcp-agents namespace on port 8000:80. Then activate the `.venv` virtual environment and run `python scripts/list_episodes.py --port 8000 --agent-id mcp-agents --limit 10` to list the most recent 10 captured episodes. The script connects to the MCP SSE endpoint and calls the `lightning_list_episodes` tool. Display the results and summarize how many episodes were captured."

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

## Step 4.6: Label Episodes with Rewards using Copilot

Review episodes and assign rewards based on quality.

### Automated Labeling

**Prompt Copilot (Agent Mode) with:**
> "Set up a kubectl port-forward from the mcp-agents service in the mcp-agents namespace on port 8000:80. Then activate the `.venv` virtual environment and run `python scripts/label_episodes.py --port 8000 --agent-id mcp-agents --limit 20` to automatically label captured episodes with rewards. The script connects to the MCP SSE endpoint, lists all episodes via `lightning_list_episodes`, scores each one using quality heuristics (tool correctness, output completeness, error detection), and assigns rewards via `lightning_assign_reward`. Display the output showing the reward distribution summary."

### Manual Override

If you want to override specific episode labels, you can call the `lightning_assign_reward` MCP tool directly for individual episodes.

### Reward Guidelines

| Reward Score | Quality Level | Criteria |
|--------------|---------------|----------|
| 0.9 - 1.0 | Excellent | Correct tool, complete response, accurate data |
| 0.7 - 0.89 | Good | Correct tool, mostly complete response |
| 0.5 - 0.69 | Acceptable | Partially correct, minor issues |
| 0.3 - 0.49 | Poor | Wrong approach but recoverable |
| 0.0 - 0.29 | Failed | Wrong tool, incorrect response, errors |

---

## Step 4.7: Build Fine-Tuning Dataset with Copilot

Create a training dataset from labeled episodes.

**Prompt Copilot (Agent Mode) with:**
> "Build a fine-tuning dataset from labeled episodes for the autonomous-agent. Run: `python -m src.lightning.cli build-dataset --agent-id autonomous-agent --name autonomous-agent-v1 --min-reward 0.7 --output-format jsonl`. This will create a JSONL file in the Azure OpenAI fine-tuning format. After building, run `python -m src.lightning.cli dataset-info --name autonomous-agent-v1` to display the dataset statistics including episode count, average reward, and token count."

This creates a JSONL file in the format expected by Azure OpenAI:

```json
{"messages": [{"role": "system", "content": "You are an analytics agent..."}, {"role": "user", "content": "Analyze customer churn for Q4 2025"}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Expected Dataset Statistics

Copilot will display the dataset info. Expected output:

```
Dataset: autonomous-agent-v1
Episodes: 150
Avg Reward: 0.82
Min Reward: 0.70
Max Reward: 0.98
Token Count: 45,230
```

---

## Step 4.8: Start Fine-Tuning Job with Copilot

Submit a fine-tuning job to Azure OpenAI.

**Prompt Copilot (Agent Mode) with:**
> "Start a fine-tuning job on Azure OpenAI for the mcp-agents agent using the `lightning_start_training` MCP tool. Use the latest dataset built in Step 4.7, base model `gpt-4o-mini`, 3 epochs, and learning rate multiplier 1.0. Note the returned training run ID and AOAI job ID."

### Monitor Training Progress

After starting the job, monitor it using the monitoring script. This polls the Azure OpenAI API directly (via MCP) and updates the Cosmos DB record in real-time.

**Prompt Copilot (Agent Mode) with:**
> "Ensure kubectl port-forward is active on port 8000 to mcp-agents. Then activate .venv and run: `python scripts/monitor_training.py --training-run-id <training_run_id> --port 8000 --interval 30`. This will poll the AOAI API until the job completes or fails."

Expected output during training:

```
Monitoring training run: b0a935c0-f6d4-4e97-855e-a9b50dde2c8c
MCP endpoint: http://localhost:8000/runtime/webhooks/mcp
Poll interval: 30s | Timeout: 120m
------------------------------------------------------------
[13:15:23] Status: running | AOAI: ftjob-50b46088abde49e5869181f7a47b782b
[13:15:53] Status: running | AOAI: ftjob-50b46088abde49e5869181f7a47b782b
...
[13:52:38] Status: succeeded | AOAI: ftjob-50b46088abde49e5869181f7a47b782b | Metrics: training_loss=0.066 | trained_tokens=11367
------------------------------------------------------------
Training SUCCEEDED!
Fine-tuned model: gpt-4o-mini-2024-07-18.ft-50b46088abde49e5869181f7a47b782b

Next step: Deploy the model with:
  python scripts/deploy_finetuned_model.py \
    --training-run-id b0a935c0-f6d4-4e97-855e-a9b50dde2c8c \
    --port 8000
```

> **Important**: The `lightning_get_training_status` MCP tool polls the Azure OpenAI API on each call, syncing the real-time status back to Cosmos DB. This ensures the Cosmos record stays current with the actual AOAI job status.

### Interpreting Training Metrics

| Metric | What It Means | Watch For |
|--------|---------------|-----------|
| Training Loss | How well the model fits training data | Should decrease each epoch |
| Validation Loss | How well the model generalizes | Should track training loss; divergence = overfitting |
| Epochs | Full passes through training data | 3 is typical; increase if loss is still decreasing |
| Learning Rate Multiplier | Step size for weight updates | Reduce (e.g., 0.05) if loss is volatile |

---

## Step 4.9: Deploy Fine-Tuned Model with Copilot

### Promote and Deploy the Model

Use the deployment script to promote the model in Cosmos and update the AKS deployment in one step.

**Prompt Copilot (Agent Mode) with:**
> "Ensure kubectl port-forward is active on port 8000 to mcp-agents. Then activate .venv and run: `python scripts/deploy_finetuned_model.py --training-run-id <training_run_id> --port 8000 --deployment mcp-agents`. This will: (1) verify the training succeeded, (2) promote the model via MCP, (3) set USE_TUNED_MODEL=true and TUNED_MODEL_NAME on the K8s deployment, and (4) wait for rollout."

Expected output:

```
============================================================
Step 1: Checking training run status...
  Status: succeeded
  Model:  gpt-4o-mini-2024-07-18.ft-50b46088abde49e5869181f7a47b782b
  Metrics: {"training_loss": 0.066, "trained_tokens": 11367}

============================================================
Step 2: Promoting fine-tuned model...
  Deployment ID: 16b85e61-82e6-4cae-9df7-14c4e449b030
  Is Active: True
  Promoted At: 2026-02-16T00:06:22.464282

============================================================
Step 3: Updating AKS deployment 'mcp-agents'...
  Set USE_TUNED_MODEL=true
  Set TUNED_MODEL_NAME=gpt-4o-mini-2024-07-18.ft-50b46088abde49e5869181f7a47b782b
  Waiting for rollout...
  Rollout complete!
  Verified env vars:
    USE_TUNED_MODEL=true
    TUNED_MODEL_NAME=gpt-4o-mini-2024-07-18.ft-50b46088abde49e5869181f7a47b782b

============================================================
Deployment complete!
```

---

## Step 4.10: Post-Training Evaluation (After Fine-Tuning) with Copilot

Run the **same evaluation dataset** from Step 4.2 against the fine-tuned model. This is the critical comparison.

**Prompt Copilot (Agent Mode) with:**
> "Run the post-training evaluation using the same dataset and thresholds as the baseline in Step 4.2. Set up a kubectl port-forward from the autonomous-agent service in the mcp-agents namespace on port 8080:80. Then activate the .venv and run: `python -m evals.evaluate_next_best_action --data evals/next_best_action_eval_data.jsonl --out evals/eval_results --direct --strict`. After the evaluation completes, read the generated eval_summary JSON and compare the scores against the baseline results from Step 4.2. Display a before/after comparison table."

### Compare Before/After Results

Fill in the table below with your actual scores. Expected improvement ranges are provided based on typical fine-tuning results:

| Evaluator | Baseline (Step 4.2) | After Fine-Tuning | Expected Delta | Status |
|-----------|--------------------|--------------------|----------------|--------|
| Intent Resolution | ___ / 5 | ___ / 5 | +0.5 to +1.5 | |
| Tool Call Accuracy | ___ / 5 | ___ / 5 | +0.5 to +1.0 | |
| Task Adherence | pass/fail | pass/fail | fewer flags | |

### Run Automated Comparison

**Prompt Copilot (Agent Mode) with:**
> "Run an automated comparison of episode quality before and after fine-tuning using: `python -m src.lightning.cli compare-versions --agent-id autonomous-agent --before-date 2026-02-07T00:00:00Z --after-date 2026-02-07T12:00:00Z`. Display the comparison table with intent resolution, tool accuracy, and task adherence metrics."

### Example Comparison Output

```
Evaluation Comparison: baseline → v1.0-finetuned

| Metric            | Baseline | Current | Change |
|-------------------|----------|---------|--------|
| Intent Resolution | 3.2      | 4.5     | +1.3   |
| Tool Accuracy     | 2.8      | 4.0     | +1.2   |
| Task Adherence    | 60% pass | 90% pass| +30%   |

✅ All metrics improved after fine-tuning
```

---

## Step 4.11: Decision Gate — Did Fine Tuning Improve the performance of the Agent?

This is the most important step. Based on your evaluation comparison, take one of three actions:

| Outcome | Signal | Action |
|---------|--------|--------|
| **Improved** | All eval scores increased | Keep the tuned model deployed |
| **Regressed** | Any eval score decreased | Rollback immediately |
| **Flat** | Scores unchanged (±0.2) | Collect more diverse training data and retrain |

### If Improved — Keep It

**Prompt Copilot (Agent Mode) with:**
> "Verify the tuned model is active by running: `python -m src.lightning.cli list-deployments --agent-id autonomous-agent`. Confirm the fine-tuned model is the active deployment."

### If Regressed — Rollback

**Prompt Copilot (Agent Mode) with:**
> "Rollback the autonomous-agent to the base model. Run: `python -m src.lightning.cli rollback --agent-id autonomous-agent --reason 'Eval scores regressed after fine-tuning'`. Then set `USE_TUNED_MODEL=false` on the autonomous-agent deployment in the mcp-agents namespace using kubectl set env, restart the deployment, and verify pods are running with the base model."

### If Flat — Diagnose and Retry

If scores didn't meaningfully change, the issue is usually one of:

| Root Cause | Fix |
|------------|-----|
| Too few training examples | Collect more episodes (aim for 100+ high-quality) |
| Reward threshold too low | Increase `--min-reward` from 0.7 to 0.85 |
| Low diversity in training data | Add episodes covering edge cases (scheduling, investigation) |
| Learning rate too high | Reduce `--learning-rate-multiplier` to 0.05 |
| Too few epochs | Increase from 3 to 5 if validation loss is still decreasing |

---

## Step 4.12: Store Evaluation Results with Copilot

### Store Results for Tracking

**Prompt Copilot (Agent Mode) with:**
> "Store the evaluation results for historical tracking. Run: `python -m evals.store_results --input evals/eval_results/eval_summary_*.json --agent-id autonomous-agent --version v1.0-finetuned`. Then verify the results were stored by querying Cosmos DB for evaluation records for the autonomous-agent."

Query historical evaluations:

```sql
SELECT c.agent_id, c.version, c.timestamp, c.combined_score
FROM c 
WHERE c.type = 'evaluation'
AND c.agent_id = 'autonomous-agent'
ORDER BY c.timestamp DESC
```

---

## Completion Checklist

- [ ] Evaluation dataset created with test cases
- [ ] **Baseline evaluation** run on base model (scores recorded)
- [ ] Episode capture enabled for agents
- [ ] Episodes generated through test interactions
- [ ] Episodes reviewed and labeled with rewards
- [ ] Fine-tuning dataset created with min reward threshold
- [ ] Fine-tuning job submitted and completed successfully
- [ ] Fine-tuned model deployed to agent
- [ ] **Post-training evaluation** re-run on same dataset
- [ ] Before/after scores compared — improvement validated
- [ ] Decision gate applied (keep, rollback, or retrain)
- [ ] Improvements validated in App Insights metrics

---

## Summary

You have completed the fine-tuning and evaluation exercise:

| Evaluation | Baseline | After Fine-Tuning | Target | Status |
|------------|----------|-------------------|--------|--------|
| Intent Resolution | ___ / 5 | ___ / 5 | ≥ 3 | |
| Tool Call Accuracy | ___ / 5 | ___ / 5 | ≥ 3 | |
| Task Adherence | | | not flagged | |
| **Overall Improved?** | | | yes | |

### Key Takeaways

1. **Evaluate before you train** — baseline scores are your control group; without them you can't prove improvement
2. **Same dataset, same thresholds** — changing the eval between runs invalidates the comparison
3. **Three dimensions cover the full picture** — intent (understanding), tools (action), task (outcome)
4. **Decision gates prevent regressions** — never deploy a tuned model without re-evaluating
5. **Rollback is not failure** — it's evidence-based model management
6. **Continuous evaluation in CI/CD** catches regressions before they reach production
7. **Eval scores as reward signals** — the evaluation framework feeds back into the Lightning reward loop, creating a self-improving system

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No episodes captured | Verify ENABLE_EPISODE_CAPTURE=true, check Cosmos DB connection |
| Training job failed | Check dataset format, ensure sufficient high-quality episodes (100+) |
| Model not improving | Increase epochs, reduce learning rate, add more diverse episodes |
| Eval scores regressed | Rollback immediately, review training data for quality issues |
| Task adherence flags correct responses | Review the evaluator prompt — may need calibration for your domain |
| Deployment issues | Verify model name, check agent environment variables |
| Flat scores after training | More data diversity needed; check if reward threshold is too low |

---

## Congratulations!

You've completed the **Build Your Own Agent** lab! Through 4 exercises, you've learned to:

1. ✅ **Reviewed** the lab objectives, solution architecture, and exercise structure
2. ✅ **Built** both autonomous and human-in-the-loop agents using GitHub Copilot with SpecKit
3. ✅ **Reviewed** governance, memory, observability, and security end-to-end
4. ✅ **Fine-tuned and evaluated** your agent using Agent Lightning with reinforcement learning and the Azure AI Evaluation SDK

Your agents now benefit from:
- **Enterprise governance** through APIM policies
- **Persistent memory** with Cosmos DB
- **Intelligent search** via AI Foundry and AI Search integration
- **Human oversight** through Agent 365 approvals
- **Continuous improvement** via Agent Lightning fine-tuning
- **Quality assurance** through eval-gated fine-tuning with before/after measurement
- **Full observability** with Azure Monitor integration

---

## Next Steps

- Review [AGENTS_ARCHITECTURE.md](../AGENTS_ARCHITECTURE.md) for deeper architectural understanding
- Check [AGENTS_EVALUATIONS.md](../AGENTS_EVALUATIONS.md) for advanced evaluation techniques
- See [AGENTS_IDENTITY_DESIGN.md](../AGENTS_IDENTITY_DESIGN.md) for security deep-dive
- Explore the [Optional Exercises](../LAB_MANUAL_BUILD_YOUR_OWN_AGENT.md#optional-exercises) for advanced topics
- Implement custom evaluators for domain-specific metrics
- Set up alerting for evaluation score regression
- Add more edge-case test queries to the evaluation dataset

---

## References

- [Azure AI Evaluation SDK](https://learn.microsoft.com/azure/ai-studio/how-to/develop/evaluate-sdk)
- [AGENTS_EVALUATIONS.md](../AGENTS_EVALUATIONS.md) - Project evaluation framework
- [Agent Lightning Design](../AGENTS_AGENT_LIGHTNING_DESIGN.md) - Fine-tuning architecture
