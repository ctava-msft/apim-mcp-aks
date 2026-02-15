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
> "Set up a kubectl port-forward from the autonomous-agent service in the mcp-agents namespace on port 8080:80. Then read the evaluation dataset at `evals/autonomous_agent_eval.jsonl` to get the list of test queries. For each query in the dataset, send a JSON-RPC request to `http://localhost:8080/message` with method `tools/call`, tool name `analyze_query`, and the query as the argument. Display the response for each query and pause 2 seconds between requests. This will generate episodes that get captured for training data."

---

## Step 4.5: Review Captured Episodes with Copilot

Query Cosmos DB for captured episodes.

**Prompt Copilot (Agent Mode) with:**
> "List the most recent 10 captured episodes for the autonomous-agent. First try using the Lightning CLI: `python -m src.lightning.cli list-episodes --agent-id autonomous-agent --limit 10`. If the CLI is not available, use the Azure CLI to query Cosmos DB: `az cosmosdb sql query --account-name $env:COSMOSDB_ACCOUNT --database-name agents --container-name rl_episodes --query \"SELECT * FROM c WHERE c.agent_id = 'autonomous-agent' ORDER BY c._ts DESC OFFSET 0 LIMIT 10\"`. Display the results and summarize how many episodes were captured."

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

### Manual Labeling

**Prompt Copilot (Agent Mode) with:**
> "List all unlabeled episodes for the autonomous-agent using `python -m src.lightning.cli list-episodes --agent-id autonomous-agent --unlabeled-only`. Review the results and for each episode, determine a reward score based on these criteria: 0.9-1.0 for correct tool + complete response, 0.7-0.89 for correct tool + mostly complete, 0.5-0.69 for partially correct, 0.3-0.49 for wrong approach, 0.0-0.29 for failed. Label each episode using `python -m src.lightning.cli label-episode --episode-id <id> --reward <score> --reason '<explanation>'`."

### Automated Labeling with Evaluators

Use the task adherence evaluator for batch labeling — this **wires evaluation scores directly into the reward signal**.

**Prompt Copilot (Agent Mode) with:**
> "Run automated batch labeling on all unlabeled episodes for the autonomous-agent using the task adherence evaluator. Execute: `python -m src.lightning.cli label-batch --agent-id autonomous-agent --auto-label --evaluator task_adherence --min-score 0.7`. Then verify the labeling results by listing labeled episodes: `python -m src.lightning.cli list-episodes --agent-id autonomous-agent --labeled-only --limit 20`. Display the reward distribution summary."

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
> "Start a fine-tuning job on Azure OpenAI for the autonomous-agent. Run: `python -m src.lightning.cli start-training --agent-id autonomous-agent --dataset-name autonomous-agent-v1 --base-model gpt-4o-mini --training-epochs 3 --learning-rate-multiplier 0.1`. Note the returned job ID. Then monitor the training progress by running `python -m src.lightning.cli get-training-status --job-id <returned_job_id>` and display the status, progress, training loss, and validation loss."

### Monitor Training Progress

**Prompt Copilot (Agent Mode) with:**
> "Check the training status for the fine-tuning job by running `python -m src.lightning.cli get-training-status --job-id <job_id>`. Report the current status, progress percentage, and training/validation loss."

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

**Prompt Copilot (Agent Mode) with:**
> "Promote and deploy the fine-tuned model for the autonomous-agent. First, promote it using: `python -m src.lightning.cli promote-model --agent-id autonomous-agent --model-name ft:gpt-4o-mini:autonomous-agent:<model_suffix> --mark-active`. Then update the AKS deployment to use the fine-tuned model by setting `USE_TUNED_MODEL=true` and `TUNED_MODEL_NAME=ft:gpt-4o-mini:autonomous-agent:<model_suffix>` on the autonomous-agent deployment in the mcp-agents namespace. Restart the deployment and wait for the rollout to complete. Verify the pods are running with the new environment variables."

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

## Step 4.11: Decision Gate — Did It Actually Improve?

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

## Step 4.12: Validate in App Insights with Copilot

Query Application Insights to validate improvements at the infrastructure level.

**Prompt Copilot (Agent Mode) with:**
> "Query Application Insights to compare error rates before and after fine-tuning. Use the Azure CLI or the Application Insights REST API to run the following KQL query against the App Insights resource associated with this deployment:
> ```
> requests | where timestamp > ago(24h) | extend model_version = tostring(customDimensions['model_version']) | summarize TotalRequests = count(), ErrorRate = round(countif(success == false) * 100.0 / count(), 2), AvgDuration = round(avg(duration), 0) by model_version, bin(timestamp, 1h) | order by timestamp desc
> ```
> Display the results showing error rates by model version."

---

## Step 4.13: Store Evaluation Results and Set Up Continuous Evaluation with Copilot

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

### Set Up Continuous Evaluation in CI/CD

Automate evaluations so that every deployment is gated on eval scores.

**Prompt Copilot (Agent Mode) with:**
> "Create a GitHub Actions workflow file at `.github/workflows/evaluate-agent.yml` that runs the agent evaluation on every push to main and on a daily schedule (midnight UTC). The workflow should: (1) check out the repo, (2) run `python -m evals.evaluate_next_best_action --data evals/next_best_action_eval_data.jsonl --out evals/eval_results --direct --strict`, (3) check if the eval_summary JSON has `all_passed: true` — if not, fail the build with exit code 1, and (4) upload the eval_results directory as a GitHub Actions artifact. Use `actions/checkout@v4` and `actions/upload-artifact@v4`."

---

## Verification Checklist

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

- Explore the [Optional Exercises](../LAB_MANUAL_BUILD_YOUR_OWN_AGENT.md#optional-exercises) for advanced topics
- Review [AGENTS_ARCHITECTURE.md](../AGENTS_ARCHITECTURE.md) for deeper architectural understanding
- Check [AGENTS_EVALUATIONS.md](../AGENTS_EVALUATIONS.md) for advanced evaluation techniques
- See [AGENTS_IDENTITY_DESIGN.md](../AGENTS_IDENTITY_DESIGN.md) for security deep-dive
- Implement custom evaluators for domain-specific metrics
- Set up alerting for evaluation score regression
- Add more edge-case test queries to the evaluation dataset

---

## References

- [Azure AI Evaluation SDK](https://learn.microsoft.com/azure/ai-studio/how-to/develop/evaluate-sdk)
- [AGENTS_EVALUATIONS.md](../AGENTS_EVALUATIONS.md) - Project evaluation framework
- [Agent Lightning Design](../AGENTS_AGENT_LIGHTNING_DESIGN.md) - Fine-tuning architecture
