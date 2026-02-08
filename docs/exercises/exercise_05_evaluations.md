# Exercise 5: Evaluations

**Duration:** 1 hour

## Overview

In this exercise, you will use the Azure AI Evaluation framework to measure agent quality across multiple dimensions including task adherence, tool call accuracy, and intent resolution. You will create evaluation datasets, run evaluations, and generate reports.

---

## Evaluation Framework Overview

The evaluation framework measures agent performance across three key dimensions:

| Evaluator | Weight | What it Measures |
|-----------|--------|------------------|
| **Task Adherence** | 40% | Does the agent complete the assigned task correctly? |
| **Tool Call Accuracy** | 30% | Does the agent select the right tools with correct parameters? |
| **Intent Resolution** | 30% | Does the agent correctly understand the user's intent? |

---

## Step 5.1: Prepare Evaluation Dataset

Create a JSONL file with test cases for evaluation:

```powershell
# Create evaluation data file
code evals/autonomous_agent_eval.jsonl
```

### Evaluation Data Format

Each line is a JSON object with:
- `query`: The user input
- `expected_intent`: The expected classified intent
- `expected_tools`: List of tools that should be called
- `expected_response_contains`: Key phrases expected in response
- `context`: Additional context for the evaluation

### Example Evaluation Data

```json
{"query": "Analyze customer churn for Q4 2025", "expected_intent": "analytics", "expected_tools": ["analyze_churn"], "expected_response_contains": ["churn rate", "at-risk customers"], "context": {"user_role": "analyst"}}
{"query": "Generate a sales report for the west region", "expected_intent": "reporting", "expected_tools": ["generate_report"], "expected_response_contains": ["sales", "west region", "revenue"], "context": {"user_role": "manager"}}
{"query": "What are the top 10 products by revenue?", "expected_intent": "analytics", "expected_tools": ["query_products"], "expected_response_contains": ["product", "revenue", "ranking"], "context": {"user_role": "analyst"}}
{"query": "Show me customer segments at risk", "expected_intent": "analytics", "expected_tools": ["analyze_segments", "get_risk_scores"], "expected_response_contains": ["segment", "risk", "customers"], "context": {"user_role": "analyst"}}
{"query": "Predict next month's sales forecast", "expected_intent": "forecasting", "expected_tools": ["forecast_sales"], "expected_response_contains": ["forecast", "prediction", "sales"], "context": {"user_role": "manager"}}
{"query": "Compare this quarter's performance to last year", "expected_intent": "analytics", "expected_tools": ["compare_periods"], "expected_response_contains": ["comparison", "growth", "year-over-year"], "context": {"user_role": "executive"}}
{"query": "Send me a weekly summary every Monday", "expected_intent": "scheduling", "expected_tools": ["schedule_report"], "expected_response_contains": ["scheduled", "Monday", "weekly"], "context": {"user_role": "manager"}}
{"query": "Why did customer ABC Corp churn?", "expected_intent": "investigation", "expected_tools": ["get_customer_history", "analyze_churn_factors"], "expected_response_contains": ["ABC Corp", "factors", "reason"], "context": {"user_role": "account_manager"}}
```

---

## Step 5.2: Run Intent Resolution Evaluation

Evaluate how well the agent understands user intent:

```powershell
# Run intent resolution evaluation
python -m evals.evaluate_intent_resolution `
  --agent-id autonomous-agent `
  --eval-data evals/autonomous_agent_eval.jsonl `
  --output evals/results/intent_resolution.json
```

### Review Results

```powershell
# View intent resolution results
Get-Content evals/results/intent_resolution.json | ConvertFrom-Json | Format-Table
```

Expected output:

```
Query                                    Expected    Actual      Match   Score
-----                                    --------    ------      -----   -----
Analyze customer churn for Q4 2025       analytics   analytics   True    1.0
Generate a sales report...               reporting   reporting   True    1.0
What are the top 10 products...          analytics   analytics   True    1.0
Send me a weekly summary...              scheduling  analytics   False   0.0
```

### Intent Resolution Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Accuracy | Correct / Total | > 90% |
| Precision | TP / (TP + FP) | > 85% |
| Recall | TP / (TP + FN) | > 85% |

---

## Step 5.3: Run Tool Call Accuracy Evaluation

Evaluate whether the agent selects the correct tools:

```powershell
# Run tool call accuracy evaluation
python -m evals.evaluate_tool_accuracy `
  --agent-id autonomous-agent `
  --eval-data evals/autonomous_agent_eval.jsonl `
  --output evals/results/tool_accuracy.json
```

### Review Results

```powershell
# View tool accuracy results
Get-Content evals/results/tool_accuracy.json | ConvertFrom-Json | Format-Table
```

Expected output:

```
Query                                    Expected Tools         Actual Tools           Match   Score
-----                                    --------------         ------------           -----   -----
Analyze customer churn for Q4 2025       [analyze_churn]        [analyze_churn]        True    1.0
Show me customer segments at risk        [analyze_segments,     [analyze_segments,     True    1.0
                                          get_risk_scores]       get_risk_scores]
Why did customer ABC Corp churn?         [get_customer_history, [analyze_churn]        Partial 0.5
                                          analyze_churn_factors]
```

### Tool Call Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Exact Match | All tools match exactly | > 80% |
| Partial Match | At least one correct tool | > 90% |
| Tool Order | Tools called in correct sequence | > 75% |
| Parameter Accuracy | Tool parameters are correct | > 85% |

---

## Step 5.4: Run Task Adherence Evaluation

Evaluate whether the agent completes tasks correctly:

```powershell
# Run task adherence evaluation
python -m evals.evaluate_task_adherence `
  --agent-id autonomous-agent `
  --eval-data evals/autonomous_agent_eval.jsonl `
  --output evals/results/task_adherence.json
```

### Task Adherence Scoring

The evaluator uses GPT-4 to assess response quality:

```python
# Task adherence prompt template
TASK_ADHERENCE_PROMPT = """
Given the user query and agent response, score the task adherence from 0 to 1:

User Query: {query}
Expected Response Should Contain: {expected_response_contains}
Agent Response: {response}

Scoring Criteria:
- 1.0: Response fully addresses the query with all expected elements
- 0.8: Response mostly addresses the query with minor omissions
- 0.6: Response partially addresses the query
- 0.4: Response is tangentially related but misses key points
- 0.2: Response is mostly irrelevant
- 0.0: Response is completely wrong or harmful

Score (0-1):
Reasoning:
"""
```

### Review Results

```powershell
# View task adherence results
Get-Content evals/results/task_adherence.json | ConvertFrom-Json | Format-Table
```

---

## Step 5.5: Run Combined Evaluation

Run all evaluators together with weighted scoring:

```powershell
# Run combined evaluation
python -m evals.evaluate_agent `
  --agent-id autonomous-agent `
  --eval-data evals/autonomous_agent_eval.jsonl `
  --evaluators intent_resolution,tool_accuracy,task_adherence `
  --weights 0.3,0.3,0.4 `
  --output evals/results/combined_evaluation.json
```

### Combined Score Calculation

```
Combined Score = (Intent Score × 0.3) + (Tool Score × 0.3) + (Task Score × 0.4)
```

---

## Step 5.6: Generate Evaluation Report

Create a comprehensive evaluation report:

```powershell
# Generate markdown report
python -m evals.generate_report `
  --input evals/results/combined_evaluation.json `
  --output evals/results/evaluation_report.md

# View the report
code evals/results/evaluation_report.md
```

### Report Contents

```markdown
# Agent Evaluation Report

## Summary

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Intent Resolution | 92% | 90% | ✅ Pass |
| Tool Call Accuracy | 85% | 80% | ✅ Pass |
| Task Adherence | 88% | 85% | ✅ Pass |
| **Combined Score** | **88%** | **85%** | **✅ Pass** |

## Detailed Results

### Intent Resolution
- Total Test Cases: 8
- Correct: 7
- Incorrect: 1
- Accuracy: 87.5%

### By Intent Category
| Intent | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| analytics | 4 | 4 | 100% |
| reporting | 1 | 1 | 100% |
| forecasting | 1 | 1 | 100% |
| scheduling | 0 | 1 | 0% |
| investigation | 1 | 1 | 100% |

### Failure Analysis

#### Case: "Send me a weekly summary every Monday"
- Expected Intent: scheduling
- Actual Intent: analytics
- Recommendation: Add more scheduling examples to training data

## Recommendations

1. Add more training data for scheduling intents
2. Review tool selection for multi-step tasks
3. Consider fine-tuning on investigation queries
```

---

## Step 5.7: Compare Evaluations Over Time

Track evaluation scores across versions:

```powershell
# Run evaluation with version tag
python -m evals.evaluate_agent `
  --agent-id autonomous-agent `
  --eval-data evals/autonomous_agent_eval.jsonl `
  --version v1.0-finetuned `
  --output evals/results/eval_v1.0-finetuned.json

# Compare versions
python -m evals.compare_evaluations `
  --baseline evals/results/eval_v1.0-baseline.json `
  --current evals/results/eval_v1.0-finetuned.json
```

### Version Comparison Output

```
Evaluation Comparison: v1.0-baseline → v1.0-finetuned

| Metric            | Baseline | Current | Change |
|-------------------|----------|---------|--------|
| Intent Resolution | 78%      | 92%     | +14%   |
| Tool Accuracy     | 72%      | 85%     | +13%   |
| Task Adherence    | 75%      | 88%     | +13%   |
| Combined Score    | 75%      | 88%     | +13%   |

✅ All metrics improved after fine-tuning
```

---

## Step 5.8: Store Evaluation Results

Save evaluation results to Cosmos DB for tracking:

```powershell
# Store results in Cosmos DB
python -m evals.store_results `
  --input evals/results/combined_evaluation.json `
  --agent-id autonomous-agent `
  --version v1.0-finetuned
```

Query historical evaluations:

```sql
-- Query evaluation history in Cosmos DB
SELECT c.agent_id, c.version, c.timestamp, c.combined_score
FROM c 
WHERE c.type = 'evaluation'
AND c.agent_id = 'autonomous-agent'
ORDER BY c.timestamp DESC
```

---

## Step 5.9: Set Up Continuous Evaluation

Configure automated evaluations in CI/CD:

```yaml
# .github/workflows/evaluate-agent.yml
name: Evaluate Agent

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Evaluations
        run: |
          python -m evals.evaluate_agent \
            --agent-id autonomous-agent \
            --eval-data evals/autonomous_agent_eval.jsonl \
            --output evals/results/daily_eval.json
      
      - name: Check Thresholds
        run: |
          python -m evals.check_thresholds \
            --input evals/results/daily_eval.json \
            --min-combined 0.85
      
      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-results
          path: evals/results/
```

---

## Verification Checklist

- [ ] Evaluation dataset created with test cases
- [ ] Intent resolution evaluation completed
- [ ] Tool call accuracy evaluation completed
- [ ] Task adherence evaluation completed
- [ ] Combined evaluation with weighted scoring
- [ ] Evaluation report generated
- [ ] Results compared to baseline
- [ ] Evaluation results stored for tracking
- [ ] All metrics meet target thresholds

---

## Summary

You have completed the evaluation framework exercise:

| Evaluation | Your Score | Target | Status |
|------------|------------|--------|--------|
| Intent Resolution | | 90% | |
| Tool Call Accuracy | | 80% | |
| Task Adherence | | 85% | |
| Combined Score | | 85% | |

### Key Takeaways

1. **Structured evaluation** ensures consistent measurement of agent quality
2. **Multiple dimensions** (intent, tools, task) provide comprehensive coverage
3. **Baseline comparisons** validate fine-tuning improvements
4. **Continuous evaluation** in CI/CD maintains quality over time
5. **Historical tracking** enables trend analysis and regression detection

---

## Lab Complete

Congratulations! You have completed all 5 exercises:

1. ✅ **Lab Review** - Environment validated and ready
2. ✅ **Build Agents** - Autonomous and approval agents deployed
3. ✅ **Review End-to-End** - Governance and observability inspected
4. ✅ **Fine-Tune Agent** - Model improved with Agent Lightning
5. ✅ **Evaluations** - Agent quality measured and validated

---

## Next Steps

- Add more test cases to evaluation dataset
- Implement custom evaluators for domain-specific metrics
- Set up alerting for evaluation score regression
- Integrate evaluations with fine-tuning feedback loop
- Explore Advanced Extensions (see Optional Extensions section)

---

## References

- [Azure AI Evaluation SDK](https://learn.microsoft.com/azure/ai-studio/how-to/develop/evaluate-sdk)
- [AGENTS_EVALUATIONS.md](../AGENTS_EVALUATIONS.md) - Project evaluation framework
- [Agent Lightning Design](../AGENTS_AGENT_LIGHTNING_DESIGN.md) - Fine-tuning architecture
