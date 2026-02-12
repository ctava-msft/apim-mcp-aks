# Lab Manual: Build Your Own Agent with Azure Agents Control Plane, GitHub Copilot and SpecKit

## Overview

This lab guides you through creating a new AI agent using Azure as the secured, managed and governed control plane and GitHub Copilot - SpecKit to realize your new agent. Using the **SpecKit methodology** for specification-driven development, you will review the project constitution to understand governance principles, then write your agent specification and deploy to Azure Kubernetes Service (AKS) as a new pod.

**Duration**: 5 hours  
**Prerequisites**: Completed environment setup and deployed the base Azure Agents Control Plane infrastructure

---

## Exercises

| Exercise | Duration | Description |
|----------|----------|-------------|
| [Exercise 1: Lab Review](exercises/exercise_01_lab_review.md) | 30 min | Review objectives, architecture, validate environment |
| [Exercise 2: Build Agents](exercises/exercise_02_build_agents.md) | 2 hr | Use GitHub Copilot and SpecKit to specify, create, test, and deploy agents |
| [Exercise 3: Review End-to-End](exercises/exercise_03_review_end_to_end.md) | 30 min | Inspect security, governance, memory, and observability. |
| [Exercise 4: Fine-Tune Agent](exercises/exercise_04_fine_tune_agent.md) | 1 hr | Use Agent Lightning to fine-tune and correct agent behavior |
| [Exercise 5: Evaluations](exercises/exercise_05_evaluations.md) | 1 hr | Use evaluation framework to measure intent resolution, tool call accuracy and task adherence |

---

## Introduction

### What is the Azure Agents Control Plane?

The Azure Agents Control Plane is a comprehensive solution accelerator that secures, governs a control plane for agents. The following solution architecture diagram depicts the Azure Agents Control Plane runtime architecture:

![Agent Control Plane ](../runtime.svg)

### Why Azure as the Enterprise Control Plane?

Traditional AI agent frameworks and architectures focus on getting something working quickly. However, enterprise deployments require:

- **Centralized Governance** - Policy enforcement, rate limiting, decision making approvals and a registry for compliance tracking
- **Security** - Threat protection, secret management, network isolation, and credential administration for all agent workloads
- **Identity-First Security** - Every agent has a Microsoft Entra ID Agent identity with Role-Based Access Control (RBAC)
- **API-First Architecture** - All agent operations flow through Azure API Management (APIM) using Model Context Protocol (MCP)
- **Multi-Cloud Capable** - Agents can execute anywhere while being governed by Azure
- **Continuous Improvement** - Built-in evaluations, learning, and fine-tuning pipelines
- **Human Oversight** - Agent 365 integration for human-in-the-loop workflows

---

## Table of Contents

1. [Exercises](#exercises)
2. [Lab Objectives](#lab-objectives)
3. [Exercise 1: Lab Review](exercises/exercise_01_lab_review.md)
4. [Exercise 2: Build Agents](exercises/exercise_02_build_agents.md)
5. [Exercise 3: Review End-to-End](exercises/exercise_03_review_end_to_end.md)
6. [Exercise 4: Fine-Tune Agent](exercises/exercise_04_fine_tune_agent.md)
7. [Exercise 5: Evaluations](exercises/exercise_05_evaluations.md)
8. [Optional Exercises](#optional-exercises)

---

## Lab Objectives

By the end of this lab, you will be able to:

### Build Your Agent
- ✅ Review the SpecKit constitution and understand governance principles for your software development lifecycle
- ✅ Write a detailed specification for your agent by following the SpecKit methodology
- ✅ Integrate with the Azure Agents Control Plane infrastructure as follows:
  - ✅ Implement an MCP-compliant agent using FastAPI
  - ✅ Containerize your agent with Docker
  - ✅ Deploy your agent to AKS as a new pod


### Enterprise Governance
- ✅ Understand how Azure API Management acts as the governance gateway
- ✅ Inspect APIM policies that enforce rate limits, quotas, and compliance with enterprise standards
- ✅ Trace requests through the control plane using distributed tracing

### Security & Identity
- ✅ Configure identity for agents running in AKS pods
- ✅ Implement least-privilege RBAC for agent resources and tooling access
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

## Getting Started

To begin the lab, proceed to **Exercise 1: Lab Review** which will be used to introduce the solution architecture and to verify your environment. 

**[Start Exercise 1: Lab Review →](exercises/exercise_01_lab_review.md)**

---

## Exercise Summaries

### Exercise 1: Lab Review (30 minutes)

Review objectives, architecture, and validate your environment is ready.

**Key Activities:**
- Review lab objectives and solution architecture
- Understand the Azure Agents Control Plane
- Run test(s) to confirm environment works

**[Full Exercise →](exercises/exercise_01_lab_review.md)**

---

### Exercise 2: Build Agents (1 hour)

Use GitHub Copilot and SpecKit to specify, create, unit test, deploy and functionally test your agent(s).

**Key Activities:**
- Review the project constitution
- Use Copilot for specification, implementation, and testing
- Build an **Autonomous Agent** that operates without human intervention
- Deploy agent(s) to AKS in a single pod

**[Full Exercise →](exercises/exercise_02_build_agents.md)**

---

### Exercise 3: Review End-to-End (30 minutes)

Inspect security, governance, memory, and observability.

**Key Activities:**
- Check APIM policies (rate limiting, MCP, OAuth, quotas)
- Check Cosmos DB for chat, plans, tasks, sessions and episodes
- Check AI Foundry/AI Search for long-term memory
- Check Fabric IQ/Storage for ontologies
- Check Log Analytics and App Insights for observability
- Check Entra ID/RBAC for identity
- Check Agent 365 for approval history
- Identify any problems with your agents

**[Full Exercise →](exercises/exercise_03_review_end_to_end.md)**

---

### Exercise 4: Fine-Tune Agent (1 hour)

Use Agent Lightning to fine-tune and correct agent behavior.

**Key Activities:**
- Enable episode capture
- Generate and review captured episodes
- Label episodes with rewards (manual or automated)
- Build fine-tuning dataset
- Submit fine-tuning job to Azure OpenAI
- Deploy tuned model
- Retest and validate improvement

**[Full Exercise →](exercises/exercise_04_fine_tune_agent.md)**

---

### Exercise 5: Evaluations (30 minutes)

Use the evaluation framework to measure agent performance and task adherence.

**Key Activities:**
- Prepare evaluation dataset with test cases
- Run intent resolution evaluation
- Run tool call accuracy evaluation
- Run task adherence evaluation
- Generate evaluation report
- Compare scores over time

**[Full Exercise →](exercises/exercise_05_evaluations.md)**

---

## Optional Exercises

For advanced topics, explore the optional exercises:

| Exercise | Duration | Description |
|----------|----------|-------------|
| [External Agents MCP Integration](exercises-optional/exercise_external_agents_mcp_integration.md) | 2 hrs | Integrate OpenAI Assistants, AWS Bedrock, Google Vertex AI with Azure governance |
| [Fabric Data Agents](exercises-optional/exercise_fabric_data_agents.md) | 2 hrs | Build agents that query Fabric Lakehouses, Warehouses, and Semantic Models |
| [Agent Red Teaming](exercises-optional/exercise_agents_redteaming.md) | 2 hrs | Red team your agents using Microsoft's PyRIT framework |
| [SRE Agent](exercises-optional/exercise_sre_agent.md) | 1 hr | Build an SRE agent for automated monitoring, diagnosis, and remediation |
| [Azure Copilot Integration](exercises-optional/exercise_azure_copilot.md) | 1 hr | Integrate your agents with Azure Copilot for natural language interaction and orchestration |
