# Exercise 2: Build Agents using GitHub Copilot and SpecKit

**Duration:** 1 hour

## Overview

In this exercise, you will use GitHub Copilot and the SpecKit to specify, create, unit test, and deploy an **Autonomous Agent** that operates independently without human intervention.

---

## Part A: Review the Project Constitution

Before creating agents, you should understand the SDLC framework defined in the constitution.

### Step A.1: Open the Constitution

In VS Code Explorer, navigate to and open `.speckit/constitution.md`.

### Step A.2: Identify Key Elements

As you review, document at a high-level the key take-home points:

| Element | Your Findings |
|---------|---------------|
| Project Overview | |
| Core Principles | |
| Development Standards | |
| Technical Architecture | |
| Success Criteria | |
| Constraints & Assumptions | |
| Governance | |

### Step A.3: Review Example Specifications

In VS Code Explorer, navigate to and open the following files:

- `.speckit/specifications/healthcare_digital_quality_agent.spec.md`
- `.speckit/specifications/customer_churn_agent.spec.md`
- `.speckit/specifications/devops_cicd_pipeline_agent.spec.md`
- `.speckit/specifications/user_security_agent.spec.mdd`

The first document: - `.speckit/specifications/healthcare_digital_quality_agent.spec.md` is what was used to build the existing next best action agent. 

what did you make of the specifications? What aspects of specification will you re-use for your use-case?
---

## Part B: Build Autonomous Agent

### Step B.1: Create Agent Specification with GitHub Copilot.

Brainstorm what you want your agent to be / done. 
For the purposes of this exercise, there is no user interface for the agent. 
The agent is a python script that can accept an english based description of a task. 
It can then determine the intent of the task, reason and plan out steps to be done and then execute on them. Keep in mind, this agent of yours needs to be given boundaries an identity and RBAC permissions to resources / tools for it to perform its job. 

You'll need to invent or select an existing use case:

- `.speckit/specifications/customer_churn_agent.spec.md`
- `.speckit/specifications/devops_cicd_pipeline_agent.spec.md`
- `.speckit/specifications/user_security_agent.spec.mdd`

Now please move onto the next step of using GitHub Copilot and Claude Opus 4.6 to help you review, write, adjust your specification and then build it. 

```powershell

In VS Code Explorer, navigate to and open `.speckit/specifications`.

**Prompt Copilot with:**
Start from scratch:
> "Create an specification for an autonomous <responsbility_of_your_agent> agent that can <do x,y,z> without human intervention. Follow the SpecKit template format. Utilize the .speckit/specifications as an example.

The specification should include:
- Overview (Spec ID, Version, Domain, Governance Model: Autonomous)
- Business Framing
- MCP Tool Catalog
- Workflow Specification
- Success Metrics
- Security Requirements"

If you want to alter an existing specification:
**Prompt Copilot with:**
> "Make a copy of the .speckit/specifications/<>.spec.md. Make the following changes to it: x,y,z."

### Step B.2: Implement Autonomous Agent, Unit Test(s) and Functional Test(s) with Copilot

For reference, review the existing `src/next_best_action_agent.py`, `src/next_best_action_agent_unit.py`, `src/next_best_action_agent_functional.py` — this is the implementation built from the `healthcare_digital_quality_agent.spec.md` specification. This reference implementation should be used as a pattern for your own agent.

To create the agent,
**Prompt Copilot with:**
> "Implement an MCP-compliant FastAPI agent based on the <autonomous_agent.spec.md> specification. Utilize `src/next_best_action_agent.py` as a reference implementation. Build the implementation similar to the reference implementation but in its own new file `src/autonomous_agent.py. Be sure to include health endpoint, SSE endpoint, and message endpoint with tools/list and tools/call handlers. Also create pytest unit tests in `tests/test_autonomous_agent_unit.py` covering the health endpoint, MCP initialize, tools/list, and tools/call methods. Create functional tests in `tests/test_autonomous_agent_functional.py` covering the health endpoint, MCP initialize, tools/list, and tools/call methods. Make a new DockerFile specific to this new agent."

### Step B.3: Run Unit Tests with Copilot

**Prompt Copilot (Agent Mode) with:**
> "Check if a .venv virtual environment exists in the project root. If it doesn't, create one. Activate it, install the dependencies from src/requirements.txt, and run the unit tests in tests/test_autonomous_agent_unit.py with verbose output."

### Step B.4: Deploy Autonomous Agent with Copilot

**Prompt Copilot (Agent Mode) with:**
> "Build the new Docker image for this agent in the src/ directory for my autonomous agent, tag it and push it to the Azure Container Registry using the CONTAINER_REGISTRY environment variable, then deploy it to AKS using k8s/autonomous-agent-deployment.yaml and verify the pods are running in the mcp-agents namespace."

### Step B.5: Run Functional Tests with Copilot

**Prompt Copilot (Agent Mode) with:**
> "Set up a kubectl port-forward from the autonomous-agent service in the mcp-agents namespace on port 8080:80. Then check if a .venv virtual environment exists in the project root — if it doesn't, create one. Activate it, install the dependencies from src/requirements.txt, and run the functional tests in tests/test_autonomous_agent_functional.py with verbose output."

---

## Verification Checklist

### Autonomous Agent
- [ ] Specification created/updated
- [ ] Agent created with MCP endpoints
- [ ] Unit tests passing
- [ ] Docker image built and pushed to ACR
- [ ] Agent deployed to AKS and running
- [ ] Functional tests passing

---

## Summary

Congratulations!, You have built an autonomous agent using GitHub Copilot and SpecKit.

The agent implements MC protocol and is integrated with the Azure Agents Control Plane for security, governance, adn observability.

---


**Next:** [Exercise 3: Review Agents Control Plane](exercise_03_review_agents_control_plane.md)
