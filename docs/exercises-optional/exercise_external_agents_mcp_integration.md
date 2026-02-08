# Optional Exercise: External Agents MCP Integration

**Objective:** Integrate external AI agents (OpenAI Assistants, AWS Bedrock Agents, Google Vertex AI Agents) with the Azure Agents Control Plane via MCP.

**Duration:** 2-3 hours

---

## Overview

This exercise demonstrates how to integrate third-party AI agents with Azure as the governance control plane. External agents benefit from:

- **Centralized API Management** through Azure APIM
- **Unified Identity** via Entra ID federation
- **Consistent Observability** with Azure Monitor
- **Policy Enforcement** for compliance and rate limiting

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Azure Agents Control Plane                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   APIM      │  │  Entra ID   │  │   Monitor   │  │    Cosmos DB        │ │
│  │  (Gateway)  │  │  (Identity) │  │  (Observ)   │  │    (Memory)         │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         └─────────────────┼─────────────────┼────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   OpenAI     │   │ AWS Bedrock  │   │ Google       │
│  Assistants  │   │   Agents     │   │ Vertex AI    │
└──────────────┘   └──────────────┘   └──────────────┘
```

---

## Prerequisites

- Azure Agents Control Plane deployed
- External agent platform access (OpenAI, AWS, or Google Cloud)
- Understanding of MCP protocol

---

## Part A: OpenAI Assistants Integration

### Step A.1: Create MCP Adapter for OpenAI Assistants

Create an MCP façade that wraps OpenAI Assistants API:

```python
# src/adapters/openai_assistant_adapter.py

import os
from fastapi import FastAPI, Request
from openai import OpenAI
from typing import Dict, Any
import json

app = FastAPI(title="OpenAI Assistant MCP Adapter")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ASSISTANT_ID = os.environ["OPENAI_ASSISTANT_ID"]

# MCP Tool definitions exposed by this adapter
MCP_TOOLS = [
    {
        "name": "openai_assistant_chat",
        "description": "Send a message to OpenAI Assistant and get response",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "User message"},
                "thread_id": {"type": "string", "description": "Optional thread ID"}
            },
            "required": ["message"]
        }
    }
]

@app.get("/health")
async def health():
    return {"status": "healthy", "adapter": "openai-assistant"}

@app.post("/message")
async def mcp_message(request: Request) -> Dict[str, Any]:
    body = await request.json()
    method = body.get("method")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "openai-assistant-adapter", "version": "1.0.0"}
            }
        }
    
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": body.get("id"), "result": {"tools": MCP_TOOLS}}
    
    if method == "tools/call":
        tool_name = body["params"]["name"]
        args = body["params"]["arguments"]
        
        if tool_name == "openai_assistant_chat":
            result = await call_openai_assistant(args["message"], args.get("thread_id"))
            return {"jsonrpc": "2.0", "id": body.get("id"), "result": result}
    
    return {"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32601, "message": "Method not found"}}

async def call_openai_assistant(message: str, thread_id: str = None) -> Dict[str, Any]:
    """Call OpenAI Assistant and return response."""
    # Create or use existing thread
    if not thread_id:
        thread = client.beta.threads.create()
        thread_id = thread.id
    
    # Add message to thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message
    )
    
    # Run assistant
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )
    
    # Get response
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    response = messages.data[0].content[0].text.value
    
    return {
        "response": response,
        "thread_id": thread_id,
        "run_id": run.id
    }
```

### Step A.2: Configure APIM Policy for OpenAI Adapter

```xml
<policies>
    <inbound>
        <!-- Validate Azure AD token -->
        <validate-jwt header-name="Authorization" failed-validation-httpcode="401">
            <openid-config url="https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration" />
        </validate-jwt>
        
        <!-- Rate limit external agent calls -->
        <rate-limit-by-key calls="50" renewal-period="60" 
                          counter-key="@(context.Request.Headers.GetValueOrDefault('X-Agent-ID','unknown'))" />
        
        <!-- Log to Azure Monitor -->
        <log-to-eventhub logger-id="azure-monitor-logger">
            @{
                return new JObject(
                    new JProperty("eventType", "ExternalAgentCall"),
                    new JProperty("adapter", "openai-assistant"),
                    new JProperty("timestamp", DateTime.UtcNow.ToString("o"))
                ).ToString();
            }
        </log-to-eventhub>
    </inbound>
    <backend>
        <forward-request />
    </backend>
</policies>
```

---

## Part B: AWS Bedrock Agents Integration

### Step B.1: Create MCP Adapter for Bedrock Agents

```python
# src/adapters/bedrock_agent_adapter.py

import boto3
import os
from fastapi import FastAPI, Request
from typing import Dict, Any

app = FastAPI(title="AWS Bedrock Agent MCP Adapter")

bedrock_agent = boto3.client(
    "bedrock-agent-runtime",
    region_name=os.environ.get("AWS_REGION", "us-east-1")
)

AGENT_ID = os.environ["BEDROCK_AGENT_ID"]
AGENT_ALIAS_ID = os.environ["BEDROCK_AGENT_ALIAS_ID"]

MCP_TOOLS = [
    {
        "name": "bedrock_agent_invoke",
        "description": "Invoke AWS Bedrock Agent",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Prompt for the agent"},
                "session_id": {"type": "string", "description": "Session ID for context"}
            },
            "required": ["prompt"]
        }
    }
]

@app.post("/message")
async def mcp_message(request: Request) -> Dict[str, Any]:
    body = await request.json()
    method = body.get("method")
    
    if method == "tools/call":
        args = body["params"]["arguments"]
        result = invoke_bedrock_agent(args["prompt"], args.get("session_id", "default"))
        return {"jsonrpc": "2.0", "id": body.get("id"), "result": result}
    
    # ... handle other methods

def invoke_bedrock_agent(prompt: str, session_id: str) -> Dict[str, Any]:
    """Invoke AWS Bedrock Agent."""
    response = bedrock_agent.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=prompt
    )
    
    completion = ""
    for event in response["completion"]:
        if "chunk" in event:
            completion += event["chunk"]["bytes"].decode()
    
    return {
        "response": completion,
        "session_id": session_id
    }
```

### Step B.2: Configure OIDC Federation for AWS

```bash
# Create OIDC identity provider in Azure
az ad app federated-credential create \
  --id <app-id> \
  --parameters '{
    "name": "aws-bedrock-federation",
    "issuer": "https://sts.amazonaws.com",
    "subject": "arn:aws:iam::123456789012:role/bedrock-agent-role",
    "audiences": ["api://AzureADTokenExchange"]
  }'
```

---

## Part C: Google Vertex AI Agents Integration

### Step C.1: Create MCP Adapter for Vertex AI

```python
# src/adapters/vertex_ai_adapter.py

from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
import os
from fastapi import FastAPI, Request
from typing import Dict, Any

app = FastAPI(title="Vertex AI Agent MCP Adapter")

aiplatform.init(
    project=os.environ["GCP_PROJECT_ID"],
    location=os.environ.get("GCP_REGION", "us-central1")
)

model = GenerativeModel("gemini-1.5-pro")

@app.post("/message")
async def mcp_message(request: Request) -> Dict[str, Any]:
    body = await request.json()
    method = body.get("method")
    
    if method == "tools/call":
        args = body["params"]["arguments"]
        result = await invoke_vertex_agent(args["prompt"])
        return {"jsonrpc": "2.0", "id": body.get("id"), "result": result}

async def invoke_vertex_agent(prompt: str) -> Dict[str, Any]:
    """Invoke Vertex AI model."""
    response = await model.generate_content_async(prompt)
    return {"response": response.text}
```

---

## Verification

### Test External Agent Integration

```powershell
# Port forward to adapter
kubectl port-forward -n mcp-agents svc/openai-assistant-adapter 8080:80

# Test via APIM
$token = az account get-access-token --resource api://<app-id> --query accessToken -o tsv

$body = @{
    jsonrpc = "2.0"
    id = 1
    method = "tools/call"
    params = @{
        name = "openai_assistant_chat"
        arguments = @{
            message = "What are the top 5 customers by revenue?"
        }
    }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "https://<apim>.azure-api.net/external-agents/openai/message" `
    -Method Post `
    -Headers @{ Authorization = "Bearer $token" } `
    -Body $body `
    -ContentType "application/json"
```

### Verify Telemetry in Azure Monitor

```kusto
// External agent calls
customEvents
| where name == "ExternalAgentCall"
| summarize count() by adapter, bin(timestamp, 1h)
| render timechart
```

---

## Summary

You've integrated external AI agents with the Azure Agents Control Plane:

| Provider | Adapter | Governance | Identity |
|----------|---------|------------|----------|
| OpenAI Assistants | MCP Façade | APIM | Entra ID |
| AWS Bedrock | MCP Façade | APIM | OIDC Federation |
| Google Vertex AI | MCP Façade | APIM | Workload Identity |

**Key Insight:** All external agents now benefit from Azure's centralized governance, identity, and observability.

---

## Related Resources

- [OpenAI Assistants API](https://platform.openai.com/docs/assistants/overview)
- [AWS Bedrock Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- [Google Vertex AI Agents](https://cloud.google.com/vertex-ai/docs/generative-ai/agents/overview)
- [Azure APIM Documentation](https://docs.microsoft.com/azure/api-management/)
