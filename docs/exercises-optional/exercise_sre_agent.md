# Optional Exercise: Build an SRE Agent

**Objective:** Build a Site Reliability Engineering (SRE) agent that monitors, diagnoses, and remediates infrastructure issues automatically.

**Duration:** 3-4 hours

---

## Overview

An SRE Agent automates operational tasks that traditionally require human intervention:

| Capability | Description |
|------------|-------------|
| **Monitoring** | Query Azure Monitor, Log Analytics, App Insights |
| **Alerting** | Receive and triage alerts automatically |
| **Diagnosis** | Analyze logs, metrics, and traces to identify root cause |
| **Remediation** | Execute runbooks to fix known issues |
| **Escalation** | Request human approval for high-risk actions |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SRE Agent                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Monitor    │  │  Diagnose   │  │  Remediate  │  │    Escalate         │ │
│  │  Tools      │  │  Tools      │  │  Tools      │  │    (Agent 365)      │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Azure Monitor│   │ AKS Cluster  │   │ Azure        │
│ Log Analytics│   │ (kubectl)    │   │ Resource Mgr │
└──────────────┘   └──────────────┘   └──────────────┘
```

---

## Prerequisites

- Azure Agents Control Plane deployed
- AKS cluster with workload identity
- Log Analytics workspace with agent logs
- Understanding of Kubernetes operations

---

## Part A: Create SRE Agent Specification

### Step A.1: Write Specification

Create `.speckit/specifications/sre_agent.spec.md`:

```markdown
# SRE Agent Specification

## Overview

| Property | Value |
|----------|-------|
| **Spec ID** | `SRE-001` |
| **Version** | `1.0.0` |
| **Domain** | Site Reliability Engineering |
| **Agent Type** | Single Agent |
| **Governance Model** | Semi-Autonomous |

## Business Framing

The SRE Agent reduces Mean Time To Resolution (MTTR) by automating 
incident detection, diagnosis, and remediation. Human approval is 
required for high-risk operations like scaling or restarting services.

## MCP Tool Catalog

| Tool Name | Description | Risk Level |
|-----------|-------------|------------|
| `query_logs` | Query Log Analytics | Low |
| `query_metrics` | Query Azure Monitor | Low |
| `get_pod_status` | Get K8s pod status | Low |
| `describe_pod` | Describe K8s pod | Low |
| `get_pod_logs` | Get pod logs | Low |
| `restart_pod` | Restart a pod | High |
| `scale_deployment` | Scale deployment | High |
| `apply_manifest` | Apply K8s manifest | High |
| `create_incident` | Create incident ticket | Low |
| `escalate_to_human` | Request human approval | Low |

## Governance

High-risk tools (`restart_pod`, `scale_deployment`, `apply_manifest`) 
require human approval via Agent 365 before execution.
```

---

## Part B: Implement SRE Agent

### Step B.1: Create Agent Implementation

```python
# src/agents/sre_agent/agent.py

import os
import json
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, MetricsQueryClient
from kubernetes import client, config
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SRE Agent",
    description="Site Reliability Engineering Agent for automated operations",
    version="1.0.0"
)

# Azure clients
credential = DefaultAzureCredential()
logs_client = LogsQueryClient(credential)
metrics_client = MetricsQueryClient(credential)

# K8s client
try:
    config.load_incluster_config()
except:
    config.load_kube_config()
k8s_core = client.CoreV1Api()
k8s_apps = client.AppsV1Api()

# Environment
LOG_ANALYTICS_WORKSPACE_ID = os.environ.get("LOG_ANALYTICS_WORKSPACE_ID")
APPROVAL_LOGIC_APP_URL = os.environ.get("APPROVAL_LOGIC_APP_URL")
NAMESPACE = os.environ.get("K8S_NAMESPACE", "mcp-agents")

# Tool definitions
MCP_TOOLS = [
    {
        "name": "query_logs",
        "description": "Query Log Analytics for application logs",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "KQL query"},
                "timespan": {"type": "string", "description": "Time range (e.g., PT1H, P1D)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "query_metrics",
        "description": "Query Azure Monitor metrics",
        "inputSchema": {
            "type": "object",
            "properties": {
                "resource_id": {"type": "string", "description": "Azure resource ID"},
                "metric_names": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["resource_id", "metric_names"]
        }
    },
    {
        "name": "get_pod_status",
        "description": "Get status of pods in namespace",
        "inputSchema": {
            "type": "object",
            "properties": {
                "namespace": {"type": "string", "description": "K8s namespace"},
                "label_selector": {"type": "string", "description": "Label selector"}
            }
        }
    },
    {
        "name": "get_pod_logs",
        "description": "Get logs from a specific pod",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pod_name": {"type": "string"},
                "namespace": {"type": "string"},
                "tail_lines": {"type": "integer", "default": 100}
            },
            "required": ["pod_name"]
        }
    },
    {
        "name": "restart_pod",
        "description": "Restart a pod (requires approval)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pod_name": {"type": "string"},
                "namespace": {"type": "string"},
                "reason": {"type": "string"}
            },
            "required": ["pod_name", "reason"]
        }
    },
    {
        "name": "scale_deployment",
        "description": "Scale a deployment (requires approval)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "deployment_name": {"type": "string"},
                "namespace": {"type": "string"},
                "replicas": {"type": "integer"},
                "reason": {"type": "string"}
            },
            "required": ["deployment_name", "replicas", "reason"]
        }
    },
    {
        "name": "diagnose_issue",
        "description": "Run automated diagnosis for an issue",
        "inputSchema": {
            "type": "object",
            "properties": {
                "issue_type": {"type": "string", "enum": ["high_latency", "pod_crash", "oom", "connection_error"]},
                "resource_name": {"type": "string"}
            },
            "required": ["issue_type", "resource_name"]
        }
    }
]

HIGH_RISK_TOOLS = ["restart_pod", "scale_deployment", "apply_manifest"]

# Tool implementations
async def query_logs(query: str, timespan: str = "PT1H") -> Dict[str, Any]:
    """Query Log Analytics."""
    response = logs_client.query_workspace(
        workspace_id=LOG_ANALYTICS_WORKSPACE_ID,
        query=query,
        timespan=timedelta(hours=1) if timespan == "PT1H" else timedelta(days=1)
    )
    
    results = []
    for table in response.tables:
        for row in table.rows:
            results.append(dict(zip([col.name for col in table.columns], row)))
    
    return {"success": True, "results": results, "count": len(results)}

async def get_pod_status(namespace: str = None, label_selector: str = None) -> Dict[str, Any]:
    """Get pod status."""
    ns = namespace or NAMESPACE
    pods = k8s_core.list_namespaced_pod(
        namespace=ns,
        label_selector=label_selector
    )
    
    pod_statuses = []
    for pod in pods.items:
        pod_statuses.append({
            "name": pod.metadata.name,
            "status": pod.status.phase,
            "ready": all(c.ready for c in (pod.status.container_statuses or [])),
            "restarts": sum(c.restart_count for c in (pod.status.container_statuses or [])),
            "age": (datetime.utcnow() - pod.metadata.creation_timestamp.replace(tzinfo=None)).total_seconds()
        })
    
    return {"success": True, "pods": pod_statuses}

async def get_pod_logs(pod_name: str, namespace: str = None, tail_lines: int = 100) -> Dict[str, Any]:
    """Get pod logs."""
    ns = namespace or NAMESPACE
    logs = k8s_core.read_namespaced_pod_log(
        name=pod_name,
        namespace=ns,
        tail_lines=tail_lines
    )
    
    return {"success": True, "pod": pod_name, "logs": logs}

async def request_approval(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Request human approval for high-risk action."""
    payload = {
        "agent_id": "sre-agent",
        "action": tool_name,
        "context": args,
        "risk_level": "high",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(APPROVAL_LOGIC_APP_URL, json=payload, timeout=30)
    
    return {"approval_requested": True, "request_id": response.json().get("request_id")}

async def restart_pod(pod_name: str, namespace: str = None, reason: str = None, approved: bool = False) -> Dict[str, Any]:
    """Restart a pod (requires approval)."""
    if not approved:
        return await request_approval("restart_pod", {"pod_name": pod_name, "namespace": namespace, "reason": reason})
    
    ns = namespace or NAMESPACE
    k8s_core.delete_namespaced_pod(name=pod_name, namespace=ns)
    
    return {"success": True, "action": "pod_restarted", "pod": pod_name, "reason": reason}

async def scale_deployment(deployment_name: str, replicas: int, namespace: str = None, reason: str = None, approved: bool = False) -> Dict[str, Any]:
    """Scale a deployment (requires approval)."""
    if not approved:
        return await request_approval("scale_deployment", {"deployment_name": deployment_name, "replicas": replicas, "reason": reason})
    
    ns = namespace or NAMESPACE
    k8s_apps.patch_namespaced_deployment_scale(
        name=deployment_name,
        namespace=ns,
        body={"spec": {"replicas": replicas}}
    )
    
    return {"success": True, "action": "deployment_scaled", "deployment": deployment_name, "replicas": replicas}

async def diagnose_issue(issue_type: str, resource_name: str) -> Dict[str, Any]:
    """Run automated diagnosis."""
    diagnosis = {"issue_type": issue_type, "resource": resource_name, "findings": [], "recommendations": []}
    
    if issue_type == "high_latency":
        # Check pod metrics
        pods = await get_pod_status(label_selector=f"app={resource_name}")
        
        # Query latency logs
        logs_result = await query_logs(
            f"requests | where cloud_RoleName == '{resource_name}' | summarize percentile(duration, 95) by bin(timestamp, 5m)"
        )
        
        diagnosis["findings"].append(f"Found {len(pods['pods'])} pods for {resource_name}")
        diagnosis["recommendations"].append("Consider scaling if P95 latency exceeds SLO")
    
    elif issue_type == "pod_crash":
        # Get pod events
        pods = await get_pod_status(label_selector=f"app={resource_name}")
        crashing = [p for p in pods["pods"] if p["restarts"] > 3]
        
        if crashing:
            diagnosis["findings"].append(f"Found {len(crashing)} pods with high restart count")
            logs = await get_pod_logs(crashing[0]["name"])
            diagnosis["findings"].append(f"Recent logs: {logs['logs'][-500:]}")
            diagnosis["recommendations"].append("Check for OOM or application errors")
    
    elif issue_type == "oom":
        logs_result = await query_logs(
            f"ContainerLogV2 | where ContainerName contains '{resource_name}' | where LogMessage contains 'OOMKilled'"
        )
        diagnosis["findings"].append(f"Found {logs_result['count']} OOM events")
        diagnosis["recommendations"].append("Increase memory limits in deployment")
    
    return diagnosis

# Tool dispatcher
TOOL_HANDLERS = {
    "query_logs": query_logs,
    "get_pod_status": get_pod_status,
    "get_pod_logs": get_pod_logs,
    "restart_pod": restart_pod,
    "scale_deployment": scale_deployment,
    "diagnose_issue": diagnose_issue,
}

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "sre-agent"}

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
                "serverInfo": {"name": "sre-agent", "version": "1.0.0"}
            }
        }
    
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": body.get("id"), "result": {"tools": MCP_TOOLS}}
    
    if method == "tools/call":
        tool_name = body["params"]["name"]
        args = body["params"].get("arguments", {})
        
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
        
        try:
            result = await handler(**args)
            return {"jsonrpc": "2.0", "id": body.get("id"), "result": result}
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32000, "message": str(e)}}
    
    return {"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32601, "message": "Method not found"}}
```

---

## Part C: Create Dockerfile and Deployment

### Step C.1: Dockerfile

```dockerfile
# src/agents/sre_agent/Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Step C.2: Requirements

```text
# src/agents/sre_agent/requirements.txt
fastapi>=0.115.0
uvicorn>=0.30.0
azure-identity>=1.19.0
azure-monitor-query>=1.4.0
kubernetes>=31.0.0
httpx>=0.27.0
pydantic>=2.0.0
```

### Step C.3: Kubernetes Deployment

```yaml
# k8s/sre-agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sre-agent
  namespace: mcp-agents
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sre-agent
  template:
    metadata:
      labels:
        app: sre-agent
        azure.workload.identity/use: "true"
    spec:
      serviceAccountName: sre-agent-sa
      containers:
        - name: sre-agent
          image: ${CONTAINER_REGISTRY}/sre-agent:latest
          ports:
            - containerPort: 8080
          env:
            - name: LOG_ANALYTICS_WORKSPACE_ID
              valueFrom:
                secretKeyRef:
                  name: sre-agent-secrets
                  key: log-analytics-workspace-id
            - name: APPROVAL_LOGIC_APP_URL
              valueFrom:
                secretKeyRef:
                  name: sre-agent-secrets
                  key: approval-logic-app-url
            - name: K8S_NAMESPACE
              value: mcp-agents
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sre-agent-sa
  namespace: mcp-agents
  annotations:
    azure.workload.identity/client-id: ${SRE_AGENT_CLIENT_ID}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sre-agent-role
  namespace: mcp-agents
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "delete"]
  - apiGroups: ["apps"]
    resources: ["deployments", "deployments/scale"]
    verbs: ["get", "list", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sre-agent-binding
  namespace: mcp-agents
subjects:
  - kind: ServiceAccount
    name: sre-agent-sa
    namespace: mcp-agents
roleRef:
  kind: Role
  name: sre-agent-role
  apiGroup: rbac.authorization.k8s.io
```

---

## Part D: Test SRE Agent

### Step D.1: Deploy and Test

```powershell
# Build and push
cd src/agents/sre_agent
docker build -t sre-agent:latest .
docker tag sre-agent:latest $env:CONTAINER_REGISTRY/sre-agent:latest
docker push $env:CONTAINER_REGISTRY/sre-agent:latest

# Deploy
kubectl apply -f k8s/sre-agent-deployment.yaml

# Port forward
kubectl port-forward -n mcp-agents svc/sre-agent 8080:80

# Test health
Invoke-RestMethod -Uri "http://localhost:8080/health"

# Test get pod status
$body = @{
    jsonrpc = "2.0"
    id = 1
    method = "tools/call"
    params = @{
        name = "get_pod_status"
        arguments = @{
            namespace = "mcp-agents"
        }
    }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:8080/message" -Method Post -Body $body -ContentType "application/json"

# Test diagnose issue
$body = @{
    jsonrpc = "2.0"
    id = 2
    method = "tools/call"
    params = @{
        name = "diagnose_issue"
        arguments = @{
            issue_type = "high_latency"
            resource_name = "next-best-action-agent"
        }
    }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:8080/message" -Method Post -Body $body -ContentType "application/json"
```

---

## Verification Checklist

- [ ] SRE Agent specification created
- [ ] Agent implementation complete
- [ ] Docker image built and pushed
- [ ] Deployed to AKS with proper RBAC
- [ ] Health endpoint working
- [ ] Low-risk tools (query_logs, get_pod_status) working
- [ ] High-risk tools trigger approval workflow
- [ ] Diagnose tool providing actionable insights

---

## Summary

| Tool | Risk Level | Approval Required |
|------|------------|-------------------|
| query_logs | Low | No |
| get_pod_status | Low | No |
| get_pod_logs | Low | No |
| diagnose_issue | Low | No |
| restart_pod | High | Yes |
| scale_deployment | High | Yes |
| apply_manifest | High | Yes |

---

## Related Resources

- [Azure Monitor Query SDK](https://learn.microsoft.com/python/api/overview/azure/monitor-query-readme)
- [Kubernetes Python Client](https://github.com/kubernetes-client/python)
- [SRE Best Practices](https://sre.google/sre-book/table-of-contents/)
- [AGENTS_APPROVALS.md](../AGENTS_APPROVALS.md) - Approval workflow design
