# Optional Exercise: Fabric Data Agents

**Objective:** Build AI agents that interact with Microsoft Fabric's data platform (Lakehouses, Warehouses, Pipelines, Semantic Models).

**Duration:** 2-3 hours

---

## Overview

Fabric Data Agents enable enterprise AI agents to interact with Microsoft Fabric's unified data platform:

| Agent Type | Capability | Query Language |
|------------|------------|----------------|
| **Lakehouse Agent** | Query/write to Fabric Lakehouses | Spark SQL |
| **Warehouse Agent** | Execute queries against Data Warehouses | T-SQL |
| **Pipeline Agent** | Trigger, monitor, manage Data Pipelines | REST API |
| **Semantic Model Agent** | Query Power BI semantic models | DAX/MDX |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Azure Agents Control Plane                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   APIM      │  │  Entra ID   │  │   Monitor   │  │    Cosmos DB        │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────────┐
          │         Fabric Data Agents               │
          │  ┌───────────┐  ┌───────────┐           │
          │  │ Lakehouse │  │ Warehouse │           │
          │  │   Agent   │  │   Agent   │           │
          │  └───────────┘  └───────────┘           │
          │  ┌───────────┐  ┌───────────┐           │
          │  │ Pipeline  │  │  Semantic │           │
          │  │   Agent   │  │   Model   │           │
          │  └───────────┘  └───────────┘           │
          └─────────────────────────────────────────┘
                            │
                            ▼
          ┌─────────────────────────────────────────┐
          │         Microsoft Fabric                 │
          │  ┌───────────┐  ┌───────────┐           │
          │  │ Lakehouse │  │ Data      │           │
          │  │           │  │ Warehouse │           │
          │  └───────────┘  └───────────┘           │
          └─────────────────────────────────────────┘
```

---

## Prerequisites

1. **Fabric Capacity**: Microsoft Fabric capacity deployed (F2 or higher)
2. **Fabric Workspace**: Workspace created with Lakehouse/Warehouse
3. **RBAC**: Agent identity assigned Reader and Contributor roles
4. **Configuration**: Environment variables set in K8s deployment

---

## Configuration

### Enable Fabric Data Agents

Update `main.parameters.json`:

```json
{
  "fabricEnabled": {
    "value": true
  },
  "fabricDataAgentsEnabled": {
    "value": true
  },
  "fabricWorkspaceId": {
    "value": "abcd1234-5678-90ab-cdef-ghijklmnopqr"
  },
  "fabricCapacityName": {
    "value": "fabriccapacitydev01"
  }
}
```

Deploy:

```bash
azd provision
azd deploy
```

---

## Part A: Lakehouse Agent - Customer Churn Analysis

### Step A.1: MCP Tool Call

Query a Fabric Lakehouse to analyze customer churn risk:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "fabric_query_lakehouse",
    "arguments": {
      "lakehouse_id": "lh-customer-analytics-001",
      "lakehouse_name": "CustomerAnalytics",
      "query": "SELECT customer_id, name, email, churn_risk, segment, monthly_spend FROM customers WHERE churn_risk > 0.7 ORDER BY churn_risk DESC LIMIT 20"
    }
  }
}
```

### Step A.2: Response

```json
{
  "success": true,
  "lakehouse_id": "lh-customer-analytics-001",
  "lakehouse_name": "CustomerAnalytics",
  "results": {
    "schema": [
      {"name": "customer_id", "type": "string"},
      {"name": "churn_risk", "type": "float"},
      {"name": "segment", "type": "string"},
      {"name": "monthly_spend", "type": "float"}
    ],
    "rows": [
      {
        "customer_id": "CUST001",
        "name": "Acme Corp",
        "churn_risk": 0.92,
        "segment": "enterprise",
        "monthly_spend": 15000.00
      }
    ]
  }
}
```

### Agent Actions

Based on results, the agent can:
1. Generate retention recommendations
2. Trigger automated retention workflows
3. Alert account managers

---

## Part B: Warehouse Agent - Sales Analysis

### Step B.1: MCP Tool Call

Query Fabric Data Warehouse for sales analytics:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "fabric_query_warehouse",
    "arguments": {
      "warehouse_id": "wh-sales-dwh-prod",
      "warehouse_name": "SalesDataWarehouse",
      "query": "SELECT TOP 10 region, SUM(revenue) as total_revenue, COUNT(DISTINCT customer_id) as customer_count FROM sales_summary WHERE sale_date >= DATEADD(month, -3, GETDATE()) GROUP BY region ORDER BY total_revenue DESC"
    }
  }
}
```

### Step B.2: Response

```json
{
  "success": true,
  "results": {
    "rows": [
      {"region": "North America", "total_revenue": 5250000.00, "customer_count": 1250},
      {"region": "Europe", "total_revenue": 3180000.00, "customer_count": 890}
    ]
  }
}
```

---

## Part C: Pipeline Agent - ETL Orchestration

### Step C.1: Trigger Data Pipeline

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "fabric_trigger_pipeline",
    "arguments": {
      "workspace_id": "ws-dataops-prod",
      "pipeline_name": "CustomerDataRefresh",
      "parameters": {
        "source_date": "2026-02-07",
        "incremental": true
      }
    }
  }
}
```

### Step C.2: Monitor Pipeline Status

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "fabric_get_pipeline_status",
    "arguments": {
      "workspace_id": "ws-dataops-prod",
      "pipeline_name": "CustomerDataRefresh",
      "run_id": "run-123456"
    }
  }
}
```

---

## Part D: Semantic Model Agent - Business Intelligence

### Step D.1: Query Power BI Semantic Model with DAX

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "fabric_query_semantic_model",
    "arguments": {
      "dataset_id": "ds-sales-analytics",
      "dax_query": "EVALUATE SUMMARIZECOLUMNS('Date'[Year], 'Product'[Category], \"Total Sales\", SUM('Sales'[Amount]))"
    }
  }
}
```

---

## Tool Implementation

### fabric_tools.py

```python
import os
import httpx
from azure.identity import DefaultAzureCredential
from typing import Dict, Any, List

FABRIC_API_ENDPOINT = os.environ.get("FABRIC_API_ENDPOINT", "https://api.fabric.microsoft.com/v1")
FABRIC_WORKSPACE_ID = os.environ.get("FABRIC_WORKSPACE_ID")

credential = DefaultAzureCredential()

async def get_fabric_token() -> str:
    """Get access token for Fabric API."""
    token = credential.get_token("https://api.fabric.microsoft.com/.default")
    return token.token

async def query_lakehouse(lakehouse_id: str, lakehouse_name: str, query: str) -> Dict[str, Any]:
    """Execute Spark SQL query against Fabric Lakehouse."""
    token = await get_fabric_token()
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{FABRIC_API_ENDPOINT}/workspaces/{FABRIC_WORKSPACE_ID}/lakehouses/{lakehouse_id}/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()

async def query_warehouse(warehouse_id: str, warehouse_name: str, query: str) -> Dict[str, Any]:
    """Execute T-SQL query against Fabric Data Warehouse."""
    token = await get_fabric_token()
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{FABRIC_API_ENDPOINT}/workspaces/{FABRIC_WORKSPACE_ID}/warehouses/{warehouse_id}/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": query}
        )
        response.raise_for_status()
        return response.json()

async def trigger_pipeline(workspace_id: str, pipeline_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger a Fabric Data Pipeline."""
    token = await get_fabric_token()
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{FABRIC_API_ENDPOINT}/workspaces/{workspace_id}/pipelines/{pipeline_name}/runs",
            headers={"Authorization": f"Bearer {token}"},
            json={"parameters": parameters}
        )
        response.raise_for_status()
        return response.json()
```

---

## Verification

### Test Lakehouse Query

```powershell
kubectl port-forward -n mcp-agents svc/next-best-action-agent 8080:80

$body = @{
    jsonrpc = "2.0"
    id = 1
    method = "tools/call"
    params = @{
        name = "fabric_query_lakehouse"
        arguments = @{
            lakehouse_id = "lh-customer-analytics"
            lakehouse_name = "CustomerAnalytics"
            query = "SELECT COUNT(*) as total FROM customers"
        }
    }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:8080/message" -Method Post -Body $body -ContentType "application/json"
```

### Verify in Azure Monitor

```kusto
customEvents
| where name == "FabricToolCall"
| summarize count() by tostring(customDimensions["tool_name"]), bin(timestamp, 1h)
| render timechart
```

---

## Summary

| Agent | Data Source | Query Type | Use Cases |
|-------|-------------|------------|-----------|
| Lakehouse | Fabric Lakehouse | Spark SQL | Analytics, ML features |
| Warehouse | Data Warehouse | T-SQL | BI, Reporting |
| Pipeline | Data Pipelines | REST API | ETL Orchestration |
| Semantic Model | Power BI | DAX | Business Intelligence |

---

## Related Resources

- [Microsoft Fabric Documentation](https://learn.microsoft.com/fabric/)
- [Fabric REST API Reference](https://learn.microsoft.com/rest/api/fabric/)
- [FABRIC_DATA_AGENTS_EXAMPLES.md](../FABRIC_DATA_AGENTS_EXAMPLES.md) - Full examples
