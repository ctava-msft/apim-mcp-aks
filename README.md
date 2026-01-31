# MCP Server on AKS with APIM Gateway

Deploy Model Context Protocol (MCP) servers on Azure Kubernetes Service with Azure API Management as a secure gateway.

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   AI Agent      │────▶│  Azure API Management│────▶│  AKS Cluster    │
│ (Claude, etc.)  │     │  (OAuth + Gateway)   │     │  (MCP Server)   │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                   │                         │
                                   ▼                         ▼
                        ┌──────────────────┐      ┌─────────────────┐
                        │  Azure Entra ID  │      │  Azure Storage  │
                        │  (Authentication)│      │  (Snippets)     │
                        └──────────────────┘      └─────────────────┘
```

**Components:**
- **AKS Cluster** - Hosts MCP server pods with workload identity
- **API Management** - OAuth 2.0 gateway with rate limiting
- **Azure Storage** - Persistent storage for snippets
- **Entra ID** - Authentication provider

## Deployment

### Prerequisites

- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
- [Azure Developer CLI](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Docker](https://docs.docker.com/get-docker/)

### Deploy Infrastructure

```bash
azd auth login
azd up
```

This deploys: AKS cluster, Container Registry, API Management, Storage, and monitoring.

### Deploy MCP Server

```bash
# Get AKS credentials
az aks get-credentials --resource-group <rg-name> --name <aks-name>

# Build and push image
./scripts/build-and-push.sh

# Deploy to AKS
kubectl apply -f k8s/mcp-server-deployment.yaml
```

## Readiness Test

### Verify Pods

```bash
kubectl get pods -n mcp-server
```

Expected output:
```
NAME                         READY   STATUS    RESTARTS   AGE
mcp-server-xxx-xxx           1/1     Running   0          1m
```

### Test Health Endpoint

```bash
kubectl port-forward -n mcp-server svc/mcp-server 8000:80 &
curl http://localhost:8000/health
```

Expected: `{"status": "ok"}`

### Run Integration Tests

```bash
python tests/test_apim_mcp_connection.py --generate-token
python tests/test_apim_mcp_use_cases.py
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `hello_mcp` | Test connectivity |
| `save_snippet` | Save text to storage |
| `get_snippet` | Retrieve saved text |

## Endpoints

- **SSE**: `https://<apim>.azure-api.net/mcp/sse`
- **Message**: `https://<apim>.azure-api.net/mcp/message`
- **OAuth**: `https://<apim>.azure-api.net/oauth/authorize`
