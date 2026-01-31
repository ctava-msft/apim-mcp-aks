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
- **Static Public IP** - Stable endpoint for LoadBalancer service
- **Azure Storage** - Persistent storage for snippets
- **Entra ID** - Authentication provider

## Deployment

### Prerequisites

- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
- [Azure Developer CLI](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [Docker](https://docs.docker.com/get-docker/)

### Deploy Everything (Recommended)

The `azd up` command deploys infrastructure AND automatically configures:
- AKS cluster with Container Registry
- API Management with OAuth endpoints
- Static public IP for LoadBalancer
- MCP server deployment with workload identity
- LoadBalancer service connected to APIM backend

```bash
azd auth login
azd up
```

The post-provision hook automatically:
1. Configures AKS credentials and RBAC
2. Attaches ACR to AKS
3. Creates federated identity for workload identity
4. Builds and pushes the MCP server container image
5. Deploys Kubernetes resources (deployment + LoadBalancer)
6. Generates test configuration

### Manual Deployment (Advanced)

If you need to deploy manually or redeploy:

```bash
# Get AKS credentials
az aks get-credentials --resource-group <rg-name> --name <aks-name> --admin

# Build and push image
./scripts/build-and-push.ps1  # or .sh

# Deploy to AKS
kubectl apply -f k8s/mcp-server-deployment-configured.yaml
kubectl apply -f k8s/mcp-server-loadbalancer-configured.yaml
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

### Verify LoadBalancer

```bash
kubectl get svc -n mcp-server mcp-server-loadbalancer
```

Expected output shows the static public IP assigned.

### Run Integration Tests

```bash
python tests/test_apim_mcp_connection.py --use-az-token
```

This runs both infrastructure and MCP protocol tests.

Options:
- `--use-az-token` - Automatic token (no browser required)
- `--skip-infra` - Skip AKS infrastructure tests
- `--generate-token` - Interactive OAuth token generation
