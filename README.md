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
kubectl apply -f k8s/mcp-agents-deployment-configured.yaml
kubectl apply -f k8s/mcp-agents-loadbalancer-configured.yaml
```

## Readiness Test

### Verify Pods

```bash
kubectl get pods -n mcp-agents
```

Expected output:
```
NAME                         READY   STATUS    RESTARTS   AGE
mcp-agents-xxx-xxx           1/1     Running   0          1m
```

### Verify LoadBalancer

```bash
kubectl get svc -n mcp-agents mcp-agents-loadbalancer
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

---

## Fully Automated Deployment

### Two-Phase Deployment Process

The deployment uses a two-phase approach to handle Entra Agent Identity requirements:

**Phase 1: Core Infrastructure** (Agent Identity disabled)
```powershell
# Login to Azure
azd auth login

# Disable Agent Identity for initial deployment
azd env set AZURE_AGENT_IDENTITY_ENABLED false

# Deploy core infrastructure
azd up
```

**Phase 2: Enable Agent Identity** (after core infrastructure is ready)
```powershell
# Enable Agent Identity
azd env set AZURE_AGENT_IDENTITY_ENABLED true

# Re-provision to create Agent Identity
azd provision
```

### Why Two Phases?

The Agent Identity Blueprint requires:
1. A storage account for deployment scripts (created in Phase 1)
2. Key-based authentication or managed identity access (configured in Phase 1)

By deploying core infrastructure first, we ensure all dependencies are in place before creating the Agent Identity.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_AGENT_IDENTITY_ENABLED` | `true` | Enable/disable Entra Agent Identity |
| `AZURE_LOCATION` | `eastus2` | Azure region for deployment |
| `FABRIC_ENABLED` | `false` | Enable Microsoft Fabric integration |

### Complete Automated Script

```powershell
# Full automated deployment script
azd auth login

# Phase 1: Core Infrastructure
azd env set AZURE_AGENT_IDENTITY_ENABLED false
azd up --no-prompt

# Wait for deployment to stabilize
Start-Sleep -Seconds 30

# Phase 2: Enable Agent Identity  
azd env set AZURE_AGENT_IDENTITY_ENABLED true
azd provision --no-prompt

# Verify deployment
kubectl get pods -n mcp-agents
kubectl get svc -n mcp-agents

# Run tests
python tests/test_apim_mcp_connection.py --use-az-token
```

### Troubleshooting

#### DeploymentActive Error
If you see "DeploymentActive" error, wait 60 seconds and retry:
```powershell
Start-Sleep -Seconds 60
azd provision --no-prompt
```

#### Storage Key Authentication Error
This occurs when Agent Identity is enabled before storage account is ready. Use the two-phase deployment process above.

#### APIM Soft-Delete Conflict
```powershell
az apim deletedservice purge --service-name <apim-name> --location eastus2
```

---

## Identity Architecture

This deployment uses multiple managed identities:

| Identity | Purpose | Scope |
|----------|---------|-------|
| `id-aks-*` | AKS control plane operations | Cluster management |
| `id-mcp-*` | MCP workload identity (legacy) | Pod authentication |
| `id-agent-*` | Entra Agent Identity | AI agent authentication |
| `id-api-*` | API Management identity | APIM operations |
| `id-ai-*` | AI Services identity | Model access |

See [docs/IDENTITY_DESIGN.md](docs/IDENTITY_DESIGN.md) for detailed identity architecture.

---

## Configuration

### Kubernetes Environment Variables

The MCP server pods are configured with these environment variables:

```yaml
# Azure Identity
AZURE_CLIENT_ID: <managed-identity-client-id>

# AI Services
FOUNDRY_PROJECT_ENDPOINT: https://<ai-services>.services.ai.azure.com/api/projects/<project>
FOUNDRY_MODEL_DEPLOYMENT_NAME: <chat-model-deployment>
EMBEDDING_MODEL_DEPLOYMENT_NAME: <embedding-model-deployment>

# Data Services
COSMOSDB_ENDPOINT: https://<cosmos>.documents.azure.com:443/
COSMOSDB_DATABASE_NAME: mcpdb
AZURE_SEARCH_ENDPOINT: https://<search>.search.windows.net
AZURE_SEARCH_INDEX_NAME: task-instructions

# Storage
AZURE_STORAGE_ACCOUNT_URL: https://<storage>.blob.core.windows.net/
ONTOLOGY_CONTAINER_NAME: ontologies

# Feature Flags
FABRIC_ENABLED: false
```

---

## Agent Lightning (Fine-Tuning & RL)

Agent Lightning enables reinforcement learning and fine-tuning for MCP agents. It captures agent interactions, labels them with rewards, builds training datasets, and manages fine-tuned model deployments.

### Enable Lightning Capture

1. **Deploy Lightning Cosmos containers** (automatically included in `azd provision`):
   - Database: `agent_rl`
   - Containers: `rl_episodes`, `rl_rewards`, `rl_datasets`, `rl_training_runs`, `rl_deployments`

2. **Enable capture in Kubernetes**:
   ```bash
   kubectl set env deployment/mcp-agents -n mcp-agents ENABLE_LIGHTNING_CAPTURE=true
   ```

3. **Verify Lightning is working**:
   ```bash
   python tests/test_demo_lightning_loop.py --direct
   ```

### Lightning Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_LIGHTNING_CAPTURE` | `false` | Enable episode capture for training |
| `USE_TUNED_MODEL` | `false` | Use fine-tuned model if available |
| `LIGHTNING_AGENT_ID` | `mcp-agents` | Agent identifier for episodes |
| `COSMOS_ACCOUNT_URI` | Same as `COSMOSDB_ENDPOINT` | Cosmos endpoint for Lightning |
| `COSMOS_DATABASE_NAME` | `agent_rl` | Lightning database name |

### Lightning Workflow

```
1. Capture Episodes     →  ask_foundry, next_best_action calls
2. Label with Rewards   →  python -m lightning.cli label
3. Build Dataset        →  python -m lightning.cli build-dataset
4. Fine-tune Model      →  python -m lightning.cli train
5. Promote Model        →  python -m lightning.cli promote
6. Enable Tuned Model   →  kubectl set env ... USE_TUNED_MODEL=true
```

See [docs/AGENT-LIGHTNING.md](docs/AGENT-LIGHTNING.md) for complete documentation.

---

## Additional Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture diagrams
- [DEPLOYMENT_NOTES.md](docs/DEPLOYMENT_NOTES.md) - Detailed deployment notes
- [IDENTITY_DESIGN.md](docs/IDENTITY_DESIGN.md) - Identity architecture design
- [AGENT-LIGHTNING.md](docs/AGENT-LIGHTNING.md) - Fine-tuning and RL documentation
