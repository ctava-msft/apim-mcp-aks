# Deployment Notes

## Infrastructure Successfully Deployed! 🎉

**Last Updated:** January 31, 2026  
**Environment:** `<your-environment-name>`  
**Region:** Configurable (default: eastus2)

---

## Automated Deployment

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

---

## Deployed Resources

### Core Infrastructure
| Resource | Name | Status |
|----------|------|--------|
| Resource Group | `rg-<env-name>` | ✅ Deployed |
| AKS Cluster | `aks-<unique-id>` | ✅ Running (K8s 1.32+, 2 nodes) |
| Container Registry | `<registry>.azurecr.io` | ✅ Ready |
| Storage Account | `st<unique-id>` | ✅ Private (blob-private-endpoint) |
| API Management | `apim-<unique-id>` | ✅ Configured |
| Virtual Network | `vnet-<unique-id>` | ✅ VNet integration enabled |
| Log Analytics | `log-<unique-id>` | ✅ Centralized logging |
| Application Insights | Integrated | ✅ Application monitoring |
| Managed Identities | AKS, MCP workload, Agent Identity | ✅ Configured |

### AI Services
| Resource | Endpoint | Status |
|----------|----------|--------|
| Azure AI Foundry | `https://<ai-services>.services.ai.azure.com` | ✅ Ready |
| AI Search | `https://<search>.search.windows.net` | ✅ Ready |
| CosmosDB | `https://<cosmos>.documents.azure.com:443/` | ✅ Ready |

### MCP Server
| Component | Status | Details |
|-----------|--------|---------|
| Docker Image | ✅ Pushed | `<registry>.azurecr.io/mcp-agents:latest` |
| K8s Deployment | ✅ Running | 2 replicas healthy |
| LoadBalancer IP | ✅ Assigned | `<public-ip>` |
| Health Endpoint | ✅ Accessible | `http://<public-ip>/health` |
| Workload Identity | ✅ Configured | Using managed identity |
| AI Agent Client | ✅ Initialized | Connected to Azure AI Foundry |

### Fabric IQ (Optional)
| Setting | Value | Description |
|---------|-------|-------------|
| `fabricEnabled` | `false` | Fabric infrastructure disabled by default |
| Switch to Fabric | Set `FABRIC_ENABLED=true` | Enable when Fabric becomes available |

---

## Configuration Details

### Environment Variables (Kubernetes Deployment)

```yaml
# Core Settings
AZURE_STORAGE_ACCOUNT_URL: https://<storage-account>.blob.core.windows.net/
AZURE_CLIENT_ID: <managed-identity-client-id>

# Azure AI Agent SDK
FOUNDRY_PROJECT_ENDPOINT: https://<ai-services>.services.ai.azure.com/api/projects/<project>
AZURE_AI_PROJECT_ENDPOINT: https://<ai-services>.services.ai.azure.com/api/projects/<project>
FOUNDRY_MODEL_DEPLOYMENT_NAME: <chat-model-deployment>
AZURE_AI_MODEL_DEPLOYMENT_NAME: <chat-model-deployment>
EMBEDDING_MODEL_DEPLOYMENT_NAME: <embedding-model-deployment>

# Data Services
COSMOSDB_ENDPOINT: https://<cosmos>.documents.azure.com:443/
COSMOSDB_DATABASE_NAME: mcpdb
AZURE_SEARCH_ENDPOINT: https://<search>.search.windows.net
AZURE_SEARCH_INDEX_NAME: task-instructions

# Fabric IQ (disabled by default)
FABRIC_ENABLED: false
ONTOLOGY_CONTAINER_NAME: ontologies
```

### Network Security

- **Storage Account**: Public access disabled, accessible only via private endpoint
- **AKS Pods**: Access storage over private network within VNet
- **Private Endpoints**: blob-private-endpoint, queue-private-endpoint

---

## MCP Server Endpoints

| Endpoint | URL | Status |
|----------|-----|--------|
| Health Check | `http://<public-ip>/health` | ✅ Working |
| Root | `http://<public-ip>/` | ✅ Working |
| MCP SSE | `http://<public-ip>/runtime/webhooks/mcp/sse` | ✅ Available |
| MCP Message | `http://<public-ip>/runtime/webhooks/mcp/message` | ✅ Available |
| Agent Chat | `http://<public-ip>/agent/chat` | ⚠️ SDK compatibility issue |
| Agent Chat Stream | `http://<public-ip>/agent/chat/stream` | ⚠️ SDK compatibility issue |

---

## Quick Commands

### Get AKS Credentials
```powershell
# Replace with your resource group and cluster name
az aks get-credentials --resource-group <resource-group> --name <aks-cluster-name> --admin --overwrite-existing
```

### Check Deployment Status
```powershell
kubectl get pods -n mcp-agents
kubectl get svc -n mcp-agents
kubectl logs -n mcp-agents deployment/mcp-agents --tail=50
```

### Test Health Endpoint
```powershell
$LB_IP = kubectl get svc -n mcp-agents mcp-agents-loadbalancer -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
Invoke-RestMethod -Uri "http://$LB_IP/health" -Method GET
```

### Redeploy MCP Server
```powershell
kubectl apply -f k8s/mcp-agents-deployment-configured.yaml
kubectl rollout restart deployment/mcp-agents -n mcp-agents
kubectl rollout status deployment/mcp-agents -n mcp-agents
```

### Upload Ontologies to Storage
```powershell
# Get storage account name from your deployment
$STORAGE_ACCOUNT = "<your-storage-account>"

# Temporarily enable public access
az storage account update --name $STORAGE_ACCOUNT --public-network-access Enabled
$myIP = (Invoke-RestMethod -Uri "https://api.ipify.org")
az storage account network-rule add --account-name $STORAGE_ACCOUNT --ip-address $myIP

# Upload files
az storage blob upload --account-name $STORAGE_ACCOUNT --container-name ontologies --name myfile.json --file ./myfile.json --auth-mode login

# Restore security
az storage account update --name $STORAGE_ACCOUNT --public-network-access Disabled
```

---

## Switching to Fabric Mode

When Microsoft Fabric becomes available in your tenant:

### 1. Enable Fabric Infrastructure
```powershell
azd env set FABRIC_ENABLED true
azd provision
```

### 2. Update Kubernetes Deployment
Update the `FABRIC_ENABLED` environment variable in `k8s/mcp-agents-deployment-configured.yaml`:
```yaml
- name: FABRIC_ENABLED
  value: "true"
- name: FABRIC_ENDPOINT
  value: "<your-fabric-endpoint>"
- name: FABRIC_WORKSPACE_ID
  value: "<your-workspace-id>"
```

### 3. Redeploy
```powershell
kubectl apply -f k8s/mcp-agents-deployment-configured.yaml
kubectl rollout restart deployment/mcp-agents -n mcp-agents
```

The FactsMemory provider will automatically switch from Azure Blob Storage to Fabric IQ OneLake.

---

## Known Issues

### 1. Agent Chat SDK Compatibility
**Status:** ⚠️ Needs Fix  
**Description:** The `/agent/chat` endpoint returns error: `'AzureAIAgentClient' object has no attribute 'run'`  
**Root Cause:** The `agent_framework` SDK interface changed; code uses `.run()` but SDK expects different method.  
**Workaround:** Use MCP protocol endpoints (`/runtime/webhooks/mcp/sse`, `/runtime/webhooks/mcp/message`) instead.

### 2. Ontology File Format
**Status:** ℹ️ Documentation  
**Description:** Task instruction files in `task_instructions/` are for AI Search long-term memory, not FactsMemory ontologies.  
**Impact:** FactsMemory falls back to sample data since no properly formatted ontology files exist.  
**Fix:** Create ontology JSON files with `entities`, `relationships`, and `facts` arrays for FactsMemory.

---

## Troubleshooting

### DeploymentActive Error
If you see "DeploymentActive" error during `azd provision`, wait and retry:
```powershell
Start-Sleep -Seconds 60
azd provision --no-prompt
```

### Storage Key Authentication Error
This occurs when Agent Identity is enabled before storage account is ready:
```
Key based authentication is not permitted on this storage account
```
**Solution:** Use the two-phase deployment process (disable Agent Identity first, then enable after core infrastructure deploys).

### If azd up shows storage account error
Transient error that doesn't affect deployed resources. Proceed with next steps.

### If Docker is not running
```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
docker info  # Verify Docker is running
```

### If APIM soft-delete error occurs
```powershell
az apim deletedservice purge --service-name <your-apim-name> --location eastus2
```

### If pods are in CrashLoopBackOff
```powershell
kubectl logs -n mcp-agents deployment/mcp-agents --previous
kubectl describe pod -n mcp-agents <pod-name>
```

### If storage access is denied from pods
Verify the managed identity has `Storage Blob Data Contributor` role on the storage account.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    Azure Resource Group                      │
                    │                     rg-<env-name>                             │
                    └─────────────────────────────────────────────────────────────┘
                                                 │
         ┌───────────────────────────────────────┼───────────────────────────────────────┐
         │                                       │                                       │
         ▼                                       ▼                                       ▼
┌─────────────────┐               ┌───────────────────────────┐               ┌─────────────────┐
│  Azure API      │               │     AKS Cluster           │               │  Azure AI       │
│  Management     │◄─────────────►│   (2 nodes, K8s 1.32+)     │◄─────────────►│  Foundry        │
│  (OAuth)        │               │                           │               │  (GPT model)    │
└─────────────────┘               └───────────────────────────┘               └─────────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
         ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
         │  MCP Server     │       │  Storage Account│       │  CosmosDB       │
         │  Pods (2x)      │◄─────►│  (Private EP)   │       │  (mcpdb)        │
         │                 │       │                 │       │                 │
         └─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## What This Deployment Provides

✅ MCP Server deployment on AKS with 2 replicas  
✅ Azure AI Agent Framework integration  
✅ Workload Identity for secure Azure service access  
✅ Private network storage via blob-private-endpoint  
✅ API Management with OAuth authentication  
✅ MCP protocol endpoints (/sse, /message)  
✅ Health monitoring and logging  
✅ Switchable Fabric IQ integration (disabled by default)

---

## Additional Resources

- [README.md](../README.md) - Main documentation
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Architecture diagrams
- [TEST_CONFIGURATION.md](./TEST_CONFIGURATION.md) - Testing setup
- [TEST_RESULTS.md](./TEST_RESULTS.md) - Test results

---

**Summary:** Infrastructure is **successfully deployed** and MCP server is running! 🎉

