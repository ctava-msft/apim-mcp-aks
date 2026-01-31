# Deployment Success Guide

## Overview

This document provides guidance on verifying a successful deployment of the MCP server on AKS with APIM.

## Verify Deployment

### Check Resource Group
```powershell
$rgName = azd env get-values | Select-String "AZURE_RESOURCE_GROUP_NAME" | ForEach-Object { ($_ -split '=')[1].Trim('"') }
az group show --name $rgName --query "properties.provisioningState" -o tsv
```

### Check AKS Cluster
```powershell
$aksName = azd env get-values | Select-String "AKS_CLUSTER_NAME" | ForEach-Object { ($_ -split '=')[1].Trim('"') }
az aks show --name $aksName --resource-group $rgName --query "provisioningState" -o tsv
```

### Check MCP Server Pods
```powershell
kubectl get pods -n mcp-server
kubectl get svc -n mcp-server
```

## Available MCP Tools

The MCP server provides three tools:

| Tool | Description | Parameters |
|------|-------------|------------|
| `hello_mcp` | Simple greeting tool for testing | None |
| `save_snippet` | Save text/code snippets to Azure Storage | `snippetname`, `snippet` |
| `get_snippet` | Retrieve saved snippets | `snippetname` |

## API Endpoints

### OAuth Endpoints
- **Authorization**: `https://<apim-name>.azure-api.net/oauth/authorize`
- **Token**: `https://<apim-name>.azure-api.net/oauth/token`
- **Well-Known**: `https://<apim-name>.azure-api.net/oauth/.well-known/oauth-authorization-server`

### MCP Endpoints
- **SSE**: `https://<apim-name>.azure-api.net/mcp/sse`
- **Message**: `https://<apim-name>.azure-api.net/mcp/message`

## Testing

### Run Integration Tests
```powershell
# Generate OAuth token first
python tests/test_apim_mcp_connection.py --generate-token

# Run MCP use case tests
python tests/test_apim_mcp_use_cases.py
```

### Manual Testing with MCP Inspector
```bash
npx @modelcontextprotocol/inspector
```

## Troubleshooting

### If pods fail to start
```powershell
# Check pod events
kubectl describe pod -n mcp-server

# Check logs
kubectl logs -n mcp-server deployment/mcp-server --all-containers=true
```

### If image pull fails
```powershell
# Login to ACR
$acrName = azd env get-values | Select-String "CONTAINER_REGISTRY" | ForEach-Object { ($_ -split '=')[1].Trim('"') }
az acr login --name $acrName

# Rebuild and push
./scripts/build-and-push.ps1

# Restart deployment
kubectl rollout restart deployment/mcp-server -n mcp-server
```

### If workload identity fails
```powershell
# Verify service account annotations
kubectl describe sa mcp-server-sa -n mcp-server
```

### If storage access fails
```powershell
# Verify managed identity has Storage Blob Data Contributor role
$identityId = azd env get-values | Select-String "MCP_SERVER_IDENTITY_CLIENT_ID" | ForEach-Object { ($_ -split '=')[1].Trim('"') }
az role assignment list --assignee $identityId --output table
```

## Monitoring

### View Application Insights
```powershell
$appInsightsName = azd env get-values | Select-String "APPLICATIONINSIGHTS_NAME" | ForEach-Object { ($_ -split '=')[1].Trim('"') }
az monitor app-insights component show --app $appInsightsName --resource-group $rgName
```

### View AKS Metrics
```powershell
kubectl top nodes
kubectl top pods -n mcp-server
```

## Scaling

### Scale MCP Server
```powershell
kubectl scale deployment mcp-server -n mcp-server --replicas=5
```

### Scale Node Pool
```powershell
az aks nodepool scale --cluster-name $aksName --resource-group $rgName --name system --node-count 3
```

## Cost Optimization

Current resources and estimated costs:
- AKS: 2 x Standard_DS2_v2 nodes (~$140/month)
- APIM: Developer tier (~$50/month)
- ACR: Standard (~$20/month)
- Storage: Standard_LRS (~$5/month)
- **Total**: ~$215/month

To reduce costs:
- Use APIM Consumption tier (pay-per-call)
- Scale AKS nodes down when not in use

---

## 🎉 Congratulations!

Your MCP server is successfully deployed and running on AKS! The infrastructure is production-ready and fully monitored. You can now integrate it with your AI agents through API Management.
