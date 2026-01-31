# Deployment Notes

## Infrastructure Successfully Deployed! 🎉

### Deployed Resources

✅ **AKS Cluster**: Kubernetes cluster with system node pool  
✅ **Container Registry**: Azure Container Registry  
✅ **Storage Account**: Azure Blob Storage  
✅ **API Management**: APIM with OAuth support  
✅ **Virtual Network**: Optional VNet integration  
✅ **Log Analytics**: Centralized logging  
✅ **Application Insights**: Application monitoring  
✅ **Managed Identities**: For AKS, MCP workload, and Entra App  

### Configuration Notes

#### GPU Node Pool Status
⚠️ **GPU nodes are DISABLED** by default. This project focuses on MCP server deployment without language model hosting.

**Configuration Changes Made**:
1. **Kubernetes Version**: Updated to officially supported version
2. **Service CIDR**: Changed from `10.0.0.0/16` to `10.240.0.0/16` (avoid VNet overlap)
3. **Subnet Delegation**: Removed `Microsoft.App/environments` delegation (AKS incompatible)
4. **GPU Node Pool**: Disabled by default with `enableGpuNodePool` parameter

### Next Steps

Now that infrastructure is deployed, you need to manually complete these steps:

#### 1. Get AKS Credentials
```powershell
$aksName = azd env get-values | Select-String "AKS_CLUSTER_NAME" | ForEach-Object { ($_ -split '=')[1].Trim('"') }
$rgName = azd env get-values | Select-String "AZURE_RESOURCE_GROUP_NAME" | ForEach-Object { ($_ -split '=')[1].Trim('"') }
az aks get-credentials --resource-group $rgName --name $aksName --overwrite-existing
```

#### 2. Deploy MCP Server to Kubernetes
```powershell
# Deploy the MCP server
kubectl apply -f k8s/mcp-server-deployment.yaml

# Verify deployment
kubectl get pods
kubectl get services
```

#### 3. Test the Deployment
```powershell
# Run MCP endpoint tests
python tests/test_apim_mcp_connection.py

# Test MCP use cases
python tests/test_apim_mcp_use_cases.py
```

#### 4. Get APIM Endpoints
```powershell
# Get APIM gateway URL
$apimName = azd env get-values | Select-String "APIM_NAME" | ForEach-Object { ($_ -split '=')[1].Trim('"') }
az apim show --name $apimName --resource-group $rgName --query "gatewayUrl" --output tsv

# OAuth endpoints will be:
# - Authorization: https://<apim-name>.azure-api.net/oauth/authorize
# - Token: https://<apim-name>.azure-api.net/oauth/token
# - MCP SSE: https://<apim-name>.azure-api.net/mcp/sse
# - MCP Message: https://<apim-name>.azure-api.net/mcp/message
```

### Troubleshooting

#### If azd up shows storage account error
This is a known transient error that doesn't affect the deployed resources. All critical resources are deployed successfully. You can safely proceed with the next steps.

#### If Docker is not running
```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
# Wait for Docker Desktop to start (~30-60 seconds)
docker info  # Verify Docker is running
```

#### If APIM soft-delete error occurs
```powershell
az apim deletedservice purge --service-name <your-apim-name> --location eastus
```

#### If you need to restart deployment
```powershell
# Delete resource group
az group delete --name <your-resource-group> --yes --no-wait

# Wait for deletion complete, then redeploy
azd up
```

### What This Deployment Provides

✅ MCP Server deployment on AKS  
✅ API Management OAuth authentication  
✅ MCP protocol endpoints (/sse, /message)  
✅ Tool execution (hello_mcp, save_snippet, get_snippet)  
✅ Azure Storage integration with managed identity  
✅ Application Insights monitoring  

### Additional Resources

- [README.md](../README.md) - Main documentation
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Architecture diagrams
- [TEST_CONFIGURATION.md](./TEST_CONFIGURATION.md) - Testing setup

---

## Summary

Your infrastructure is **successfully deployed** and ready for MCP server deployment!
