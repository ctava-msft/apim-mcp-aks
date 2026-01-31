#!/usr/bin/env pwsh
# Post-provision hook for AKS setup

Write-Host "🔧 Post-provision setup..." -ForegroundColor Cyan

# Get environment values from azd
Write-Host "`n📝 Loading environment values..." -ForegroundColor Cyan
$envValues = azd env get-values | ConvertFrom-StringData -Delimiter '='

$aksName = $envValues.AKS_CLUSTER_NAME.Trim('"')
$rgName = $envValues.AZURE_RESOURCE_GROUP_NAME.Trim('"')
$containerRegistry = $envValues.CONTAINER_REGISTRY.Trim('"')
$storageUrl = $envValues.AZURE_STORAGE_ACCOUNT_URL.Trim('"')
$mcpIdentityClientId = $envValues.MCP_SERVER_IDENTITY_CLIENT_ID.Trim('"')
$mcpPublicIpAddress = $envValues.MCP_PUBLIC_IP_ADDRESS.Trim('"')

Write-Host "  AKS Cluster: $aksName" -ForegroundColor White
Write-Host "  Resource Group: $rgName" -ForegroundColor White
Write-Host "  Container Registry: $containerRegistry" -ForegroundColor White
Write-Host "  MCP Public IP: $mcpPublicIpAddress" -ForegroundColor White

if (-not $aksName -or -not $rgName) {
  Write-Host "⚠️  Could not find AKS cluster name or resource group" -ForegroundColor Yellow
  exit 1
}

# Get AKS credentials
Write-Host "`n🔑 Getting AKS credentials..." -ForegroundColor Cyan
az aks get-credentials --resource-group $rgName --name $aksName --overwrite-existing --admin
Write-Host "✅ AKS credentials configured" -ForegroundColor Green

# Grant current user AKS RBAC access
Write-Host "`n🔐 Granting AKS RBAC access..." -ForegroundColor Cyan
$userId = az ad signed-in-user show --query id -o tsv
$aksResourceId = az aks show --resource-group $rgName --name $aksName --query id -o tsv
az role assignment create --role "Azure Kubernetes Service RBAC Cluster Admin" --assignee $userId --scope $aksResourceId 2>$null
Write-Host "✅ RBAC access granted" -ForegroundColor Green

# Attach ACR to AKS
Write-Host "`n🔗 Attaching ACR to AKS..." -ForegroundColor Cyan
$acrName = $containerRegistry -replace '\.azurecr\.io$', ''
az aks update --resource-group $rgName --name $aksName --attach-acr $acrName
Write-Host "✅ ACR attached" -ForegroundColor Green

# Configure Kubernetes deployment files
Write-Host "`n📄 Configuring Kubernetes manifests..." -ForegroundColor Cyan

# Read and configure deployment template
$deploymentTemplate = Get-Content -Path "./k8s/mcp-server-deployment.yaml" -Raw
$configuredDeployment = $deploymentTemplate `
  -replace '\$\{CONTAINER_REGISTRY\}', $containerRegistry `
  -replace '\$\{IMAGE_TAG\}', 'latest' `
  -replace '\$\{AZURE_STORAGE_ACCOUNT_URL\}', $storageUrl `
  -replace '\$\{AZURE_CLIENT_ID\}', $mcpIdentityClientId
$configuredDeployment | Out-File -FilePath "./k8s/mcp-server-deployment-configured.yaml" -Encoding utf8
Write-Host "  ✅ Configured mcp-server-deployment-configured.yaml" -ForegroundColor Green

# Read and configure loadbalancer template
$lbTemplate = Get-Content -Path "./k8s/mcp-server-loadbalancer.yaml" -Raw
$configuredLb = $lbTemplate `
  -replace '\$\{AZURE_RESOURCE_GROUP_NAME\}', $rgName `
  -replace '\$\{MCP_PUBLIC_IP_ADDRESS\}', $mcpPublicIpAddress
$configuredLb | Out-File -FilePath "./k8s/mcp-server-loadbalancer-configured.yaml" -Encoding utf8
Write-Host "  ✅ Configured mcp-server-loadbalancer-configured.yaml" -ForegroundColor Green

# Create federated identity for workload identity
Write-Host "`n🔐 Configuring workload identity..." -ForegroundColor Cyan
$oidcIssuer = az aks show --resource-group $rgName --name $aksName --query "oidcIssuerProfile.issuerUrl" -o tsv
$identityName = "id-mcp-" + ($aksName -replace '^aks-', '')

# Check if federated credential already exists
$existingCred = az identity federated-credential show --name mcp-server-federated --identity-name $identityName --resource-group $rgName 2>$null
if (-not $existingCred) {
  az identity federated-credential create `
    --name mcp-server-federated `
    --identity-name $identityName `
    --resource-group $rgName `
    --issuer $oidcIssuer `
    --subject "system:serviceaccount:mcp-server:mcp-server-sa" `
    --audience "api://AzureADTokenExchange"
  Write-Host "✅ Federated identity credential created" -ForegroundColor Green
} else {
  Write-Host "✅ Federated identity credential already exists" -ForegroundColor Green
}

# Build and push container image
Write-Host "`n🐳 Building and pushing container image..." -ForegroundColor Cyan
$env:CONTAINER_REGISTRY = $containerRegistry
& "./scripts/build-and-push.ps1"

# Deploy to Kubernetes
Write-Host "`n🚀 Deploying to Kubernetes..." -ForegroundColor Cyan
kubectl apply -f ./k8s/mcp-server-deployment-configured.yaml
kubectl apply -f ./k8s/mcp-server-loadbalancer-configured.yaml

# Wait for deployment to be ready
Write-Host "`n⏳ Waiting for deployment to be ready..." -ForegroundColor Cyan
kubectl rollout status deployment/mcp-server -n mcp-server --timeout=300s

# Wait for LoadBalancer to get external IP
Write-Host "`n⏳ Waiting for LoadBalancer IP assignment..." -ForegroundColor Cyan
$maxRetries = 30
$retry = 0
$lbReady = $false
while ($retry -lt $maxRetries -and -not $lbReady) {
  $lbStatus = kubectl get svc mcp-server-loadbalancer -n mcp-server -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>$null
  if ($lbStatus -eq $mcpPublicIpAddress) {
    $lbReady = $true
    Write-Host "✅ LoadBalancer ready with IP: $lbStatus" -ForegroundColor Green
  } else {
    Write-Host "  Waiting for LoadBalancer... ($retry/$maxRetries)" -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    $retry++
  }
}

if (-not $lbReady) {
  Write-Host "⚠️  LoadBalancer IP assignment timed out" -ForegroundColor Yellow
}

# Generate test configuration
Write-Host "`n📝 Generating test configuration..." -ForegroundColor Cyan
& "./scripts/generate-test-config.ps1"

Write-Host "`n" -NoNewline
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "🎉 Post-provision setup complete!" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "`n📝 Run integration tests:" -ForegroundColor Cyan
Write-Host "   python tests/test_apim_mcp_connection.py --use-az-token" -ForegroundColor White
Write-Host ""
