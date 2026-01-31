#!/usr/bin/env pwsh
# Post-provision hook for AKS setup

Write-Host "🔧 Post-provision setup..." -ForegroundColor Cyan

# Generate test configuration
Write-Host "`n📝 Generating test configuration..." -ForegroundColor Cyan
& "./scripts/generate-test-config.ps1"

# Get AKS credentials
$aksName = azd env get-values | Select-String "AKS_CLUSTER_NAME" | ForEach-Object { ($_ -split '=')[1].Trim('"') }
$rgName = azd env get-values | Select-String "AZURE_RESOURCE_GROUP_NAME" | ForEach-Object { ($_ -split '=')[1].Trim('"') }

if ($aksName -and $rgName) {
  Write-Host "`nGetting AKS credentials..." -ForegroundColor Yellow
  az aks get-credentials --resource-group $rgName --name $aksName --overwrite-existing
  
  Write-Host "✅ AKS credentials configured" -ForegroundColor Green
  Write-Host ""
  Write-Host "📝 Next steps:" -ForegroundColor Cyan
  Write-Host "1. Build and deploy: ./scripts/build-and-push.ps1" -ForegroundColor White
  Write-Host "2. Deploy MCP server: kubectl apply -f k8s/mcp-server-deployment.yaml" -ForegroundColor White
  Write-Host "3. Run tests: python tests/test_mcp_fixed_session.py" -ForegroundColor White
} else {
  Write-Host "⚠️  Could not find AKS cluster name or resource group" -ForegroundColor Yellow
}
