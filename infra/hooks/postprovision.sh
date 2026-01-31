#!/bin/bash
set -e

# Post-provision hook for AKS setup

echo "🔧 Post-provision setup..."

# Generate test configuration
echo ""
echo "📝 Generating test configuration..."
./scripts/generate-test-config.sh

# Get AKS credentials
export AKS_NAME=$(azd env get-values | grep AKS_CLUSTER_NAME | cut -d'=' -f2 | tr -d '"')
export RG_NAME=$(azd env get-values | grep AZURE_RESOURCE_GROUP_NAME | cut -d'=' -f2 | tr -d '"')

if [ -n "$AKS_NAME" ] && [ -n "$RG_NAME" ]; then
  echo ""
  echo "Getting AKS credentials..."
  az aks get-credentials --resource-group "$RG_NAME" --name "$AKS_NAME" --overwrite-existing
  
  echo "✅ AKS credentials configured"
  echo ""
  echo "📝 Next steps:"
  echo "1. Build and deploy: ./scripts/build-and-push.sh"
  echo "2. Deploy MCP server: kubectl apply -f k8s/mcp-server-deployment.yaml"
  echo "3. Run tests: python tests/test_mcp_fixed_session.py"
else
  echo "⚠️  Could not find AKS cluster name or resource group"
fi

exit 0