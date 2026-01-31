#!/bin/bash
set -e

# Post-provision hook for AKS setup

echo "🔧 Post-provision setup..."

# Get environment values from azd
echo ""
echo "📝 Loading environment values..."
eval $(azd env get-values | sed 's/^/export /')

AKS_NAME=$(echo $AKS_CLUSTER_NAME | tr -d '"')
RG_NAME=$(echo $AZURE_RESOURCE_GROUP_NAME | tr -d '"')
CONTAINER_REG=$(echo $CONTAINER_REGISTRY | tr -d '"')
STORAGE_URL=$(echo $AZURE_STORAGE_ACCOUNT_URL | tr -d '"')
MCP_IDENTITY_CLIENT_ID=$(echo $MCP_SERVER_IDENTITY_CLIENT_ID | tr -d '"')
MCP_PUBLIC_IP=$(echo $MCP_PUBLIC_IP_ADDRESS | tr -d '"')

echo "  AKS Cluster: $AKS_NAME"
echo "  Resource Group: $RG_NAME"
echo "  Container Registry: $CONTAINER_REG"
echo "  MCP Public IP: $MCP_PUBLIC_IP"

if [ -z "$AKS_NAME" ] || [ -z "$RG_NAME" ]; then
  echo "⚠️  Could not find AKS cluster name or resource group"
  exit 1
fi

# Get AKS credentials
echo ""
echo "🔑 Getting AKS credentials..."
az aks get-credentials --resource-group "$RG_NAME" --name "$AKS_NAME" --overwrite-existing --admin
echo "✅ AKS credentials configured"

# Grant current user AKS RBAC access
echo ""
echo "🔐 Granting AKS RBAC access..."
USER_ID=$(az ad signed-in-user show --query id -o tsv)
AKS_RESOURCE_ID=$(az aks show --resource-group "$RG_NAME" --name "$AKS_NAME" --query id -o tsv)
az role assignment create --role "Azure Kubernetes Service RBAC Cluster Admin" --assignee "$USER_ID" --scope "$AKS_RESOURCE_ID" 2>/dev/null || true
echo "✅ RBAC access granted"

# Attach ACR to AKS
echo ""
echo "🔗 Attaching ACR to AKS..."
ACR_NAME=$(echo "$CONTAINER_REG" | sed 's/\.azurecr\.io$//')
az aks update --resource-group "$RG_NAME" --name "$AKS_NAME" --attach-acr "$ACR_NAME"
echo "✅ ACR attached"

# Configure Kubernetes deployment files
echo ""
echo "📄 Configuring Kubernetes manifests..."

# Read and configure deployment template
sed -e "s|\${CONTAINER_REGISTRY}|$CONTAINER_REG|g" \
    -e "s|\${IMAGE_TAG}|latest|g" \
    -e "s|\${AZURE_STORAGE_ACCOUNT_URL}|$STORAGE_URL|g" \
    -e "s|\${AZURE_CLIENT_ID}|$MCP_IDENTITY_CLIENT_ID|g" \
    ./k8s/mcp-server-deployment.yaml > ./k8s/mcp-server-deployment-configured.yaml
echo "  ✅ Configured mcp-server-deployment-configured.yaml"

# Read and configure loadbalancer template
sed -e "s|\${AZURE_RESOURCE_GROUP_NAME}|$RG_NAME|g" \
    -e "s|\${MCP_PUBLIC_IP_ADDRESS}|$MCP_PUBLIC_IP|g" \
    ./k8s/mcp-server-loadbalancer.yaml > ./k8s/mcp-server-loadbalancer-configured.yaml
echo "  ✅ Configured mcp-server-loadbalancer-configured.yaml"

# Create federated identity for workload identity
echo ""
echo "🔐 Configuring workload identity..."
OIDC_ISSUER=$(az aks show --resource-group "$RG_NAME" --name "$AKS_NAME" --query "oidcIssuerProfile.issuerUrl" -o tsv)
IDENTITY_NAME="id-mcp-${AKS_NAME#aks-}"

# Check if federated credential already exists
if ! az identity federated-credential show --name mcp-server-federated --identity-name "$IDENTITY_NAME" --resource-group "$RG_NAME" 2>/dev/null; then
  az identity federated-credential create \
    --name mcp-server-federated \
    --identity-name "$IDENTITY_NAME" \
    --resource-group "$RG_NAME" \
    --issuer "$OIDC_ISSUER" \
    --subject "system:serviceaccount:mcp-server:mcp-server-sa" \
    --audience "api://AzureADTokenExchange"
  echo "✅ Federated identity credential created"
else
  echo "✅ Federated identity credential already exists"
fi

# Build and push container image
echo ""
echo "🐳 Building and pushing container image..."
export CONTAINER_REGISTRY="$CONTAINER_REG"
./scripts/build-and-push.sh

# Deploy to Kubernetes
echo ""
echo "🚀 Deploying to Kubernetes..."
kubectl apply -f ./k8s/mcp-server-deployment-configured.yaml
kubectl apply -f ./k8s/mcp-server-loadbalancer-configured.yaml

# Wait for deployment to be ready
echo ""
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/mcp-server -n mcp-server --timeout=300s

# Wait for LoadBalancer to get external IP
echo ""
echo "⏳ Waiting for LoadBalancer IP assignment..."
MAX_RETRIES=30
RETRY=0
LB_READY=false
while [ $RETRY -lt $MAX_RETRIES ] && [ "$LB_READY" = "false" ]; do
  LB_STATUS=$(kubectl get svc mcp-server-loadbalancer -n mcp-server -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
  if [ "$LB_STATUS" = "$MCP_PUBLIC_IP" ]; then
    LB_READY=true
    echo "✅ LoadBalancer ready with IP: $LB_STATUS"
  else
    echo "  Waiting for LoadBalancer... ($RETRY/$MAX_RETRIES)"
    sleep 10
    RETRY=$((RETRY + 1))
  fi
done

if [ "$LB_READY" = "false" ]; then
  echo "⚠️  LoadBalancer IP assignment timed out"
fi

# Generate test configuration
echo ""
echo "📝 Generating test configuration..."
./scripts/generate-test-config.sh

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🎉 Post-provision setup complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📝 Run integration tests:"
echo "   python tests/test_apim_mcp_connection.py --use-az-token"
echo ""

exit 0