#!/bin/bash
# Generate test configuration from Azure deployment outputs

set -e

DEPLOYMENT_NAME="${1:-main}"
OUTPUT_DIR="${2:-tests}"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       Generating Test Configuration from Deployment     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Get current subscription
SUBSCRIPTION_INFO=$(az account show)
SUBSCRIPTION_ID=$(echo "$SUBSCRIPTION_INFO" | jq -r '.id')
TENANT_ID=$(echo "$SUBSCRIPTION_INFO" | jq -r '.tenantId')
SUBSCRIPTION_NAME=$(echo "$SUBSCRIPTION_INFO" | jq -r '.name')

echo "📍 Subscription: $SUBSCRIPTION_NAME"
echo "📍 Tenant: $TENANT_ID"
echo ""

# Find the most recent deployment if no deployment name specified
if [ "$DEPLOYMENT_NAME" = "main" ]; then
    echo "🔍 Finding most recent deployment..."
    RECENT_DEPLOYMENT=$(az deployment sub list --query "[?starts_with(name, 'apim-mcp')] | [0].name" -o tsv)
    if [ -n "$RECENT_DEPLOYMENT" ]; then
        DEPLOYMENT_NAME="$RECENT_DEPLOYMENT"
        echo "✅ Found deployment: $DEPLOYMENT_NAME"
    else
        echo "⚠️  No recent deployment found, checking 'main'..."
    fi
fi

# Get deployment outputs
echo ""
echo "📤 Retrieving deployment outputs..."
OUTPUTS=$(az deployment sub show --name "$DEPLOYMENT_NAME" --query "properties.outputs" -o json 2>/dev/null) || {
    echo "❌ Error: Could not retrieve deployment outputs"
    echo "   Make sure the deployment exists and completed successfully"
    echo "   Deployment name: $DEPLOYMENT_NAME"
    exit 1
}

# Extract values using jq
APIM_BASE_URL=$(echo "$OUTPUTS" | jq -r '.MCP_BASE_URL.value // empty')
APIM_GATEWAY_URL=$(echo "$OUTPUTS" | jq -r '.APIM_GATEWAY_URL.value // empty')
APIM_OAUTH_AUTHORIZE_URL=$(echo "$OUTPUTS" | jq -r '.MCP_OAUTH_AUTHORIZE_URL.value // empty')
APIM_OAUTH_TOKEN_URL=$(echo "$OUTPUTS" | jq -r '.MCP_OAUTH_TOKEN_URL.value // empty')
MCP_CLIENT_ID=$(echo "$OUTPUTS" | jq -r '.MCP_CLIENT_ID.value // empty')
AZURE_TENANT_ID=$(echo "$OUTPUTS" | jq -r '.AZURE_TENANT_ID.value // empty')
AZURE_SUBSCRIPTION_ID=$(echo "$OUTPUTS" | jq -r '.AZURE_SUBSCRIPTION_ID.value // empty')
AZURE_RESOURCE_GROUP_NAME=$(echo "$OUTPUTS" | jq -r '.AZURE_RESOURCE_GROUP_NAME.value // empty')
AZURE_LOCATION=$(echo "$OUTPUTS" | jq -r '.AZURE_LOCATION.value // empty')
AKS_CLUSTER_NAME=$(echo "$OUTPUTS" | jq -r '.AKS_CLUSTER_NAME.value // empty')
CONTAINER_REGISTRY=$(echo "$OUTPUTS" | jq -r '.CONTAINER_REGISTRY.value // empty')

REDIRECT_URI="http://localhost:8080/callback"

echo "✅ Configuration extracted:"
echo "   APIM_BASE_URL = ${APIM_BASE_URL:-<not set>}"
echo "   MCP_CLIENT_ID = ${MCP_CLIENT_ID:-<not set>}"
echo "   AZURE_RESOURCE_GROUP_NAME = ${AZURE_RESOURCE_GROUP_NAME:-<not set>}"
echo "   AKS_CLUSTER_NAME = ${AKS_CLUSTER_NAME:-<not set>}"

# Ensure output directory exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_PATH="$SCRIPT_DIR/../$OUTPUT_DIR"
mkdir -p "$OUTPUT_PATH"

# Generate .env file
ENV_FILE="$OUTPUT_PATH/mcp_test.env"
echo ""
echo "📝 Writing .env file: $ENV_FILE"

cat > "$ENV_FILE" <<EOF
# MCP Test Configuration
# Generated: $(date -u +"%Y-%m-%d %H:%M:%S")
# Deployment: $DEPLOYMENT_NAME

# APIM Endpoints
APIM_BASE_URL=$APIM_BASE_URL
APIM_GATEWAY_URL=$APIM_GATEWAY_URL
APIM_OAUTH_AUTHORIZE_URL=$APIM_OAUTH_AUTHORIZE_URL
APIM_OAUTH_TOKEN_URL=$APIM_OAUTH_TOKEN_URL

# OAuth Configuration
MCP_CLIENT_ID=$MCP_CLIENT_ID
REDIRECT_URI=$REDIRECT_URI

# Azure Resources
AZURE_TENANT_ID=$AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID=$AZURE_SUBSCRIPTION_ID
AZURE_RESOURCE_GROUP_NAME=$AZURE_RESOURCE_GROUP_NAME
AZURE_LOCATION=$AZURE_LOCATION
AKS_CLUSTER_NAME=$AKS_CLUSTER_NAME
CONTAINER_REGISTRY=$CONTAINER_REGISTRY
EOF

echo "✅ .env file created: $ENV_FILE"

# Generate .json file
JSON_FILE="$OUTPUT_PATH/mcp_test_config.json"
echo "📝 Writing JSON file: $JSON_FILE"

cat > "$JSON_FILE" <<EOF
{
  "generated_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "deployment_name": "$DEPLOYMENT_NAME",
  "apim": {
    "base_url": "$APIM_BASE_URL",
    "gateway_url": "$APIM_GATEWAY_URL",
    "oauth_authorize_url": "$APIM_OAUTH_AUTHORIZE_URL",
    "oauth_token_url": "$APIM_OAUTH_TOKEN_URL"
  },
  "oauth": {
    "client_id": "$MCP_CLIENT_ID",
    "redirect_uri": "$REDIRECT_URI",
    "scope": "openid profile email"
  },
  "azure": {
    "tenant_id": "$AZURE_TENANT_ID",
    "subscription_id": "$AZURE_SUBSCRIPTION_ID",
    "resource_group": "$AZURE_RESOURCE_GROUP_NAME",
    "location": "$AZURE_LOCATION"
  },
  "resources": {
    "aks_cluster_name": "$AKS_CLUSTER_NAME",
    "container_registry": "$CONTAINER_REGISTRY"
  }
}
EOF

echo "✅ JSON file created: $JSON_FILE"

echo ""
echo "🎉 Configuration files generated successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Review the generated configuration files"
echo "   2. Run test with: python tests/test_mcp_fixed_session.py"
echo "   3. Generate OAuth token with: --generate-token flag"
echo ""
