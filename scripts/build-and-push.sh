#!/bin/bash
# Build and push MCP server Docker image to Azure Container Registry

set -e

# Check required environment variables
if [ -z "$CONTAINER_REGISTRY" ]; then
    echo "❌ CONTAINER_REGISTRY environment variable is not set"
    exit 1
fi

# Set defaults
IMAGE_NAME="${IMAGE_NAME:-mcp-server}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE_NAME="${CONTAINER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "🏗️  Building Docker image: ${FULL_IMAGE_NAME}"

# Build the Docker image
cd src
docker build -t "${FULL_IMAGE_NAME}" .

echo "✅ Docker image built successfully"

# Login to Azure Container Registry
echo "🔐 Logging in to Azure Container Registry..."
az acr login --name "${CONTAINER_REGISTRY%%.*}"

# Push the image
echo "📤 Pushing Docker image to registry..."
docker push "${FULL_IMAGE_NAME}"

echo "✅ Docker image pushed successfully: ${FULL_IMAGE_NAME}"
