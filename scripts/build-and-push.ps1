# Build and push MCP server Docker image to Azure Container Registry
# PowerShell version

$ErrorActionPreference = "Stop"

# Check required environment variables
if (-not $env:CONTAINER_REGISTRY) {
    Write-Host "❌ CONTAINER_REGISTRY environment variable is not set" -ForegroundColor Red
    exit 1
}

# Set defaults
$IMAGE_NAME = if ($env:IMAGE_NAME) { $env:IMAGE_NAME } else { "mcp-server" }
$IMAGE_TAG = if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { "latest" }
$FULL_IMAGE_NAME = "$($env:CONTAINER_REGISTRY)/$($IMAGE_NAME):$($IMAGE_TAG)"

Write-Host "🏗️  Building Docker image: $FULL_IMAGE_NAME" -ForegroundColor Cyan

# Build the Docker image
Push-Location src
docker build -t $FULL_IMAGE_NAME .
Pop-Location

Write-Host "✅ Docker image built successfully" -ForegroundColor Green

# Login to Azure Container Registry
Write-Host "🔐 Logging in to Azure Container Registry..." -ForegroundColor Yellow
$registryName = $env:CONTAINER_REGISTRY -replace '\..*', ''
az acr login --name $registryName

# Push the image
Write-Host "📤 Pushing Docker image to registry..." -ForegroundColor Yellow
docker push $FULL_IMAGE_NAME

Write-Host "✅ Docker image pushed successfully: $FULL_IMAGE_NAME" -ForegroundColor Green
