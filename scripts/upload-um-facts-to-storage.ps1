# =============================================================================
# Upload UM Facts to Azure Blob Storage
# =============================================================================
# This script uploads utilization management facts JSON files to the Azure Blob
# Storage container for use by the utilization-management-facts knowledge source.
#
# Usage:
#   ./scripts/upload-um-facts-to-storage.ps1
#
# Prerequisites:
#   - Azure CLI installed and logged in
#   - azd environment configured with storage account details
# =============================================================================

param(
    [string]$ContainerName = "um-facts",
    [string]$FactsPath = "facts/um"
)

Write-Host "=== Upload UM Facts to Azure Blob Storage ===" -ForegroundColor Cyan

# Get storage account URL from azd environment
$storageUrl = azd env get-value AZURE_STORAGE_ACCOUNT_URL 2>$null
if (-not $storageUrl) {
    Write-Host "Error: AZURE_STORAGE_ACCOUNT_URL not found in azd environment" -ForegroundColor Red
    Write-Host "Run 'azd provision' first to create the infrastructure" -ForegroundColor Yellow
    exit 1
}

# Extract storage account name from URL
$storageAccountName = ($storageUrl -replace "https://", "" -replace ".blob.core.windows.net/", "").Trim("/")
Write-Host "Storage Account: $storageAccountName" -ForegroundColor Green
Write-Host "Container: $ContainerName" -ForegroundColor Green

# Check if container exists, create if not
Write-Host "`nChecking container..." -ForegroundColor Yellow
$containerExists = az storage container exists `
    --account-name $storageAccountName `
    --name $ContainerName `
    --auth-mode login `
    --query exists -o tsv 2>$null

if ($containerExists -ne "true") {
    Write-Host "Creating container: $ContainerName" -ForegroundColor Yellow
    az storage container create `
        --account-name $storageAccountName `
        --name $ContainerName `
        --auth-mode login
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create container" -ForegroundColor Red
        exit 1
    }
    Write-Host "Container created successfully" -ForegroundColor Green
} else {
    Write-Host "Container already exists" -ForegroundColor Green
}

# Upload UM facts files
Write-Host "`nUploading UM facts files..." -ForegroundColor Yellow
$factsDir = Join-Path $PSScriptRoot ".." $FactsPath

if (-not (Test-Path $factsDir)) {
    Write-Host "Error: Facts directory not found at $factsDir" -ForegroundColor Red
    exit 1
}

$files = Get-ChildItem -Path $factsDir -Filter "*.json"
if ($files.Count -eq 0) {
    Write-Host "Warning: No JSON files found in $factsDir" -ForegroundColor Yellow
    exit 0
}

$uploaded = 0
foreach ($file in $files) {
    Write-Host "  Uploading: $($file.Name)" -ForegroundColor White
    az storage blob upload `
        --account-name $storageAccountName `
        --container-name $ContainerName `
        --file $file.FullName `
        --name $file.Name `
        --auth-mode login `
        --overwrite 2>$null
    if ($LASTEXITCODE -eq 0) {
        $uploaded++
        Write-Host "    Uploaded successfully" -ForegroundColor Green
    } else {
        Write-Host "    Error uploading $($file.Name)" -ForegroundColor Red
    }
}

Write-Host "`n=== Upload Complete ===" -ForegroundColor Cyan
Write-Host "Uploaded $uploaded of $($files.Count) files to $ContainerName" -ForegroundColor Green
Write-Host "Storage URL: $storageUrl$ContainerName" -ForegroundColor Green
