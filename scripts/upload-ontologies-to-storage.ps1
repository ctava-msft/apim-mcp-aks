# =============================================================================
# Upload Ontology Files to Azure Blob Storage
# =============================================================================
# This script uploads ontology JSON files to the Azure Blob Storage container
# for use by the FactsMemory provider when Fabric is disabled.
#
# Usage:
#   ./scripts/upload-ontologies-to-storage.ps1
#
# Prerequisites:
#   - Azure CLI installed and logged in
#   - azd environment configured with storage account details
# =============================================================================

param(
    [string]$ContainerName = "ontologies",
    [string]$OntologyPath = "facts/ontology"
)

Write-Host "=== Upload Ontologies to Azure Blob Storage ===" -ForegroundColor Cyan

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

# Upload ontology files
Write-Host "`nUploading ontology files..." -ForegroundColor Yellow
$ontologyDir = Join-Path $PSScriptRoot ".." $OntologyPath
if (-not (Test-Path $ontologyDir)) {
    Write-Host "Error: Ontology directory not found: $ontologyDir" -ForegroundColor Red
    exit 1
}

$jsonFiles = Get-ChildItem -Path $ontologyDir -Filter "*.json"
if ($jsonFiles.Count -eq 0) {
    Write-Host "No JSON files found in $ontologyDir" -ForegroundColor Yellow
    exit 0
}

$uploadedCount = 0
foreach ($file in $jsonFiles) {
    Write-Host "  Uploading: $($file.Name)" -ForegroundColor Cyan
    az storage blob upload `
        --account-name $storageAccountName `
        --container-name $ContainerName `
        --file $file.FullName `
        --name $file.Name `
        --overwrite `
        --auth-mode login 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "    ✓ Uploaded successfully" -ForegroundColor Green
        $uploadedCount++
    } else {
        Write-Host "    ✗ Failed to upload" -ForegroundColor Red
    }
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "Uploaded $uploadedCount of $($jsonFiles.Count) ontology files" -ForegroundColor Green
Write-Host "Storage URL: $storageUrl$ContainerName" -ForegroundColor Green

Write-Host "`nTo use these ontologies in FactsMemory:" -ForegroundColor Yellow
Write-Host "  facts_memory = FactsMemory(" -ForegroundColor White
Write-Host "      storage_account_url='$storageUrl'," -ForegroundColor White
Write-Host "      ontology_container='$ContainerName'," -ForegroundColor White
Write-Host "      fabric_enabled=False" -ForegroundColor White
Write-Host "  )" -ForegroundColor White
Write-Host "  await facts_memory.load_all_ontologies()" -ForegroundColor White
