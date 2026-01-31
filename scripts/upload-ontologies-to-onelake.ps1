# =========================================
# Upload Ontologies to OneLake
# =========================================
# This script uploads ontology JSON files to OneLake for Fabric IQ integration.
# It uses Azure CLI and azcopy for authentication and upload.
#
# Prerequisites:
# - Azure CLI installed and logged in
# - azcopy installed (or uses Azure CLI storage commands)
# - Fabric workspace created with OneLake enabled
#
# Usage:
#   ./upload-ontologies-to-onelake.ps1 -WorkspaceId <guid> -LakehouseName <name>
#
# =========================================

param(
    [Parameter(Mandatory=$true)]
    [string]$WorkspaceId,
    
    [Parameter(Mandatory=$true)]
    [string]$LakehouseName,
    
    [Parameter(Mandatory=$false)]
    [string]$OneLakeEndpoint = "https://onelake.dfs.fabric.microsoft.com",
    
    [Parameter(Mandatory=$false)]
    [string]$OntologySourcePath = "../facts/ontology"
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "OneLake Ontology Upload Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Validate Azure CLI login
Write-Host "Checking Azure CLI login status..." -ForegroundColor Yellow
try {
    $account = az account show 2>&1 | ConvertFrom-Json
    Write-Host "Logged in as: $($account.user.name)" -ForegroundColor Green
    Write-Host "Subscription: $($account.name)" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Not logged into Azure CLI. Please run 'az login' first." -ForegroundColor Red
    exit 1
}

# Validate source path
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$fullSourcePath = Join-Path $scriptDir $OntologySourcePath
if (-not (Test-Path $fullSourcePath)) {
    Write-Host "ERROR: Ontology source path not found: $fullSourcePath" -ForegroundColor Red
    exit 1
}

Write-Host "Source path: $fullSourcePath" -ForegroundColor Gray
Write-Host ""

# Get access token for OneLake
Write-Host "Getting Azure access token for OneLake..." -ForegroundColor Yellow
$accessToken = az account get-access-token --resource "https://storage.azure.com/" --query accessToken -o tsv 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to get access token: $accessToken" -ForegroundColor Red
    exit 1
}
Write-Host "Access token obtained successfully" -ForegroundColor Green

# Construct OneLake paths
# OneLake path format: https://onelake.dfs.fabric.microsoft.com/{workspaceId}/{lakehouseName}/Files/ontology
$workspaceIdNoDashes = $WorkspaceId -replace "-", ""
$firstTwoChars = $workspaceIdNoDashes.Substring(0, 2).ToLower()

# Use workspace-specific FQDN for private endpoint support
$workspaceFqdn = "${workspaceIdNoDashes}.z${firstTwoChars}.dfs.fabric.microsoft.com"
$oneLakePath = "https://${workspaceFqdn}/${WorkspaceId}/${LakehouseName}.Lakehouse/Files/ontology"

Write-Host ""
Write-Host "OneLake Configuration:" -ForegroundColor Cyan
Write-Host "  Workspace ID: $WorkspaceId" -ForegroundColor Gray
Write-Host "  Lakehouse Name: $LakehouseName" -ForegroundColor Gray
Write-Host "  OneLake Path: $oneLakePath" -ForegroundColor Gray
Write-Host ""

# Get list of ontology files
$ontologyFiles = Get-ChildItem -Path $fullSourcePath -Filter "*.json"
if ($ontologyFiles.Count -eq 0) {
    Write-Host "WARNING: No JSON files found in $fullSourcePath" -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($ontologyFiles.Count) ontology file(s) to upload:" -ForegroundColor Yellow
foreach ($file in $ontologyFiles) {
    Write-Host "  - $($file.Name)" -ForegroundColor Gray
}
Write-Host ""

# Upload each ontology file
$successCount = 0
$failCount = 0

foreach ($file in $ontologyFiles) {
    $fileName = $file.Name
    $filePath = $file.FullName
    $destinationUrl = "$oneLakePath/$fileName"
    
    Write-Host "Uploading: $fileName" -ForegroundColor Yellow
    
    try {
        # Use Invoke-RestMethod to upload file
        $fileContent = Get-Content -Path $filePath -Raw
        $headers = @{
            "Authorization" = "Bearer $accessToken"
            "x-ms-version" = "2023-11-03"
            "Content-Type" = "application/json"
        }
        
        # Create the file (PUT request to create blob)
        $createUrl = "${destinationUrl}?resource=file"
        Invoke-RestMethod -Uri $createUrl -Method Put -Headers $headers -Body "" -ContentType "application/json" 2>&1 | Out-Null
        
        # Append content (PATCH request)
        $appendUrl = "${destinationUrl}?action=append&position=0"
        Invoke-RestMethod -Uri $appendUrl -Method Patch -Headers $headers -Body $fileContent -ContentType "application/json" 2>&1 | Out-Null
        
        # Flush (complete the write)
        $contentLength = [System.Text.Encoding]::UTF8.GetByteCount($fileContent)
        $flushUrl = "${destinationUrl}?action=flush&position=${contentLength}"
        Invoke-RestMethod -Uri $flushUrl -Method Patch -Headers $headers 2>&1 | Out-Null
        
        Write-Host "  SUCCESS: $fileName uploaded to OneLake" -ForegroundColor Green
        $successCount++
    } catch {
        Write-Host "  FAILED: $fileName - $($_.Exception.Message)" -ForegroundColor Red
        $failCount++
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Upload Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Successful: $successCount" -ForegroundColor Green
Write-Host "  Failed: $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Gray" })
Write-Host ""

if ($failCount -gt 0) {
    Write-Host "Some files failed to upload. Check the errors above." -ForegroundColor Yellow
    exit 1
}

Write-Host "All ontology files uploaded successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Navigate to your Fabric workspace in the Fabric portal" -ForegroundColor Gray
Write-Host "2. Open the Lakehouse: $LakehouseName" -ForegroundColor Gray
Write-Host "3. Verify the ontology files in Files/ontology/" -ForegroundColor Gray
Write-Host "4. Configure Fabric IQ to use these ontologies" -ForegroundColor Gray
Write-Host ""
