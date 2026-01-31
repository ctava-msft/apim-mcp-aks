# Test Configuration Guide

This guide explains how to configure and run tests for the MCP (Model Context Protocol) server.

## Quick Reference: azd Commands

| Command | Purpose | Provisions Infrastructure | Deploys Code | Generates Test Config |
|---------|---------|---------------------------|--------------|----------------------|
| `azd up` | Full deployment | ✅ Yes | ✅ Yes | ✅ Yes (via postprovision hook) |
| `azd provision` | Infrastructure only | ✅ Yes | ❌ No | ✅ Yes (via postprovision hook) |
| `azd deploy` | Code only | ❌ No | ✅ Yes | ❌ No (requires existing config) |

**💡 Key Point**: You must run `azd provision` or `azd up` at least once to generate test configuration. `azd deploy` alone will not work for initial setup.

## Overview

The test configuration is automatically generated from your Azure deployment outputs. No hardcoded values are needed - all configuration is derived from the actual deployed infrastructure.

## Configuration Files

The test scripts use configuration from these files (in order of precedence):

1. **`tests/mcp_test_config.json`** - JSON configuration file
2. **`tests/mcp_test.env`** - Environment variables file
3. **Environment variables** - System environment variables

## Generating Configuration

### Automatic Generation (Recommended)

Configuration is automatically generated after infrastructure provisioning when using `azd`:

**First time deployment (provision + deploy):**
```bash
azd up
```

**Infrastructure changes only:**
```bash
azd provision
```

**Code changes only (infrastructure already exists):**
```bash
azd deploy
```

> **Note**: The test configuration requires infrastructure outputs from Bicep deployment. You must run `azd provision` or `azd up` at least once to generate the configuration. `azd deploy` alone will NOT generate configuration since it only deploys code without provisioning.

The postprovision hook will automatically call `generate-test-config.ps1` (or `.sh` on Unix) after infrastructure provisioning.

### Manual Generation

If you need to regenerate the configuration:

**Windows (PowerShell):**
```powershell
.\scripts\generate-test-config.ps1
```

**Linux/Mac (Bash):**
```bash
./scripts/generate-test-config.sh
```

**With specific deployment:**
```powershell
.\scripts\generate-test-config.ps1 -DeploymentName "apim-mcp-20260130"
```

## Configuration Contents

The generated configuration includes:

- **APIM Endpoints**: Base URL, OAuth authorize/token URLs
- **OAuth Settings**: Client ID, redirect URI, scopes
- **Azure Resources**: Tenant ID, subscription, resource group, location
- **Infrastructure**: AKS cluster name, container registry

### Example `mcp_test_config.json`:

```json
{
  "generated_at": "2026-01-30T22:15:04Z",
  "deployment_name": "apim-mcp-20260130",
  "apim": {
    "base_url": "https://apim-xxx.azure-api.net/mcp",
    "gateway_url": "https://apim-xxx.azure-api.net",
    "oauth_authorize_url": "https://apim-xxx.azure-api.net/mcp/oauth/authorize",
    "oauth_token_url": "https://apim-xxx.azure-api.net/mcp/oauth/token"
  },
  "oauth": {
    "client_id": "6441e54f-8149-487b-aac4-3a55a049a362",
    "redirect_uri": "http://localhost:8080/callback",
    "scope": "openid profile email"
  },
  "azure": {
    "tenant_id": "...",
    "subscription_id": "...",
    "resource_group": "rg-apim-mcp-aks-kaito",
    "location": "eastus"
  },
  "resources": {
    "aks_cluster_name": "aks-xxx",
    "container_registry": "crxxx.azurecr.io"
  }
}
```

## Running Tests

### 1. Generate Configuration

```powershell
.\scripts\generate-test-config.ps1
```

### 2. Generate OAuth Token (First Time)

```bash
cd tests
python test_mcp_fixed_session.py --generate-token
```

This will:
1. Open your browser to the OAuth authorization page
2. Prompt you to complete the login
3. Ask you to paste the authorization code
4. Exchange the code for an access token
5. Save the token to `tests/mcp_tokens.json`

### 3. Run Tests

```bash
python tests/test_mcp_fixed_session.py
```

The test will:
- Load configuration from `mcp_test_config.json`
- Load OAuth token from `mcp_tokens.json`
- Establish SSE session with APIM
- Discover and test MCP tools

## Command-Line Options

```bash
# Show help
python test_mcp_fixed_session.py --help

# Generate OAuth token interactively
python test_mcp_fixed_session.py --generate-token

# Run full test (default)
python test_mcp_fixed_session.py
```

## Troubleshooting

### Configuration Not Found

**Problem**: `❌ No configuration found!`

**Solution**: 
```powershell
# Regenerate configuration (requires infrastructure to be provisioned)
.\scripts\generate-test-config.ps1

# Or re-provision infrastructure
azd provision

# Check if deployment succeeded
az deployment sub show --name main --query "properties.provisioningState"
```

### Invalid OAuth Token

**Problem**: `❌ Could not load access token`

**Solution**:
```bash
# Delete old token and regenerate
rm tests/mcp_tokens.json
python tests/test_mcp_fixed_session.py --generate-token
```

### Deployment Outputs Missing

**Problem**: Configuration values are `null`

**Solution**:
```bash
# Ensure the Bicep deployment completed successfully
az deployment sub show --name <deployment-name> --query "properties.provisioningState"

# Check for deployment errors
az deployment sub show --name <deployment-name> --query "properties.error"

# Re-provision infrastructure if needed
azd provision  # or: azd up
```

### When to Use Each Command

**`azd up`**: First time setup or when you want to provision + deploy in one command
- Provisions infrastructure (runs Bicep)
- Generates test configuration (postprovision hook)
- Deploys application code

**`azd provision`**: Infrastructure changes or initial setup without deploying code
- Provisions infrastructure (runs Bicep)
- Generates test configuration (postprovision hook)
- Does NOT deploy application code

**`azd deploy`**: Application code changes only
- Deploys application code to existing infrastructure
- Does NOT provision infrastructure
- Does NOT generate test configuration
- Requires infrastructure to already exist from previous `azd provision` or `azd up`

## Environment Variables

You can also set configuration via environment variables:

```bash
export APIM_BASE_URL="https://apim-xxx.azure-api.net/mcp"
export APIM_OAUTH_AUTHORIZE_URL="https://apim-xxx.azure-api.net/mcp/oauth/authorize"
export APIM_OAUTH_TOKEN_URL="https://apim-xxx.azure-api.net/mcp/oauth/token"
export MCP_CLIENT_ID="6441e54f-8149-487b-aac4-3a55a049a362"
export REDIRECT_URI="http://localhost:8080/callback"
```

Then run tests normally:
```bash
python tests/test_mcp_fixed_session.py
```

## CI/CD Integration

For automated testing in CI/CD pipelines:

1. Provision infrastructure (once per environment)
2. Generate configuration files
3. Store OAuth tokens securely (e.g., Azure Key Vault, GitHub Secrets)
4. Run tests with configuration

Example GitHub Actions:
```yaml
- name: Provision Infrastructure
  run: azd provision --no-prompt
  
- name: Generate Test Config  
  run: ./scripts/generate-test-config.sh

- name: Run Tests
  env:
    MCP_ACCESS_TOKEN: ${{ secrets.MCP_ACCESS_TOKEN }}
  run: |
    echo '{"access_token":"$MCP_ACCESS_TOKEN"}' > tests/mcp_tokens.json
    python tests/test_mcp_fixed_session.py
```

**For code-only deployments in CI/CD:**
```yaml
# After initial infrastructure is provisioned
- name: Deploy Code Only
  run: azd deploy --no-prompt

- name: Run Tests (using existing config)
  env:
    MCP_ACCESS_TOKEN: ${{ secrets.MCP_ACCESS_TOKEN }}
  run: |
    echo '{"access_token":"$MCP_ACCESS_TOKEN"}' > tests/mcp_tokens.json
    python tests/test_mcp_fixed_session.py
```
