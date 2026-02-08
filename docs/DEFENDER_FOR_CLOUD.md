# Microsoft Defender for Cloud - Deployment Testing Guide

## Overview
This guide provides instructions for testing the Microsoft Defender for Cloud Bicep implementation.

## Prerequisites
- Azure CLI installed and authenticated (`az login`)
- Appropriate permissions to deploy Defender for Cloud at subscription level
- Valid email address for security contact notifications

## Bicep Validation Tests

### 1. Syntax Validation
All Bicep files have been validated using Azure Bicep CLI:

```bash
# Validate main template
az bicep build --file infra/main.bicep

# Validate defender module
az bicep build --file infra/core/security/defender.bicep

# Lint check
az bicep lint --file infra/main.bicep
az bicep lint --file infra/core/security/defender.bicep
```

**Result**: All files compile successfully with no errors.

### 2. Template Structure Validation
The defender module includes:
- ✅ Microsoft.Security/pricings resources for 6 Defender plans
- ✅ Microsoft.Security/securityContacts resource
- ✅ Microsoft.Security/autoProvisioningSettings resource
- ✅ Conditional deployment logic
- ✅ Output variables for deployment validation

## Deployment Testing

### Test Scenario 1: Enable Defender with All Plans
```bash
# Set environment variables
azd env set DEFENDER_ENABLED true
azd env set DEFENDER_SECURITY_CONTACT_EMAIL "security@example.com"
azd env set DEFENDER_SECURITY_CONTACT_PHONE "+1-555-0100"

# Deploy or update infrastructure
azd provision
```

**Expected Results:**
- Defender module deploys successfully
- All 6 Defender plans enabled at subscription level
- Security contact configured
- Auto-provisioning enabled
- Output variables show enabled status

### Test Scenario 2: Defender Disabled (Default Behavior)
```bash
# Don't set security contact email (required)
azd env set DEFENDER_ENABLED true
# DEFENDER_SECURITY_CONTACT_EMAIL not set

# Deploy infrastructure
azd provision
```

**Expected Results:**
- Defender module is NOT deployed (conditional deployment)
- No Defender plans enabled
- No security contact configured
- No errors or warnings

### Test Scenario 3: Disable Defender Explicitly
```bash
# Explicitly disable Defender
azd env set DEFENDER_ENABLED false

# Deploy infrastructure
azd provision
```

**Expected Results:**
- Defender module is NOT deployed
- No Defender plans enabled
- Existing infrastructure unaffected

### Test Scenario 4: Selective Plan Enablement
Individual Defender plans can be disabled by modifying parameters in `infra/main.parameters.json`:

```json
{
  "defenderForContainersEnabled": {
    "value": "${DEFENDER_FOR_CONTAINERS_ENABLED=false}"
  }
}
```

## Verification Steps

### 1. Azure Portal Verification
1. Navigate to **Microsoft Defender for Cloud** in Azure Portal
2. Go to **Environment Settings** → Select your subscription
3. Verify the following plans show as **On**:
   - Containers
   - Key Vaults
   - Azure Cosmos DB
   - API Management
   - Resource Manager
   - Container Registry (legacy)

4. Navigate to **Environment Settings** → **Email notifications**
5. Verify security contact is configured with:
   - Email address
   - Phone number (if provided)
   - Alert notifications: On
   - Notification by role: On (Owner)

### 2. Azure CLI Verification
```bash
# Check Defender pricing tiers
az security pricing list --subscription <subscription-id>

# Check security contact
az security contact list --subscription <subscription-id>

# Check auto-provisioning settings
az security auto-provisioning-setting list --subscription <subscription-id>
```

### 3. Output Variables Verification
After deployment, check the output variables:

```bash
# List all outputs
azd env get-values | grep DEFENDER

# Expected outputs:
# DEFENDER_ENABLED=true
# DEFENDER_DEPLOYED=true
# DEFENDER_FOR_CONTAINERS_ENABLED=true
# DEFENDER_FOR_KEY_VAULT_ENABLED=true
# DEFENDER_FOR_COSMOS_DB_ENABLED=true
# DEFENDER_FOR_APIS_ENABLED=true
# DEFENDER_FOR_RESOURCE_MANAGER_ENABLED=true
```

## Rollback Procedure

If you need to disable Defender for Cloud after deployment:

```bash
# Disable Defender
azd env set DEFENDER_ENABLED false

# Re-deploy infrastructure
azd provision
```

**Note:** This will not automatically remove existing Defender plans. You'll need to manually disable them in the Azure Portal or use Azure CLI:

```bash
az security pricing update --name <plan-name> --tier Free --subscription <subscription-id>
```

## Cost Considerations

Microsoft Defender for Cloud pricing varies by plan and resource consumption. Before enabling:

1. Review [Microsoft Defender for Cloud pricing](https://azure.microsoft.com/pricing/details/defender-for-cloud/)
2. Estimate costs based on your resource count
3. Consider starting with a subset of plans for testing

**Estimated Monthly Costs (as of documentation date):**
- Defender for Containers: ~$7 per vCore
- Defender for Key Vault: ~$0.02 per 10,000 operations
- Defender for Cosmos DB: ~$0.0012 per 100 RU/s
- Defender for APIs: ~$2.80 per million API calls
- Defender for Resource Manager: ~$5 per subscription

## Troubleshooting

### Issue: Defender module not deploying
**Cause:** `DEFENDER_SECURITY_CONTACT_EMAIL` not set  
**Solution:** Set the required environment variable before deployment

### Issue: Deployment permission error
**Cause:** Insufficient permissions at subscription level  
**Solution:** Ensure deploying user has Owner or Security Admin role at subscription level

### Issue: Plan already enabled warning
**Cause:** Defender plan previously enabled  
**Solution:** This is expected behavior; deployment will update existing configuration

## Security Summary

✅ **No security vulnerabilities introduced**
- All changes are infrastructure-as-code (Bicep)
- No application code changes
- CodeQL analysis: N/A (Bicep files not analyzed)
- Bicep linter: No errors or warnings
- Microsoft Defender for Cloud enhances overall security posture

## Additional Resources

- [Microsoft Defender for Cloud Documentation](https://learn.microsoft.com/azure/defender-for-cloud/)
- [Bicep Security Resources](https://learn.microsoft.com/azure/azure-resource-manager/bicep/scenarios-security)
- [Defender for Cloud Best Practices](https://learn.microsoft.com/azure/defender-for-cloud/recommendations-reference)
