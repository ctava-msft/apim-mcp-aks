targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the the environment which is used to generate a short unique hash used in all resources.')
param environmentName string


@minLength(1)
@description('Primary location for all resources')
@allowed(['australiaeast', 'eastasia', 'eastus', 'eastus2', 'northeurope', 'southcentralus', 'southeastasia', 'swedencentral', 'uksouth', 'westus2', 'eastus2euap'])
@metadata({
  azd: {
    type: 'location'
  }
})
param location string
param vnetEnabled bool
param apiServiceName string = ''
param apiUserAssignedIdentityName string = ''
param applicationInsightsName string = ''
param logAnalyticsName string = ''
param resourceGroupName string = ''
param storageAccountName string = ''
param vNetName string = ''
param mcpEntraApplicationDisplayName string = ''
param mcpEntraApplicationUniqueName string = ''
param existingEntraAppId string = ''
param disableLocalAuth bool = true

// MCP Client APIM gateway specific variables

var oauth_scopes = 'openid https://graph.microsoft.com/.default'


var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))
var tags = { 'azd-env-name': environmentName }
var functionAppName = !empty(apiServiceName) ? apiServiceName : '${abbrs.webSitesFunctions}api-${resourceToken}'
var deploymentStorageContainerName = 'app-package-${take(functionAppName, 32)}-${take(toLower(uniqueString(functionAppName, resourceToken)), 7)}'
var serviceVirtualNetworkName = !empty(vNetName) ? vNetName : '${abbrs.networkVirtualNetworks}${resourceToken}'
var serviceVirtualNetworkAppSubnetName = 'app'
var serviceVirtualNetworkPrivateEndpointSubnetName = 'private-endpoints-subnet'


// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: !empty(resourceGroupName) ? resourceGroupName : '${abbrs.resourcesResourceGroups}${environmentName}'
  location: location
  tags: tags
}

var apimResourceToken = toLower(uniqueString(subscription().id, resourceGroupName, environmentName, location))
var apiManagementName = '${abbrs.apiManagementService}${apimResourceToken}'

// apim service deployment
module apimService './core/apim/apim.bicep' = {
  name: apiManagementName
  scope: rg
  params:{
    apiManagementName: apiManagementName
  }
}

// MCP client oauth via APIM gateway
module oauthAPIModule './app/apim-oauth/oauth.bicep' = {
  name: 'oauthAPIModule'
  scope: rg
  params: {
    location: location
    entraAppUniqueName: !empty(mcpEntraApplicationUniqueName) ? mcpEntraApplicationUniqueName : 'mcp-oauth-${abbrs.applications}${apimResourceToken}'
    entraAppDisplayName: !empty(mcpEntraApplicationDisplayName) ? mcpEntraApplicationDisplayName : 'MCP-OAuth-${abbrs.applications}${apimResourceToken}'
    apimServiceName: apimService.name
    oauthScopes: oauth_scopes
    entraAppUserAssignedIdentityPrincipleId: apimService.outputs.entraAppUserAssignedIdentityPrincipleId
    entraAppUserAssignedIdentityClientId: apimService.outputs.entraAppUserAssignedIdentityClientId
    existingEntraAppId: existingEntraAppId
  }
}

// MCP server API endpoints pointing to AKS service via public LoadBalancer
module mcpApiModule './app/apim-mcp/mcp-api.bicep' = {
  name: 'mcpApiModule'
  scope: rg
  params: {
    apimServiceName: apimService.name
    mcpServerBackendUrl: 'http://${mcpPublicIp.outputs.publicIpAddress}/runtime/webhooks/mcp'
  }
  dependsOn: [
    aksCluster
    oauthAPIModule
    mcpPublicIp
  ]
}


// User assigned managed identity for AKS cluster
module aksUserAssignedIdentity './core/identity/userAssignedIdentity.bicep' = {
  name: 'aksUserAssignedIdentity'
  scope: rg
  params: {
    location: location
    tags: tags
    identityName: !empty(apiUserAssignedIdentityName) ? apiUserAssignedIdentityName : '${abbrs.managedIdentityUserAssignedIdentities}aks-${resourceToken}'
  }
}

// User assigned managed identity for MCP server workload
module mcpUserAssignedIdentity './core/identity/userAssignedIdentity.bicep' = {
  name: 'mcpUserAssignedIdentity'
  scope: rg
  params: {
    location: location
    tags: tags
    identityName: '${abbrs.managedIdentityUserAssignedIdentities}mcp-${resourceToken}'
  }
}

// Virtual Network (created before AKS if vnetEnabled)
module serviceVirtualNetworkEarly 'app/vnet.bicep' = if (vnetEnabled) {
  name: 'serviceVirtualNetworkEarly'
  scope: rg
  params: {
    location: location
    tags: tags
    vNetName: serviceVirtualNetworkName
  }
}

// Azure Container Registry for Docker images
module containerRegistry './core/acr/container-registry.bicep' = {
  name: 'containerRegistry'
  scope: rg
  params: {
    containerRegistryName: '${abbrs.containerRegistryRegistries}${resourceToken}'
    location: location
    tags: tags
    sku: 'Standard'
  }
}

// AKS Cluster
module aksCluster './core/aks/aks-cluster.bicep' = {
  name: 'aksCluster'
  scope: rg
  params: {
    aksClusterName: '${abbrs.containerServiceManagedClusters}${resourceToken}'
    location: location
    tags: tags
    kubernetesVersion: '1.32.0'
    systemNodePoolVmSize: 'Standard_DS2_v2'
    systemNodePoolCount: 2
    userAssignedIdentityId: aksUserAssignedIdentity.outputs.identityId
    logAnalyticsWorkspaceId: monitoring.outputs.logAnalyticsWorkspaceId
    subnetId: vnetEnabled ? '${rg.id}/providers/Microsoft.Network/virtualNetworks/${serviceVirtualNetworkName}/subnets/${serviceVirtualNetworkAppSubnetName}' : ''
  }
  dependsOn: vnetEnabled ? [
    serviceVirtualNetworkEarly
  ] : []
}

// Static Public IP for MCP Server LoadBalancer
// This IP is used by the Kubernetes LoadBalancer service and APIM backend
module mcpPublicIp './core/network/public-ip.bicep' = {
  name: 'mcpPublicIp'
  scope: rg
  params: {
    publicIpName: 'pip-mcp-${resourceToken}'
    location: location
    tags: tags
    dnsLabel: 'mcp-${resourceToken}'
  }
}

// Grant AKS pull access to ACR
var acrPullRoleDefinitionId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
module acrPullRoleAssignment 'core/acr/acr-role-assignment.bicep' = {
  name: 'acrPullRoleAssignment'
  scope: rg
  params: {
    containerRegistryName: containerRegistry.outputs.containerRegistryName
    roleDefinitionID: acrPullRoleDefinitionId
    principalID: aksUserAssignedIdentity.outputs.identityPrincipalId
  }
  dependsOn: [
    aksCluster
  ]
}

// Backing storage for Azure functions api
module storage './core/storage/storage-account.bicep' = {
  name: 'storage'
  scope: rg
  params: {
    name: !empty(storageAccountName) ? storageAccountName : '${abbrs.storageStorageAccounts}${resourceToken}'
    location: location
    tags: tags
    containers: [{name: deploymentStorageContainerName}, {name: 'snippets'}]
    publicNetworkAccess: vnetEnabled ? 'Disabled' : 'Enabled'
    networkAcls: !vnetEnabled ? {} : {
      defaultAction: 'Deny'
    }
    // Shared key access is required for azd to upload the deployment package. The function runtime still uses managed identity.
    allowSharedKeyAccess: true
  }
}

var StorageBlobDataOwner = 'b7e6dc6d-f1e8-4753-8033-0f276bb0955b'
var StorageQueueDataContributor = '974c5e8b-45b9-4653-ba55-5f855dd0fb88'

// Allow access from MCP server workload identity to blob storage
module blobRoleAssignmentMcp 'app/storage-Access.bicep' = {
  name: 'blobRoleAssignmentMcp'
  scope: rg
  params: {
    storageAccountName: storage.outputs.name
    roleDefinitionID: StorageBlobDataOwner
    principalID: mcpUserAssignedIdentity.outputs.identityPrincipalId
  }
}

// Allow access from MCP server workload identity to queue storage
module queueRoleAssignmentMcp 'app/storage-Access.bicep' = {
  name: 'queueRoleAssignmentMcp'
  scope: rg
  params: {
    storageAccountName: storage.outputs.name
    roleDefinitionID: StorageQueueDataContributor
    principalID: mcpUserAssignedIdentity.outputs.identityPrincipalId
  }
}

// Virtual Network & private endpoint to blob storage
module serviceVirtualNetwork 'app/vnet.bicep' =  if (vnetEnabled) {
  name: 'serviceVirtualNetwork'
  scope: rg
  params: {
    location: location
    tags: tags
    vNetName: serviceVirtualNetworkName
  }
}

module storagePrivateEndpoint 'app/storage-PrivateEndpoint.bicep' = if (vnetEnabled) {
  name: 'servicePrivateEndpoint'
  scope: rg
  params: {
    location: location
    tags: tags
    virtualNetworkName: serviceVirtualNetworkName
    subnetName: vnetEnabled ? serviceVirtualNetworkPrivateEndpointSubnetName : ''
    resourceName: storage.outputs.name
  }
  dependsOn: [
    serviceVirtualNetwork
  ]
}

// Monitor application with Azure Monitor
module monitoring './core/monitor/monitoring.bicep' = {
  name: 'monitoring'
  scope: rg
  params: {
    location: location
    tags: tags
    logAnalyticsName: !empty(logAnalyticsName) ? logAnalyticsName : '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
    applicationInsightsName: !empty(applicationInsightsName) ? applicationInsightsName : '${abbrs.insightsComponents}${resourceToken}'
    disableLocalAuth: disableLocalAuth  
  }
}

var monitoringRoleDefinitionId = '3913510d-42f4-4e42-8a64-420c390055eb' // Monitoring Metrics Publisher role ID

// Allow access from MCP server workload identity to application insights
module appInsightsRoleAssignmentMcp './core/monitor/appinsights-access.bicep' = {
  name: 'appInsightsRoleAssignmentMcp'
  scope: rg
  params: {
    appInsightsName: monitoring.outputs.applicationInsightsName
    roleDefinitionID: monitoringRoleDefinitionId
    principalID: mcpUserAssignedIdentity.outputs.identityPrincipalId
  }
}



// App outputs
output APPLICATIONINSIGHTS_CONNECTION_STRING string = monitoring.outputs.applicationInsightsConnectionString
output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output AKS_CLUSTER_NAME string = aksCluster.outputs.aksClusterName
output CONTAINER_REGISTRY string = containerRegistry.outputs.containerRegistryLoginServer
output AZURE_STORAGE_ACCOUNT_URL string = storage.outputs.primaryEndpoints.blob
output MCP_SERVER_IDENTITY_CLIENT_ID string = mcpUserAssignedIdentity.outputs.identityClientId
output SERVICE_API_ENDPOINTS array = [ '${apimService.outputs.gatewayUrl}/mcp/sse' ]
output APIM_GATEWAY_URL string = apimService.outputs.gatewayUrl
output MCP_BASE_URL string = '${apimService.outputs.gatewayUrl}/mcp'
output MCP_OAUTH_AUTHORIZE_URL string = '${apimService.outputs.gatewayUrl}/mcp/oauth/authorize'
output MCP_OAUTH_TOKEN_URL string = '${apimService.outputs.gatewayUrl}/mcp/oauth/token'
output MCP_CLIENT_ID string = existingEntraAppId
output AZURE_RESOURCE_GROUP_NAME string = rg.name
output AZURE_SUBSCRIPTION_ID string = subscription().subscriptionId
output MCP_PUBLIC_IP_ADDRESS string = mcpPublicIp.outputs.publicIpAddress
output MCP_PUBLIC_IP_NAME string = 'pip-mcp-${resourceToken}'
output AKS_NODE_RESOURCE_GROUP string = 'MC_${rg.name}_${abbrs.containerServiceManagedClusters}${resourceToken}_${location}'

