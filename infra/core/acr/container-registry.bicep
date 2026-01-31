@description('Name of the Azure Container Registry')
param containerRegistryName string

@description('Location for the container registry')
param location string = resourceGroup().location

@description('Tags for the container registry')
param tags object = {}

@description('SKU for the container registry')
@allowed(['Basic', 'Standard', 'Premium'])
param sku string = 'Standard'

@description('Enable admin user')
param adminUserEnabled bool = false

// Azure Container Registry
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  tags: tags
  sku: {
    name: sku
  }
  properties: {
    adminUserEnabled: adminUserEnabled
    publicNetworkAccess: 'Enabled'
    networkRuleBypassOptions: 'AzureServices'
  }
}

// Output container registry details
output containerRegistryId string = containerRegistry.id
output containerRegistryName string = containerRegistry.name
output containerRegistryLoginServer string = containerRegistry.properties.loginServer
