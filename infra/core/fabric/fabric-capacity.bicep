// =========================================
// Microsoft Fabric Capacity
// =========================================
// This module deploys a Microsoft Fabric capacity that provides
// compute resources for Fabric workspaces, including Fabric IQ
// for ontology-grounded facts.

@description('Name of the Fabric capacity')
param name string

@description('Location for the Fabric capacity')
param location string = resourceGroup().location

@description('Tags for the Fabric capacity')
param tags object = {}

@description('SKU name for the Fabric capacity (F2, F4, F8, F16, F32, F64, F128, F256, F512, F1024, F2048)')
@allowed([
  'F2'
  'F4'
  'F8'
  'F16'
  'F32'
  'F64'
  'F128'
  'F256'
  'F512'
  'F1024'
  'F2048'
])
param skuName string = 'F2'

@description('List of admin members (email addresses) for the Fabric capacity')
param adminMembers array

// Fabric Capacity resource
resource fabricCapacity 'Microsoft.Fabric/capacities@2023-11-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: skuName
    tier: 'Fabric'
  }
  properties: {
    administration: {
      members: adminMembers
    }
  }
}

// Outputs
output id string = fabricCapacity.id
output name string = fabricCapacity.name
output state string = fabricCapacity.properties.state
