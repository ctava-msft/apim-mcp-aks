param principalID string
param roleDefinitionID string
param containerRegistryName string

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: containerRegistryName
}

// Allow access from AKS to Container Registry using a managed identity
resource acrRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistry.id, principalID, roleDefinitionID)
  scope: containerRegistry
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', roleDefinitionID)
    principalId: principalID
    principalType: 'ServicePrincipal'
  }
}

output roleAssignmentName string = acrRoleAssignment.name
