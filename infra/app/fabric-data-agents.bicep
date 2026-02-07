// =========================================
// Microsoft Fabric Data Agents Module
// =========================================
// This module configures Fabric Data Agents for enterprise AI agents to interact
// with Microsoft Fabric's data platform capabilities including:
// - Lakehouses: Query/write Spark SQL against Fabric Lakehouses
// - Warehouses: Execute T-SQL queries against Fabric Data Warehouses
// - Pipelines: Trigger, monitor, and manage Fabric Data Pipelines
// - Semantic Models: Query Power BI semantic models via DAX/MDX
//
// Security:
// - Workload identity federation with Fabric workspace
// - Least-privilege RBAC (Reader, Contributor roles as needed)
// - Data classification and sensitivity labels support
// - Audit logging for all Fabric operations

@description('Principal ID of the agent identity')
param agentPrincipalId string

@description('Resource ID of the Fabric capacity')
param fabricCapacityId string

@description('Name of the Fabric capacity')
param fabricCapacityName string

@description('Fabric workspace ID for data agents')
param fabricWorkspaceId string = ''

@description('Location for Fabric resources')
param location string = resourceGroup().location

@description('Tags for all resources')
param tags object = {}

@description('Enable Fabric Data Agents (default: true when Fabric is enabled)')
param fabricDataAgentsEnabled bool = false

@description('Unique suffix for role assignment names')
param deploymentSuffix string = 'fabric-data-v1'

// =========================================
// Built-in Role Definition IDs for Fabric
// =========================================

// Note: Microsoft Fabric uses Azure RBAC for workspace access
// These are standard Azure RBAC roles that apply to Fabric resources

// Reader - View Fabric workspace and resources
var FabricReader = 'acdd72a7-3385-48ef-bd42-f606fba81ae7'

// Contributor - Manage Fabric resources (lakehouses, warehouses, pipelines)
var FabricContributor = 'b24988ac-6180-42a0-ab88-20f7382dd24c'

// Storage Blob Data Reader - Read OneLake data
var StorageBlobDataReader = '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1'

// Storage Blob Data Contributor - Read/write OneLake data
var StorageBlobDataContributor = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'

// =========================================
// Fabric Capacity Reference
// =========================================

resource fabricCapacity 'Microsoft.Fabric/capacities@2023-11-01' existing = {
  name: fabricCapacityName
}

// =========================================
// RBAC Role Assignments for Fabric Data Agents
// =========================================

// Fabric Reader - Allows agents to view workspace and resources
resource fabricReaderRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (fabricDataAgentsEnabled) {
  name: guid(fabricCapacity.id, agentPrincipalId, FabricReader, deploymentSuffix)
  scope: fabricCapacity
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', FabricReader)
    principalId: agentPrincipalId
    principalType: 'ServicePrincipal'
    description: 'Allows Fabric Data Agents to view workspace and resources'
  }
}

// Fabric Contributor - Allows agents to manage Fabric resources
// This is required for:
// - Triggering data pipelines
// - Creating/modifying lakehouse tables
// - Executing warehouse queries with write operations
resource fabricContributorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (fabricDataAgentsEnabled) {
  name: guid(fabricCapacity.id, agentPrincipalId, FabricContributor, deploymentSuffix)
  scope: fabricCapacity
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', FabricContributor)
    principalId: agentPrincipalId
    principalType: 'ServicePrincipal'
    description: 'Allows Fabric Data Agents to manage lakehouses, warehouses, and pipelines'
  }
}

// Storage Blob Data Contributor - OneLake access for data operations
// OneLake is built on Azure Data Lake Storage Gen2
resource onelakeDataContributorRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (fabricDataAgentsEnabled) {
  name: guid(fabricCapacity.id, agentPrincipalId, StorageBlobDataContributor, deploymentSuffix, 'onelake')
  scope: fabricCapacity
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', StorageBlobDataContributor)
    principalId: agentPrincipalId
    principalType: 'ServicePrincipal'
    description: 'Allows Fabric Data Agents to read/write data through OneLake'
  }
}

// =========================================
// Outputs
// =========================================

output fabricReaderRoleAssignmentId string = fabricDataAgentsEnabled ? fabricReaderRoleAssignment.id : ''
output fabricContributorRoleAssignmentId string = fabricDataAgentsEnabled ? fabricContributorRoleAssignment.id : ''
output onelakeDataContributorRoleAssignmentId string = fabricDataAgentsEnabled ? onelakeDataContributorRoleAssignment.id : ''
output fabricWorkspaceId string = fabricWorkspaceId
