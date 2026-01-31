// =========================================
// Microsoft Fabric Private Endpoint
// =========================================
// This module creates private endpoints for Microsoft Fabric
// to enable secure connectivity from AKS to Fabric/OneLake
// over the Microsoft backbone network.
//
// DNS Zones required for Fabric Private Link:
// - privatelink.dfs.fabric.microsoft.com (OneLake DFS API)
// - privatelink.blob.fabric.microsoft.com (OneLake Blob API)
// - privatelink.api.fabric.microsoft.com (Fabric API)

@description('Specifies the name of the virtual network.')
param virtualNetworkName string

@description('Specifies the name of the subnet which contains the private endpoints.')
param subnetName string

@description('Specifies the resource ID of the Fabric capacity.')
param fabricCapacityId string

@description('Specifies the name of the Fabric capacity (for naming conventions).')
param fabricCapacityName string

@description('Specifies the location.')
param location string = resourceGroup().location

@description('Tags for all resources.')
param tags object = {}

// Virtual Network reference
resource vnet 'Microsoft.Network/virtualNetworks@2021-08-01' existing = {
  name: virtualNetworkName
}

// DNS Zone names for Fabric private endpoints
var fabricDfsPrivateDNSZoneName = 'privatelink.dfs.fabric.microsoft.com'
var fabricBlobPrivateDNSZoneName = 'privatelink.blob.fabric.microsoft.com'
var fabricApiPrivateDNSZoneName = 'privatelink.api.fabric.microsoft.com'
// Use environment() for cloud-agnostic storage suffix
var onelakeDfsPrivateDNSZoneName = 'privatelink.dfs.${environment().suffixes.storage}'

// DNS Zone link naming
var fabricDfsDnsZoneLinkName = format('{0}-fabric-dfs-link-{1}', fabricCapacityName, take(toLower(uniqueString(fabricCapacityName, virtualNetworkName)), 4))
var fabricBlobDnsZoneLinkName = format('{0}-fabric-blob-link-{1}', fabricCapacityName, take(toLower(uniqueString(fabricCapacityName, virtualNetworkName)), 4))
var fabricApiDnsZoneLinkName = format('{0}-fabric-api-link-{1}', fabricCapacityName, take(toLower(uniqueString(fabricCapacityName, virtualNetworkName)), 4))
var onelakeDfsDnsZoneLinkName = format('{0}-onelake-dfs-link-{1}', fabricCapacityName, take(toLower(uniqueString(fabricCapacityName, virtualNetworkName)), 4))

// =========================================
// Private DNS Zones for Fabric
// =========================================

// Private DNS Zone for Fabric DFS (Data File System)
resource fabricDfsPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: fabricDfsPrivateDNSZoneName
  location: 'global'
  tags: tags
  properties: {}
  dependsOn: [
    vnet
  ]
}

// Private DNS Zone for Fabric Blob
resource fabricBlobPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: fabricBlobPrivateDNSZoneName
  location: 'global'
  tags: tags
  properties: {}
  dependsOn: [
    vnet
  ]
}

// Private DNS Zone for Fabric API
resource fabricApiPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: fabricApiPrivateDNSZoneName
  location: 'global'
  tags: tags
  properties: {}
  dependsOn: [
    vnet
  ]
}

// Private DNS Zone for OneLake DFS (core.windows.net)
resource onelakeDfsPrivateDnsZone 'Microsoft.Network/privateDnsZones@2020-06-01' = {
  name: onelakeDfsPrivateDNSZoneName
  location: 'global'
  tags: tags
  properties: {}
  dependsOn: [
    vnet
  ]
}

// =========================================
// Virtual Network Links
// =========================================

// VNet link for Fabric DFS DNS Zone
resource fabricDfsPrivateDnsZoneVirtualNetworkLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: fabricDfsPrivateDnsZone
  name: fabricDfsDnsZoneLinkName
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

// VNet link for Fabric Blob DNS Zone
resource fabricBlobPrivateDnsZoneVirtualNetworkLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: fabricBlobPrivateDnsZone
  name: fabricBlobDnsZoneLinkName
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

// VNet link for Fabric API DNS Zone
resource fabricApiPrivateDnsZoneVirtualNetworkLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: fabricApiPrivateDnsZone
  name: fabricApiDnsZoneLinkName
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

// VNet link for OneLake DFS DNS Zone
resource onelakeDfsPrivateDnsZoneVirtualNetworkLink 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = {
  parent: onelakeDfsPrivateDnsZone
  name: onelakeDfsDnsZoneLinkName
  location: 'global'
  tags: tags
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: vnet.id
    }
  }
}

// =========================================
// Private Endpoints
// Note: Fabric workspace private endpoints are created via Fabric portal
// or REST API. This creates the DNS infrastructure for private connectivity.
// The actual private endpoint connection to a workspace requires:
// 1. Fabric workspace creation (done via Fabric portal/API)
// 2. Private endpoint approval in the workspace network settings
// =========================================

// Private Endpoint for OneLake (tenant-level)
// This enables private access to OneLake global endpoints:
// - onelake.dfs.fabric.microsoft.com
// - onelake.blob.fabric.microsoft.com
resource onelakePrivateEndpoint 'Microsoft.Network/privateEndpoints@2021-08-01' = {
  name: 'pe-${fabricCapacityName}-onelake'
  location: location
  tags: tags
  properties: {
    privateLinkServiceConnections: [
      {
        name: 'onelakePrivateLinkConnection'
        properties: {
          // Note: For tenant-level private links, this needs to be configured
          // through the Fabric Admin portal. The actual privateLinkServiceId
          // comes from the Fabric tenant's Azure Private Link resource.
          // This template creates the networking infrastructure.
          privateLinkServiceId: fabricCapacityId
          groupIds: [
            'onelake'
          ]
        }
      }
    ]
    subnet: {
      id: '${vnet.id}/subnets/${subnetName}'
    }
  }
}

// Private DNS Zone Group for OneLake endpoint
resource onelakePrivateEndpointDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2021-08-01' = {
  parent: onelakePrivateEndpoint
  name: 'onelakePrivateDnsZoneGroup'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'fabric-dfs-config'
        properties: {
          privateDnsZoneId: fabricDfsPrivateDnsZone.id
        }
      }
      {
        name: 'fabric-blob-config'
        properties: {
          privateDnsZoneId: fabricBlobPrivateDnsZone.id
        }
      }
      {
        name: 'fabric-api-config'
        properties: {
          privateDnsZoneId: fabricApiPrivateDnsZone.id
        }
      }
      {
        name: 'onelake-dfs-config'
        properties: {
          privateDnsZoneId: onelakeDfsPrivateDnsZone.id
        }
      }
    ]
  }
}

// =========================================
// Manual DNS Records for OneLake
// These are required to route traffic through the private endpoint
// =========================================

// A record for OneLake DFS global endpoint
resource onelakeDfsARecord 'Microsoft.Network/privateDnsZones/A@2020-06-01' = {
  parent: fabricDfsPrivateDnsZone
  name: 'onelake'
  properties: {
    ttl: 3600
    aRecords: [
      {
        ipv4Address: onelakePrivateEndpoint.properties.customDnsConfigs[0].ipAddresses[0]
      }
    ]
  }
}

// A record for OneLake Blob global endpoint
resource onelakeBlobARecord 'Microsoft.Network/privateDnsZones/A@2020-06-01' = {
  parent: fabricBlobPrivateDnsZone
  name: 'onelake'
  properties: {
    ttl: 3600
    aRecords: [
      {
        ipv4Address: onelakePrivateEndpoint.properties.customDnsConfigs[0].ipAddresses[0]
      }
    ]
  }
}

// Outputs
output privateEndpointId string = onelakePrivateEndpoint.id
output fabricDfsPrivateDnsZoneId string = fabricDfsPrivateDnsZone.id
output fabricBlobPrivateDnsZoneId string = fabricBlobPrivateDnsZone.id
output fabricApiPrivateDnsZoneId string = fabricApiPrivateDnsZone.id
output onelakeDfsPrivateDnsZoneId string = onelakeDfsPrivateDnsZone.id
