@description('Name of the public IP address')
param publicIpName string

@description('Location for the public IP')
param location string = resourceGroup().location

@description('Tags for the resource')
param tags object = {}

@description('SKU for the public IP (Standard required for LoadBalancer)')
param sku string = 'Standard'

@description('Allocation method (Static required for LoadBalancer)')
param allocationMethod string = 'Static'

@description('DNS label for the public IP (optional)')
param dnsLabel string = ''

resource publicIp 'Microsoft.Network/publicIPAddresses@2023-05-01' = {
  name: publicIpName
  location: location
  tags: tags
  sku: {
    name: sku
    tier: 'Regional'
  }
  properties: {
    publicIPAllocationMethod: allocationMethod
    dnsSettings: !empty(dnsLabel) ? {
      domainNameLabel: dnsLabel
    } : null
  }
}

output publicIpId string = publicIp.id
output publicIpAddress string = publicIp.properties.ipAddress
output publicIpFqdn string = !empty(dnsLabel) ? publicIp.properties.dnsSettings.fqdn : ''
