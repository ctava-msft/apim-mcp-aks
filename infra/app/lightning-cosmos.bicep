// =========================================
// Agent Lightning Cosmos DB Resources
// =========================================
// This module provisions the Cosmos DB database and containers
// required for Agent Lightning's RL fine-tuning loop:
// - rl_episodes: Agent interactions (input, tools, output)
// - rl_rewards: Labels and scores attached to episodes
// - rl_datasets: Dataset manifests for fine-tuning
// - rl_training_runs: Training job records
// - rl_deployments: Active tuned model registry
//
// See docs/AGENT-LIGHTNING.md for details

@description('Name of the parent Azure Cosmos DB account.')
param parentAccountName string

@description('Name of the Lightning database (default: agent_rl).')
param databaseName string = 'agent_rl'

@description('Tags for all resources.')
param tags object = {}

// Reference to existing Cosmos account
resource account 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' existing = {
  name: parentAccountName
}

// =========================================
// Lightning Database
// =========================================
resource lightningDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2024-05-15' = {
  name: databaseName
  parent: account
  tags: tags
  properties: {
    resource: {
      id: databaseName
    }
  }
}

// =========================================
// RL Episodes Container
// Stores agent interactions (prompt → tool calls → response)
// Partition key: agent_id
// =========================================
resource episodesContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  name: 'rl_episodes'
  parent: lightningDatabase
  tags: tags
  properties: {
    resource: {
      id: 'rl_episodes'
      partitionKey: {
        paths: ['/agent_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
      // TTL disabled by default - episodes are long-lived training data
      defaultTtl: -1
    }
  }
}

// =========================================
// RL Rewards Container
// Stores labels/scores attached to episodes
// Partition key: agent_id
// =========================================
resource rewardsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  name: 'rl_rewards'
  parent: lightningDatabase
  tags: tags
  properties: {
    resource: {
      id: 'rl_rewards'
      partitionKey: {
        paths: ['/agent_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// =========================================
// RL Datasets Container
// Stores dataset manifests for fine-tuning
// Partition key: agent_id
// =========================================
resource datasetsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  name: 'rl_datasets'
  parent: lightningDatabase
  tags: tags
  properties: {
    resource: {
      id: 'rl_datasets'
      partitionKey: {
        paths: ['/agent_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// =========================================
// RL Training Runs Container
// Stores training job records
// Partition key: agent_id
// =========================================
resource trainingRunsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  name: 'rl_training_runs'
  parent: lightningDatabase
  tags: tags
  properties: {
    resource: {
      id: 'rl_training_runs'
      partitionKey: {
        paths: ['/agent_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// =========================================
// RL Deployments Container
// Stores active tuned model registry
// Partition key: agent_id
// =========================================
resource deploymentsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  name: 'rl_deployments'
  parent: lightningDatabase
  tags: tags
  properties: {
    resource: {
      id: 'rl_deployments'
      partitionKey: {
        paths: ['/agent_id']
        kind: 'Hash'
        version: 2
      }
      indexingPolicy: {
        automatic: true
        indexingMode: 'consistent'
        includedPaths: [
          { path: '/*' }
        ]
        excludedPaths: [
          { path: '/"_etag"/?' }
        ]
      }
    }
  }
}

// =========================================
// Outputs
// =========================================
output databaseName string = lightningDatabase.name
output episodesContainerName string = episodesContainer.name
output rewardsContainerName string = rewardsContainer.name
output datasetsContainerName string = datasetsContainer.name
output trainingRunsContainerName string = trainingRunsContainer.name
output deploymentsContainerName string = deploymentsContainer.name
