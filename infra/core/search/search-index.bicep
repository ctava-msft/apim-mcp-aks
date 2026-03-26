// Azure AI Search Index for Prior Authorization Knowledge Base
// Creates the search index with vector fields for text-embedding-3-large (3072 dimensions)
// and configures semantic search for hybrid retrieval.
// Creates 3 knowledge sources:
//   1. utilization-management-guidance (searchIndex) – policy documents in AI Search
//   2. cms-pa-rule (web) – CMS Interoperability & Prior Authorization Final Rule website
//   3. utilization-management-facts (azureBlob) – JSON fact documents in blob storage
// All 3 are connected to a single knowledge base for agentic retrieval.
// NOTE: Bicep cannot create search indexes/KB directly via ARM.
// This module provisions a deployment script that creates them via the REST API.

@description('Name of the Azure AI Search service')
param searchServiceName string

@description('Location for deployment script resources')
param location string = resourceGroup().location

@description('Tags for resources')
param tags object = {}

@description('Name of the search index to create')
param indexName string = 'prior-authorization'

@description('Name of the search-index knowledge source (policy data)')
param ksSearchIndexName string = 'utilization-management-guidance'

@description('Name of the web knowledge source (CMS PA rule)')
param ksWebName string = 'cms-pa-rule'

@description('Name of the blob knowledge source (UM facts)')
param ksBlobName string = 'utilization-management-facts'

@description('Name of the knowledge base to create')
param knowledgeBaseName string = 'prior-authorization-kb'

@description('Azure AI Foundry endpoint for KB model (agentic retrieval reasoning)')
param foundryEndpoint string = ''

@description('Model deployment name for KB reasoning')
param modelDeploymentName string = 'gpt-5'

@description('Model name for KB reasoning (API enum value)')
param modelName string = 'gpt-5'

@description('Embedding model deployment name for blob ingestion')
param embeddingModelDeploymentName string = 'text-embedding-3-large'

@description('Managed identity resource ID for the deployment script')
param managedIdentityId string

@description('Storage account connection string in ResourceId format for blob knowledge source')
param storageConnectionString string = ''

@description('Blob container name for UM facts')
param umFactsContainerName string = 'um-facts'

@description('CMS Prior Authorization rule domain')
param cmsDomain string = 'www.cms.gov'

// Reference the existing search service
resource searchService 'Microsoft.Search/searchServices@2024-06-01-preview' existing = {
  name: searchServiceName
}

// Deployment script to create the search index and knowledge sources via REST API
resource createSearchIndex 'Microsoft.Resources/deploymentScripts@2023-08-01' = {
  name: 'create-search-index-${indexName}'
  location: location
  tags: tags
  kind: 'AzureCLI'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentityId}': {}
    }
  }
  properties: {
    azCliVersion: '2.63.0'
    timeout: 'PT15M'
    retentionInterval: 'P1D'
    cleanupPreference: 'OnSuccess'
    environmentVariables: [
      {
        name: 'SEARCH_SERVICE_NAME'
        value: searchServiceName
      }
      {
        name: 'INDEX_NAME'
        value: indexName
      }
      {
        name: 'KS_SEARCH_INDEX_NAME'
        value: ksSearchIndexName
      }
      {
        name: 'KS_WEB_NAME'
        value: ksWebName
      }
      {
        name: 'KS_BLOB_NAME'
        value: ksBlobName
      }
      {
        name: 'KNOWLEDGE_BASE_NAME'
        value: knowledgeBaseName
      }
      {
        name: 'FOUNDRY_ENDPOINT'
        value: foundryEndpoint
      }
      {
        name: 'MODEL_DEPLOYMENT_NAME'
        value: modelDeploymentName
      }
      {
        name: 'MODEL_NAME'
        value: modelName
      }
      {
        name: 'EMBEDDING_MODEL_DEPLOYMENT_NAME'
        value: embeddingModelDeploymentName
      }
      {
        name: 'STORAGE_CONNECTION_STRING'
        value: storageConnectionString
      }
      {
        name: 'UM_FACTS_CONTAINER'
        value: umFactsContainerName
      }
      {
        name: 'CMS_DOMAIN'
        value: cmsDomain
      }
    ]
    scriptContent: '''
      #!/bin/bash
      set -e

      # Get access token for Azure AI Search
      TOKEN=$(az account get-access-token --resource https://search.azure.com --query accessToken -o tsv)
      SEARCH_URL="https://${SEARCH_SERVICE_NAME}.search.windows.net"
      KS_API="2025-11-01-preview"

      # ── 1. Create or verify the search index ──────────────────
      STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${TOKEN}" \
        "${SEARCH_URL}/indexes/${INDEX_NAME}?api-version=2024-07-01")

      if [ "$STATUS" = "200" ]; then
        echo "Index ${INDEX_NAME} already exists. Skipping creation."
      else
        curl -s -X PUT \
          -H "Authorization: Bearer ${TOKEN}" \
          -H "Content-Type: application/json" \
          "${SEARCH_URL}/indexes/${INDEX_NAME}?api-version=2024-07-01" \
          -d '{
            "name": "'${INDEX_NAME}'",
            "fields": [
              {"name": "id", "type": "Edm.String", "key": true, "searchable": false, "filterable": true},
              {"name": "content", "type": "Edm.String", "searchable": true, "filterable": false, "sortable": false},
              {"name": "title", "type": "Edm.String", "searchable": true, "filterable": true, "sortable": true},
              {"name": "source", "type": "Edm.String", "searchable": false, "filterable": true, "sortable": true},
              {"name": "source_type", "type": "Edm.String", "searchable": false, "filterable": true, "facetable": true},
              {"name": "policy_id", "type": "Edm.String", "searchable": false, "filterable": true},
              {"name": "effective_date", "type": "Edm.String", "searchable": false, "filterable": true, "sortable": true},
              {"name": "version", "type": "Edm.String", "searchable": false, "filterable": true},
              {"name": "line_of_business", "type": "Edm.String", "searchable": false, "filterable": true, "facetable": true},
              {"name": "chunk_index", "type": "Edm.Int32", "searchable": false, "filterable": true, "sortable": true},
              {"name": "metadata", "type": "Edm.String", "searchable": false, "filterable": false},
              {"name": "embedding", "type": "Collection(Edm.Single)", "searchable": true, "dimensions": 3072, "vectorSearchProfile": "vector-profile"}
            ],
            "vectorSearch": {
              "algorithms": [
                {
                  "name": "hnsw-algorithm",
                  "kind": "hnsw",
                  "hnswParameters": {
                    "metric": "cosine",
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500
                  }
                }
              ],
              "profiles": [
                {
                  "name": "vector-profile",
                  "algorithm": "hnsw-algorithm"
                }
              ]
            },
            "semantic": {
              "configurations": [
                {
                  "name": "semantic-config",
                  "prioritizedFields": {
                    "contentFields": [{"fieldName": "content"}],
                    "titleField": {"fieldName": "title"}
                  }
                }
              ]
            }
          }'
        echo "Index ${INDEX_NAME} created successfully."
      fi
      echo ""

      # ── 2. Delete old knowledge source if it exists ───────────
      OLD_KS="prior-authorization-source"
      OLD_KS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${TOKEN}" \
        "${SEARCH_URL}/knowledgesources/${OLD_KS}?api-version=${KS_API}")
      if [ "$OLD_KS_STATUS" = "200" ]; then
        echo "Deleting old knowledge source: ${OLD_KS}..."
        curl -s -X DELETE \
          -H "Authorization: Bearer ${TOKEN}" \
          "${SEARCH_URL}/knowledgesources/${OLD_KS}?api-version=${KS_API}"
        echo "Old knowledge source deleted."
      fi

      # ── 3. Knowledge Source 1: searchIndex (policy data) ──────
      echo "Creating Knowledge Source: ${KS_SEARCH_INDEX_NAME} (searchIndex)..."
      KS1_URL="${SEARCH_URL}/knowledgesources/${KS_SEARCH_INDEX_NAME}?api-version=${KS_API}"

      # Always PUT (create or update) to ensure correct configuration
      curl -s -X PUT \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" \
        -H "Prefer: return=representation" \
        "${KS1_URL}" \
        -d '{
          "name": "'${KS_SEARCH_INDEX_NAME}'",
          "kind": "searchIndex",
          "description": "Utilization management policy documents ingested into AI Search index",
          "searchIndexParameters": {
            "searchIndexName": "'${INDEX_NAME}'",
            "semanticConfigurationName": "semantic-config",
            "sourceDataFields": [
              {"name": "id"},
              {"name": "title"},
              {"name": "content"},
              {"name": "source_type"},
              {"name": "policy_id"}
            ]
          }
        }'
      echo ""
      echo "Knowledge Source ${KS_SEARCH_INDEX_NAME} created/updated."

      # ── 4. Knowledge Source 2: web (CMS PA rule) ──────────────
      echo "Creating Knowledge Source: ${KS_WEB_NAME} (web)..."
      KS2_URL="${SEARCH_URL}/knowledgesources/${KS_WEB_NAME}?api-version=${KS_API}"

      curl -s -X PUT \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" \
        -H "Prefer: return=representation" \
        "${KS2_URL}" \
        -d '{
          "name": "'${KS_WEB_NAME}'",
          "kind": "web",
          "description": "CMS Interoperability and Prior Authorization Final Rule (CMS-0057-F)",
          "webParameters": {
            "domains": {
              "allowedDomains": [
                {
                  "address": "'${CMS_DOMAIN}'",
                  "includeSubpages": true
                }
              ]
            }
          }
        }'
      echo ""
      echo "Knowledge Source ${KS_WEB_NAME} created/updated."

      # ── 5. Knowledge Source 3: azureBlob (UM facts) ───────────
      echo "Creating Knowledge Source: ${KS_BLOB_NAME} (azureBlob)..."
      KS3_URL="${SEARCH_URL}/knowledgesources/${KS_BLOB_NAME}?api-version=${KS_API}"

      if [ -n "${STORAGE_CONNECTION_STRING}" ]; then
        # Build blob knowledge source with ingestion parameters
        BLOB_BODY='{
          "name": "'${KS_BLOB_NAME}'",
          "kind": "azureBlob",
          "description": "Utilization management facts (medical necessity, step therapy, exception logic) in JSON format",
          "azureBlobParameters": {
            "connectionString": "'${STORAGE_CONNECTION_STRING}'",
            "containerName": "'${UM_FACTS_CONTAINER}'",
            "ingestionParameters": {
              "embeddingModel": {
                "kind": "azureOpenAI",
                "azureOpenAIParameters": {
                  "resourceUri": "'${FOUNDRY_ENDPOINT}'",
                  "deploymentId": "'${EMBEDDING_MODEL_DEPLOYMENT_NAME}'",
                  "modelName": "text-embedding-3-large"
                }
              },
              "chatCompletionModel": {
                "kind": "azureOpenAI",
                "azureOpenAIParameters": {
                  "resourceUri": "'${FOUNDRY_ENDPOINT}'",
                  "deploymentId": "'${MODEL_DEPLOYMENT_NAME}'",
                  "modelName": "'${MODEL_NAME}'"
                }
              }
            }
          }
        }'

        curl -s -X PUT \
          -H "Authorization: Bearer ${TOKEN}" \
          -H "Content-Type: application/json" \
          -H "Prefer: return=representation" \
          "${KS3_URL}" \
          -d "${BLOB_BODY}"
        echo ""
        echo "Knowledge Source ${KS_BLOB_NAME} created/updated."
      else
        echo "WARNING: STORAGE_CONNECTION_STRING not provided. Skipping blob knowledge source."
      fi

      # ── 6. Delete old KB and recreate with all 3 sources ──────
      echo "Creating/updating Knowledge Base: ${KNOWLEDGE_BASE_NAME}..."
      KB_URL="${SEARCH_URL}/knowledgebases/${KNOWLEDGE_BASE_NAME}?api-version=${KS_API}"

      # Build model config if foundry endpoint provided
      MODELS_JSON="[]"
      if [ -n "${FOUNDRY_ENDPOINT}" ]; then
        MODELS_JSON='[{"kind":"azureOpenAI","azureOpenAIParameters":{"resourceUri":"'${FOUNDRY_ENDPOINT}'","deploymentId":"'${MODEL_DEPLOYMENT_NAME}'","modelName":"'${MODEL_NAME}'"}}]'
      fi

      # Build knowledge sources list
      KS_LIST='[{"name":"'${KS_SEARCH_INDEX_NAME}'"},{"name":"'${KS_WEB_NAME}'"}'
      if [ -n "${STORAGE_CONNECTION_STRING}" ]; then
        KS_LIST="${KS_LIST}"',{"name":"'${KS_BLOB_NAME}'"}'
      fi
      KS_LIST="${KS_LIST}]"

      # Delete existing KB first to update knowledge sources list
      KB_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${TOKEN}" \
        "${KB_URL}")
      if [ "$KB_STATUS" = "200" ]; then
        echo "Deleting existing KB to update knowledge sources..."
        curl -s -X DELETE \
          -H "Authorization: Bearer ${TOKEN}" \
          "${KB_URL}"
        sleep 2
      fi

      curl -s -X PUT \
        -H "Authorization: Bearer ${TOKEN}" \
        -H "Content-Type: application/json" \
        -H "Prefer: return=representation" \
        "${KB_URL}" \
        -d '{
          "name": "'${KNOWLEDGE_BASE_NAME}'",
          "description": "Prior authorization knowledge base with policy data, CMS regulatory guidance, and UM facts",
          "outputMode": "answerSynthesis",
          "retrievalReasoningEffort": {"kind": "medium"},
          "knowledgeSources": '${KS_LIST}',
          "models": '${MODELS_JSON}'
        }'
      echo ""
      echo "Knowledge Base ${KNOWLEDGE_BASE_NAME} created with all knowledge sources."

      echo "{\"indexCreated\": true, \"knowledgeSources\": [\"${KS_SEARCH_INDEX_NAME}\", \"${KS_WEB_NAME}\", \"${KS_BLOB_NAME}\"], \"knowledgeBase\": \"${KNOWLEDGE_BASE_NAME}\"}" > $AZ_SCRIPTS_OUTPUT_PATH
    '''
  }
}

output indexName string = indexName
output ksSearchIndexName string = ksSearchIndexName
output ksWebName string = ksWebName
output ksBlobName string = ksBlobName
output knowledgeBaseName string = knowledgeBaseName
output searchServiceEndpoint string = 'https://${searchServiceName}.search.windows.net'
