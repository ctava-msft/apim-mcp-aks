# Microsoft Purview Integration - Implementation Guide

## Overview
This guide provides comprehensive documentation for the Microsoft Purview integration into the Azure Agents Control Plane, enabling data governance, classification, lineage tracking, and compliance management for AI agent operations.

## Architecture Overview
Microsoft Purview enhances the following Enterprise Agent Governance pillars:
- **Pillar 1 (API Governance)**: Data classification and sensitivity tracking through APIM
- **Pillar 6 (Observability)**: Data lineage and flow tracking
- **Pillar 7 (Secrets & Artifacts)**: Data catalog and compliance reporting

### Key Components
- **Purview Account**: Central data governance and catalog service
- **Data Sources**: Cosmos DB, Azure Storage, Fabric workspace
- **Private Endpoints**: Secure VNet connectivity
- **Role-Based Access**: Managed identity integration

## Prerequisites
- Azure CLI installed and authenticated (`az login`)
- Azure subscription with appropriate permissions
- Purview resource provider registered: `Microsoft.Purview`
- VNet-enabled deployment (recommended for production)
- Agent Identity enabled for runtime data classification checks

## Phase 1: Infrastructure Setup ✅

### Bicep Modules Created
1. **`infra/core/purview/purview.bicep`**: Core Purview account module
2. **`infra/app/purview-PrivateEndpoint.bicep`**: Private networking for Purview
3. **`infra/app/purview-RoleAssignment.bicep`**: RBAC role assignments

### Deployment Instructions

#### Enable Purview Deployment
```bash
# Set environment variable to enable Purview
azd env set PURVIEW_ENABLED true

# Deploy infrastructure
azd provision
```

#### Verify Deployment
```bash
# Check Purview account
az purview account show \
  --name $(azd env get-value PURVIEW_ACCOUNT_NAME) \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP_NAME)

# Verify private endpoints (if VNet enabled)
az network private-endpoint list \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP_NAME) \
  --query "[?contains(name, 'purview')]"
```

### Infrastructure Outputs
After deployment, the following outputs are available:
- `PURVIEW_ENABLED`: Boolean indicating if Purview is enabled
- `PURVIEW_ACCOUNT_NAME`: Name of the Purview account
- `PURVIEW_ENDPOINT`: Main Purview endpoint URL
- `PURVIEW_CATALOG_ENDPOINT`: Data Catalog API endpoint
- `PURVIEW_SCAN_ENDPOINT`: Scanning API endpoint
- `PURVIEW_MANAGED_RESOURCE_GROUP`: Managed resource group for Purview resources

## Phase 2: Data Source Registration

### Overview
After infrastructure deployment, data sources must be registered in Purview for scanning and cataloging. This can be done through the Azure Portal, Azure CLI, or Purview REST API.

### 1. Register Cosmos DB (Agent Memory)

#### Purpose
- Track agent conversation history and memory storage
- Classify sensitive data in agent interactions
- Monitor data retention compliance

#### Registration Steps

**Using Azure Portal:**
1. Navigate to Purview Studio: `https://<purview-account-name>.purview.azure.com`
2. Go to **Data Map** → **Sources**
3. Click **Register** → Select **Azure Cosmos DB**
4. Provide connection details:
   - **Name**: `cosmos-agent-memory`
   - **Subscription**: Select your subscription
   - **Cosmos DB Account**: Select the account from `COSMOSDB_ACCOUNT_NAME` output
   - **Collection**: Create new collection `agent-data-sources`

**Using Azure CLI:**
```bash
# Get Purview account details
PURVIEW_ACCOUNT=$(azd env get-value PURVIEW_ACCOUNT_NAME)
RESOURCE_GROUP=$(azd env get-value AZURE_RESOURCE_GROUP_NAME)
COSMOS_ACCOUNT=$(azd env get-value COSMOSDB_ACCOUNT_NAME)
COSMOS_ENDPOINT=$(azd env get-value COSMOSDB_ENDPOINT)

# Register Cosmos DB data source
# Note: This requires the Purview REST API or Portal
echo "Register Cosmos DB at: https://$PURVIEW_ACCOUNT.purview.azure.com"
echo "Data Source: $COSMOS_ACCOUNT"
echo "Endpoint: $COSMOS_ENDPOINT"
```

#### Scanning Configuration
**Scan Details:**
- **Scan Name**: `cosmos-agent-memory-scan`
- **Scope**: Select databases: `mcpdb` (or your configured database)
- **Collections**: `tasks`, `conversations`, `approvals`, `rl_checkpoints`
- **Credential**: Use Purview Managed Identity (already configured via RBAC)

**Scan Schedule:**
- **Frequency**: Weekly (recommended)
- **Time**: Off-peak hours (e.g., 2:00 AM UTC Sunday)
- **Incremental**: Enable incremental scans after initial full scan

**Classification Rules:**
- Enable built-in classification rules for PII detection
- Custom rules for agent-specific patterns (see Phase 3)

### 2. Register Azure Storage (Ontologies and Task Instructions)

#### Purpose
- Catalog ontology definitions and agent knowledge bases
- Track task instruction versions and lineage
- Classify data sensitivity levels

#### Registration Steps

**Data Source Details:**
- **Name**: `storage-ontologies-tasks`
- **Type**: Azure Blob Storage
- **Account**: From `AZURE_STORAGE_ACCOUNT_URL` output
- **Collection**: `agent-data-sources`

**Containers to Include:**
- `ontologies`: Fabric IQ ontology definitions (when Fabric is disabled)
- Task instruction containers from Azure AI Search integration
- Agent artifact storage containers

**Using Azure Portal:**
1. Navigate to Purview Studio → **Data Map** → **Sources**
2. Click **Register** → Select **Azure Blob Storage**
3. Provide storage account details
4. Select specific containers for scanning

#### Scanning Configuration
**Scan Details:**
- **Scan Name**: `storage-ontologies-scan`
- **Scope**: Select containers: `ontologies`, `task-instructions`
- **File Types**: JSON, YAML, Markdown, Text
- **Credential**: Purview Managed Identity

**Scan Schedule:**
- **Frequency**: Daily (ontologies may change frequently)
- **Time**: Off-peak hours
- **Incremental**: Enabled

### 3. Register Fabric Workspace (Facts and Knowledge Assets)

#### Purpose
- Catalog Fabric lakehouses containing agent facts
- Track semantic models and data warehouses
- Monitor data pipeline lineage

#### Prerequisites
- Fabric enabled: `FABRIC_ENABLED=true`
- Fabric workspace ID available: `FABRIC_WORKSPACE_ID`

#### Registration Steps

**Data Source Details:**
- **Name**: `fabric-agent-workspace`
- **Type**: Microsoft Fabric Workspace
- **Workspace ID**: From `FABRIC_WORKSPACE_ID` output
- **Collection**: `agent-data-sources`

**Assets to Include:**
- **Lakehouses**: Facts and entity storage
- **Warehouses**: Consolidated agent data
- **Semantic Models**: Power BI reports on agent performance
- **Data Pipelines**: Fact synchronization workflows

**Using Azure Portal:**
1. Navigate to Purview Studio → **Data Map** → **Sources**
2. Click **Register** → Select **Microsoft Fabric**
3. Provide workspace details
4. Configure authentication (Purview managed identity should have Fabric workspace access)

#### Scanning Configuration
**Scan Details:**
- **Scan Name**: `fabric-workspace-scan`
- **Scope**: All lakehouses and warehouses
- **Credential**: Purview Managed Identity (requires Fabric Reader role)
- **Metadata**: Include schema, columns, relationships

**Scan Schedule:**
- **Frequency**: Daily
- **Time**: After Fabric pipeline completion (e.g., 6:00 AM UTC)
- **Incremental**: Enabled

### 4. Automatic Scanning Configuration

#### Global Scan Settings
Configure automatic scanning policies for all data sources:

1. **Navigate to**: Purview Studio → **Data Map** → **Scans**
2. **Enable automatic discovery**:
   - New resources in subscription
   - Schema change detection
   - Metadata updates

#### Scan Triggers
Set up triggers for automatic scans:
- **Event-based**: Trigger on data source changes (via Azure Event Grid)
- **Schedule-based**: Daily/weekly recurring scans
- **Manual**: On-demand scans via API or Portal

#### Monitoring and Alerts
Configure alerts for scan failures:
1. Navigate to **Monitor** → **Scan alerts**
2. Set up email notifications for:
   - Scan failures
   - Schema drift detection
   - New sensitive data discovery

## Phase 3: Classification & Lineage

### 1. Custom Classification Rules for Agent-Specific Data Patterns

#### Agent Conversation Patterns
Create custom classification rules to detect agent-specific data:

**Rule: Agent Conversation ID**
- **Pattern**: `conv_[a-f0-9]{32}` (regex)
- **Sensitivity**: Internal
- **Description**: Identifies agent conversation identifiers
- **Data type**: String

**Rule: Agent Task ID**
- **Pattern**: `task_[a-f0-9]{32}` (regex)
- **Sensitivity**: Internal
- **Description**: Identifies agent task identifiers

**Rule: Agent Memory Key**
- **Pattern**: `memory_[a-zA-Z0-9_-]+` (regex)
- **Sensitivity**: Confidential
- **Description**: Agent memory and fact identifiers

**Rule: Agent Prompt Template**
- **Keywords**: `system_prompt`, `user_prompt`, `assistant_response`
- **Sensitivity**: Internal
- **Description**: Agent prompt engineering artifacts

#### Creating Custom Rules

**Using Azure Portal:**
1. Navigate to Purview Studio → **Data Map** → **Classifications**
2. Click **+ New classification**
3. Provide details:
   - **Name**: `Agent Conversation ID`
   - **Type**: Custom
   - **Pattern type**: Regex
   - **Pattern**: `conv_[a-f0-9]{32}`
   - **Minimum match threshold**: 60%
4. Save and publish

**Using REST API:**
```bash
# Example: Create custom classification via API
# Note: Requires authentication token
PURVIEW_ENDPOINT=$(azd env get-value PURVIEW_CATALOG_ENDPOINT)

curl -X PUT "$PURVIEW_ENDPOINT/api/atlas/v2/types/typedefs" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "classificationDefs": [{
      "name": "AgentConversationID",
      "description": "Agent conversation identifier pattern",
      "attributeDefs": [{
        "name": "pattern",
        "typeName": "string",
        "isOptional": false,
        "defaultValue": "conv_[a-f0-9]{32}"
      }]
    }]
  }'
```

### 2. Sensitivity Labels Configuration

#### Label Hierarchy
Configure sensitivity labels aligned with organizational data classification:

**Label Tiers:**
1. **Public**: Publicly shareable agent information
   - Example: Public API documentation, agent capabilities
2. **Internal**: Internal-only agent data
   - Example: Agent configuration, task templates
3. **Confidential**: Business confidential data
   - Example: Agent memory, user interactions, task results
4. **Highly Confidential**: PII and regulated data
   - Example: User PII, financial data, health information

#### Configuring Labels

**Using Microsoft Purview Studio:**
1. Navigate to **Data Map** → **Sensitivity labels**
2. Integrate with Microsoft Purview Information Protection
3. Map Purview classifications to sensitivity labels:
   - PII classifications → Highly Confidential
   - Agent conversation data → Confidential
   - Agent configuration → Internal

**Auto-labeling Policies:**
Create policies to automatically apply labels:
- Files containing email addresses → Highly Confidential
- Files matching agent conversation patterns → Confidential
- Task instruction files → Internal

### 3. Lineage Capture Implementation

#### Data Lineage Architecture
Purview captures lineage through:
1. **Automated lineage**: From supported Azure services (Cosmos DB, Storage)
2. **Custom lineage**: Using Atlas API for agent orchestration flows
3. **Pipeline lineage**: From Fabric Data Factory pipelines

#### Implementing Custom Lineage for Agent Flows

**Agent Request → Response Flow:**
```
API Request (APIM) 
  → Agent Orchestration (AKS)
    → Memory Retrieval (Cosmos DB)
      → Task Execution (Agent Runtime)
        → Result Storage (Cosmos DB)
          → API Response (APIM)
```

**Lineage Capture Points:**
1. **APIM Policy**: Log request/response metadata
2. **Agent Runtime**: Emit lineage events to Purview
3. **Cosmos DB**: Automatic lineage via Purview connector

**Custom Lineage via Atlas API:**
```python
# Example: Report lineage from agent runtime to Purview
from azure.purview.catalog import PurviewCatalogClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
purview_endpoint = os.environ["PURVIEW_CATALOG_ENDPOINT"]
client = PurviewCatalogClient(endpoint=purview_endpoint, credential=credential)

# Define lineage relationship
lineage = {
    "entity": {
        "typeName": "agent_execution_process",
        "attributes": {
            "name": "agent-task-execution",
            "qualifiedName": f"agent://{task_id}",
            "inputs": [
                {"typeName": "agent_memory", "guid": memory_guid}
            ],
            "outputs": [
                {"typeName": "task_result", "guid": result_guid}
            ]
        }
    }
}

client.entity.create_or_update(body=lineage)
```

### 4. APIM Policy Integration for Data Classification Headers

#### Purpose
Add data classification metadata to API requests/responses for downstream processing and compliance.

#### APIM Policy Example
```xml
<!-- Inbound policy: Add classification headers -->
<inbound>
    <base />
    <!-- Set classification context -->
    <set-header name="X-Data-Classification" exists-action="override">
        <value>Confidential</value>
    </set-header>
    <set-header name="X-Purview-Lineage-ID" exists-action="override">
        <value>@(Guid.NewGuid().ToString())</value>
    </set-header>
</inbound>

<!-- Outbound policy: Log to Application Insights with classification -->
<outbound>
    <base />
    <log-to-eventhub logger-id="purview-lineage-logger">
        @{
            var lineageId = context.Request.Headers.GetValueOrDefault("X-Purview-Lineage-ID", "");
            var classification = context.Request.Headers.GetValueOrDefault("X-Data-Classification", "Internal");
            
            return new {
                timestamp = DateTime.UtcNow,
                lineageId = lineageId,
                classification = classification,
                requestPath = context.Request.Url.Path,
                responseStatus = context.Response.StatusCode,
                userId = context.Request.Headers.GetValueOrDefault("X-User-ID", "anonymous")
            };
        }
    </log-to-eventhub>
</outbound>
```

#### Policy Implementation Steps
1. Navigate to APIM → APIs → MCP API
2. Select **Inbound processing** → Add policy
3. Add classification headers based on endpoint sensitivity
4. Configure Event Hub for Purview lineage ingestion (optional)

## Phase 4: Compliance Integration

### 1. Purview Compliance Reports

#### Available Reports
Purview provides several out-of-box compliance reports:
- **Data Classification Summary**: Overview of data sensitivity
- **Data Lineage Report**: End-to-end data flow visualization
- **Access Control Report**: Who has access to what data
- **Scan Coverage Report**: Percentage of data sources scanned

#### Accessing Reports

**Using Purview Studio:**
1. Navigate to **Insights** → **Reports**
2. Select report type
3. Filter by:
   - Collection: `agent-data-sources`
   - Date range
   - Classification type

**Exporting Reports:**
- **Format**: CSV, PDF, Excel
- **Schedule**: Daily, weekly, monthly
- **Distribution**: Email, Power BI, Azure Logic Apps

#### Integration with Governance Dashboard
If using a custom governance dashboard (e.g., Grafana, Power BI):
1. Use Purview REST API to fetch compliance metrics
2. Create scheduled queries to pull data
3. Visualize in dashboard

**Example: Power BI Integration**
```powerquery
let
    Source = Json.Document(Web.Contents(
        "https://" & PurviewAccount & ".purview.azure.com/api/atlas/v2/search/basic",
        [Headers=[#"Authorization"="Bearer " & Token]]
    )),
    entities = Source[entities]
in
    entities
```

### 2. Retention Policies Configuration

#### Data Retention Requirements
Align retention policies with data residency and compliance requirements:
- **Agent conversations**: 90 days (GDPR compliance)
- **Task instructions**: 7 years (audit trail)
- **Agent memory**: Variable (user-driven)
- **Audit logs**: 1 year (SOC2 compliance)

#### Configuring Retention in Purview

**Data Lifecycle Management:**
1. Navigate to **Data Map** → **Collections** → `agent-data-sources`
2. Select **Policies** → **Retention policies**
3. Create policy:
   - **Name**: `agent-conversation-retention`
   - **Duration**: 90 days
   - **Action**: Delete or archive
   - **Scope**: Cosmos DB conversations container

**Cosmos DB TTL Configuration:**
```bicep
// Set TTL in Cosmos DB for automatic deletion
resource conversationsContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2024-05-15' = {
  // ... other properties
  properties: {
    defaultTtl: 7776000  // 90 days in seconds
  }
}
```

**Azure Storage Lifecycle Management:**
```bash
# Configure blob lifecycle for ontologies
az storage account management-policy create \
  --account-name $(azd env get-value AZURE_STORAGE_ACCOUNT_NAME) \
  --policy @lifecycle-policy.json
```

**lifecycle-policy.json:**
```json
{
  "rules": [{
    "enabled": true,
    "name": "archive-old-ontologies",
    "type": "Lifecycle",
    "definition": {
      "actions": {
        "baseBlob": {
          "tierToCool": { "daysAfterModificationGreaterThan": 30 },
          "tierToArchive": { "daysAfterModificationGreaterThan": 180 },
          "delete": { "daysAfterModificationGreaterThan": 2555 }
        }
      },
      "filters": {
        "blobTypes": ["blockBlob"],
        "prefixMatch": ["ontologies/"]
      }
    }
  }]
}
```

### 3. Audit Log Forwarding from Application Insights to Purview

#### Overview
Forward agent operation logs from Application Insights to Purview for compliance auditing and lineage tracking.

#### Architecture
```
Application Insights 
  → Log Analytics Workspace 
    → Azure Data Explorer (optional)
      → Purview Lineage API
```

#### Implementation Steps

**1. Configure Diagnostic Settings:**
```bash
# Enable diagnostic logging to Log Analytics
APP_INSIGHTS_ID=$(az monitor app-insights component show \
  --app $(azd env get-value APPLICATIONINSIGHTS_NAME) \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP_NAME) \
  --query id -o tsv)

LOG_ANALYTICS_ID=$(az monitor log-analytics workspace show \
  --workspace-name $(azd env get-value LOG_ANALYTICS_NAME) \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP_NAME) \
  --query id -o tsv)

az monitor diagnostic-settings create \
  --name purview-audit-logs \
  --resource $APP_INSIGHTS_ID \
  --logs '[{"category": "Audit", "enabled": true}]' \
  --workspace $LOG_ANALYTICS_ID
```

**2. Create Log Analytics Query:**
```kusto
// Query for agent operations with data access
AppTraces
| where TimeGenerated > ago(24h)
| where Message contains "agent_execution" or Message contains "data_access"
| extend 
    TaskId = tostring(Properties.task_id),
    UserId = tostring(Properties.user_id),
    DataSource = tostring(Properties.data_source),
    Classification = tostring(Properties.classification)
| project 
    TimeGenerated, 
    TaskId, 
    UserId, 
    DataSource, 
    Classification, 
    Message
| order by TimeGenerated desc
```

**3. Schedule Log Export to Purview:**
Use Azure Logic Apps or Azure Functions to periodically:
1. Query Log Analytics for agent operations
2. Transform logs to Purview lineage format
3. POST to Purview Atlas API

**Example Azure Function (Python):**
```python
import azure.functions as func
from azure.monitor.query import LogsQueryClient
from azure.purview.catalog import PurviewCatalogClient
from azure.identity import DefaultAzureCredential
import os

def main(mytimer: func.TimerRequest) -> None:
    credential = DefaultAzureCredential()
    
    # Query Log Analytics
    logs_client = LogsQueryClient(credential)
    workspace_id = os.environ["LOG_ANALYTICS_WORKSPACE_ID"]
    query = """
        AppTraces
        | where TimeGenerated > ago(1h)
        | where Message contains "agent_execution"
    """
    response = logs_client.query_workspace(workspace_id, query, timespan="PT1H")
    
    # Send to Purview
    purview_client = PurviewCatalogClient(
        endpoint=os.environ["PURVIEW_CATALOG_ENDPOINT"],
        credential=credential
    )
    
    for row in response.tables[0].rows:
        lineage_event = {
            "entity": {
                "typeName": "agent_operation",
                "attributes": {
                    "name": row[4],  # Message
                    "timestamp": row[0],  # TimeGenerated
                    "task_id": row[1],  # TaskId
                }
            }
        }
        purview_client.entity.create_or_update(body=lineage_event)
```

### 4. Compliance Attestation Workflows

#### Overview
Implement approval workflows for compliance attestations using the existing Agents Approval Logic App.

#### Integration with Approval Logic App
The deployment includes an optional Agents Approval Logic App (see `AGENTS_APPROVALS.md`). Extend this for Purview compliance:

**Use Cases:**
1. **Data Access Requests**: Require approval before accessing sensitive data
2. **Retention Policy Exceptions**: Approval for extending retention periods
3. **Classification Override**: Approval for changing data sensitivity labels

#### Configuration Steps

**1. Enable Approval Logic App:**
```bash
azd env set APPROVAL_LOGIC_APP_ENABLED true
azd env set TEAMS_CHANNEL_ID "<your-teams-channel>"
azd env set TEAMS_GROUP_ID "<your-teams-group>"
azd provision
```

**2. Create Compliance Approval Workflow:**
Add a new Logic App workflow for Purview-specific approvals:

```json
{
  "definition": {
    "$schema": "https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#",
    "triggers": {
      "manual": {
        "type": "Request",
        "kind": "Http",
        "inputs": {
          "schema": {
            "type": "object",
            "properties": {
              "requestType": { "type": "string" },
              "resourceId": { "type": "string" },
              "requestedBy": { "type": "string" },
              "justification": { "type": "string" }
            }
          }
        }
      }
    },
    "actions": {
      "Post_to_Teams": {
        "type": "ApiConnection",
        "inputs": {
          "host": {
            "connection": {
              "name": "@parameters('$connections')['teams']['connectionId']"
            }
          },
          "method": "post",
          "body": {
            "message": "Purview Compliance Request: @{triggerBody()?['requestType']}"
          }
        }
      },
      "Log_to_Cosmos": {
        "type": "ApiConnection",
        "inputs": {
          "host": {
            "connection": {
              "name": "@parameters('$connections')['cosmosdb']['connectionId']"
            }
          },
          "method": "post",
          "body": {
            "id": "@{guid()}",
            "requestType": "@{triggerBody()?['requestType']}",
            "status": "pending",
            "timestamp": "@{utcNow()}"
          }
        }
      }
    }
  }
}
```

**3. Integrate with Purview API:**
When approvals are granted, automatically update Purview:
- Update retention policies
- Override classifications
- Grant temporary access

## Testing and Validation

### Bicep Validation
```bash
# Validate all Purview Bicep modules
az bicep build --file infra/core/purview/purview.bicep
az bicep build --file infra/app/purview-PrivateEndpoint.bicep
az bicep build --file infra/app/purview-RoleAssignment.bicep
az bicep build --file infra/main.bicep

# Lint check
az bicep lint --file infra/main.bicep
```

### Deployment Testing

**Test Scenario 1: Purview Enabled with VNet**
```bash
azd env set PURVIEW_ENABLED true
azd env set VNET_ENABLED true
azd provision

# Verify outputs
azd env get-values | grep PURVIEW
```

**Test Scenario 2: Purview Disabled (Default)**
```bash
# Don't set PURVIEW_ENABLED (defaults to false)
azd provision

# Verify Purview resources not created
PURVIEW_ENABLED=$(azd env get-value PURVIEW_ENABLED)
echo "Purview enabled: $PURVIEW_ENABLED"  # Should be false
```

### Data Source Registration Validation
```bash
# Verify data sources are registered
PURVIEW_ACCOUNT=$(azd env get-value PURVIEW_ACCOUNT_NAME)

# List registered data sources (requires Purview CLI or API)
# Note: Azure CLI doesn't have native Purview data source commands yet
echo "Access Purview Studio: https://$PURVIEW_ACCOUNT.purview.azure.com"
```

### Scanning Validation
1. Navigate to Purview Studio
2. Go to **Data Map** → **Scans**
3. Verify scans are running or completed
4. Check for scan errors or warnings

### Lineage Validation
1. Navigate to Purview Studio → **Data Catalog**
2. Search for an entity (e.g., Cosmos DB container)
3. Click on the entity → **Lineage** tab
4. Verify lineage graph shows expected data flows

### Classification Validation
1. Navigate to Purview Studio → **Data Catalog**
2. Search for entities
3. Verify classifications are applied correctly
4. Check sensitivity labels

## Troubleshooting

### Common Issues

**Issue: Purview deployment fails**
- **Cause**: Insufficient permissions or quota limits
- **Solution**: 
  - Verify Owner or Contributor role on subscription
  - Check Purview resource provider is registered
  - Verify region supports Purview

**Issue: Private endpoint DNS resolution fails**
- **Cause**: Private DNS zone not linked correctly
- **Solution**: 
  - Verify VNet is linked to private DNS zones
  - Check NSG rules allow traffic to private endpoints
  - Verify DNS resolution: `nslookup <purview-account>.purview.azure.com`

**Issue: Data source scan fails**
- **Cause**: Insufficient permissions for Purview managed identity
- **Solution**: 
  - Verify Purview MI has Data Reader role on Cosmos DB
  - Check Storage Blob Data Reader role on Storage account
  - Ensure Fabric workspace grants Reader access

**Issue: Lineage not appearing**
- **Cause**: Custom lineage events not being sent correctly
- **Solution**: 
  - Verify Purview Atlas API authentication
  - Check lineage event JSON format
  - Review Application Insights logs for errors

### Debug Commands
```bash
# Check Purview account status
az purview account show \
  --name $(azd env get-value PURVIEW_ACCOUNT_NAME) \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP_NAME)

# Check private endpoint connections
az network private-endpoint list \
  --resource-group $(azd env get-value AZURE_RESOURCE_GROUP_NAME) \
  --query "[?contains(name, 'purview')].{Name:name, State:privateLinkServiceConnections[0].privateLinkServiceConnectionState.status}"

# Check role assignments
az role assignment list \
  --scope /subscriptions/$(az account show --query id -o tsv)/resourceGroups/$(azd env get-value AZURE_RESOURCE_GROUP_NAME) \
  --query "[?contains(principalType, 'ServicePrincipal') && contains(roleDefinitionName, 'Purview')]"
```

## Security Considerations

### Best Practices
1. **Always use Private Endpoints** in production for secure connectivity
2. **Enable Azure AD authentication** for Purview access
3. **Use Managed Identities** for all service-to-service authentication
4. **Implement least-privilege RBAC**: 
   - Agents: Purview Data Reader only
   - Admins: Purview Data Curator
   - Scanners: Purview Data Source Administrator
5. **Enable audit logging** for all Purview operations
6. **Regularly review** data classifications and access reports

### Compliance Frameworks Supported
- **SOC 2**: Audit trails, access controls, retention policies
- **ISO 27001**: Data classification, security controls, risk management
- **GDPR**: Data discovery, right to be forgotten, data lineage
- **HIPAA**: Audit logs, access controls, data encryption

## References
- [Microsoft Purview Documentation](https://learn.microsoft.com/azure/purview/)
- [Purview Data Catalog](https://learn.microsoft.com/azure/purview/catalog-introduction)
- [Purview Data Lineage](https://learn.microsoft.com/azure/purview/concept-data-lineage)
- [Purview Private Endpoints](https://learn.microsoft.com/azure/purview/catalog-private-link)
- [Purview REST API](https://learn.microsoft.com/rest/api/purview/)
- [Azure Agents Control Plane - Eight Pillars of Governance](/.speckit/constitution.md)

## Support and Feedback
For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review [Purview documentation](https://learn.microsoft.com/azure/purview/)
3. Open an issue in the repository
4. Contact the Azure Agents Control Plane team
