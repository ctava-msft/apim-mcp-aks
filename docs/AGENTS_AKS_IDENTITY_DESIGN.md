# Identity Design for MCP Agents on AKS

## Overview

This document describes the identity architecture design for the MCP (Model Context Protocol) agents running on Azure Kubernetes Service (AKS). The design implements a multi-layered identity model with clear security boundaries between infrastructure, deployment, monitoring, and AI agent runtime planes.

## Design Goals

1. **Least Privilege** - Each identity has only the permissions required for its specific function
2. **Security Isolation** - Clear separation between infrastructure, deployment, and runtime identities
3. **AI Agent Governance** - Purpose-built identity type (Entra Agent ID) for AI agent workloads
4. **Audit & Accountability** - Sponsor-based accountability for autonomous AI agents
5. **Workload Identity** - Passwordless authentication via AKS workload identity federation

---

## Identity Architecture

### Identity Summary

| Identity | Plane | Purpose | Type |
|----------|-------|---------|------|
| `id-aks-*` | Infrastructure | AKS Cluster Control | User Assigned Managed Identity |
| `aks-*-agentpool` | Infrastructure | Kubelet/Node Pool | System Managed (AKS) |
| `omsagent-aks-*` | Monitoring | Container Insights | System Managed (AKS) |
| `entra-app-user-assigned-identity` | Deployment | APIM OAuth | User Assigned Managed Identity |
| `id-mcp-*` | Deployment | Agent Identity Bootstrap | User Assigned Managed Identity |
| **Entra Agent Identity** | **AI Agent Runtime** | **MCP Workload** | **Entra Agent Identity** |

### Security Planes

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE PLANE                         │
│  (AKS Control, Node Operations, ACR)                           │
│                                                                 │
│  • id-aks-* (Cluster Identity)                                 │
│  • aks-*-agentpool (Kubelet Identity)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MONITORING PLANE                             │
│  (Log Collection, Metrics, Diagnostics)                         │
│                                                                 │
│  • omsagent-aks-* (Container Insights)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PLANE                             │
│  (Infrastructure Provisioning, Identity Bootstrap)             │
│                                                                 │
│  • id-mcp-* (Agent Identity Blueprint Creation)                │
│  • entra-app-user-assigned-identity (APIM OAuth)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    AI AGENT RUNTIME PLANE                       │
│  (AI Agent Workload Authentication)                            │
│                                                                 │
│  • Entra Agent Identity (NextBestAction-Agent-*)               │
│  • ServiceAccount: mcp-agent-sa                                │
│  • OAuth2 Scope: next_best_action                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure Plane Identities

### 1. AKS Cluster Identity (`id-aks-*`)

| Property | Value |
|----------|-------|
| **Type** | User Assigned Managed Identity |
| **Created By** | `aksUserAssignedIdentity` module in `main.bicep` |
| **Naming Pattern** | `${abbrs.managedIdentityUserAssignedIdentities}aks-${resourceToken}` |

**Purpose:**
- AKS cluster control plane operations
- Azure Container Registry (ACR) image pulls
- Load balancer and public IP management
- Network resource configuration

**Role Assignments:**
| Resource | Role | Purpose |
|----------|------|---------|
| Azure Container Registry | AcrPull | Pull Docker images |

**Used By:**
- AKS Cluster (`aks-cluster.bicep`) - `identity.userAssignedIdentities`
- ACR Role Assignment (`acr-role-assignment.bicep`)

---

### 2. Kubelet Identity (`aks-*-agentpool`)

| Property | Value |
|----------|-------|
| **Type** | User Assigned Managed Identity |
| **Created By** | AKS (Automatically) |
| **Location** | AKS Managed Resource Group (`MC_*`) |

**Purpose:**
- Node-level authentication to Azure
- Container image pulls via kubelet
- Persistent volume operations (Azure Disk/File)
- Node pool scaling operations

**Note:** This identity is system-managed by AKS and cannot be customized or replaced.

---

## Monitoring Plane Identity

### 3. Container Insights Identity (`omsagent-aks-*`)

| Property | Value |
|----------|-------|
| **Type** | User Assigned Managed Identity |
| **Created By** | AKS Container Insights Add-on (Automatically) |
| **Location** | AKS Managed Resource Group (`MC_*`) |

**Purpose:**
- Container log collection (stdout/stderr)
- Performance metrics collection
- Log Analytics workspace integration
- Prometheus metrics scraping

**Configuration in Bicep:**
```bicep
addonProfiles: {
  omsagent: {
    enabled: true
    config: {
      logAnalyticsWorkspaceResourceID: logAnalyticsWorkspaceId
    }
  }
}
```

**Note:** This identity is system-managed by the Container Insights add-on.

---

## Deployment Plane Identities

### 4. APIM OAuth Identity (`entra-app-user-assigned-identity`)

| Property | Value |
|----------|-------|
| **Type** | User Assigned Managed Identity |
| **Created By** | APIM Module (`apim.bicep`) |

**Purpose:**
- Azure API Management (APIM) Entra ID operations
- Microsoft Graph API authentication for OAuth/OIDC flows
- Dynamic Entra app registration for MCP clients
- OAuth token exchange handling

**Used By:**
- APIM OAuth policies (`oauth.bicep`, `token.policy.xml`)
- Entra app registration module (`entra-app.bicep`)

**Required Permissions:**
- `Application.ReadWrite.All` (Graph API)
- OAuth consent flow permissions

---

### 5. Deployment Scripts Identity (`id-mcp-*`)

| Property | Value |
|----------|-------|
| **Type** | User Assigned Managed Identity |
| **Created By** | `mcpUserAssignedIdentity` module in `main.bicep` |
| **Naming Pattern** | `${abbrs.managedIdentityUserAssignedIdentities}mcp-${resourceToken}` |

**Purpose:**
- Agent Identity Blueprint creation via Microsoft Graph API
- Bootstrap identity for Entra Agent Identity provisioning
- Federated credential configuration for blueprints
- Backward compatibility with legacy deployments

**Role Assignments:**
| Resource | Role | Purpose |
|----------|------|---------|
| Storage Account | Blob Data Owner | Deployment script access |
| Storage Account | Queue Data Contributor | Deployment script access |

**Used By:**
- Agent Identity Blueprint deployment scripts (`agentIdentityBlueprint.bicep`)
- Agent Identity creation scripts (`agentIdentity.bicep`)
- Legacy ServiceAccount `mcp-agents-sa` (backward compatibility)

---

## AI Agent Runtime Plane

### 6. Entra Agent Identity (Next Best Action Agent)

| Property | Value |
|----------|-------|
| **Type** | Entra Agent Identity |
| **Created By** | `agentIdentity.bicep` via Microsoft Graph API |
| **Display Name** | `NextBestAction-Agent-{resourceToken}` |
| **Blueprint** | `NextBestAction-Blueprint-{resourceToken}` |

**What is Entra Agent ID?**

Entra Agent ID is a purpose-built identity type for AI agents. Unlike traditional managed identities or service principals:

1. **Purpose-built for AI agents** - Designed for autonomous AI agent representation in Entra ID
2. **Blueprint-based** - Uses Agent Identity Blueprint to define capabilities and security boundaries
3. **Sponsor accountability** - User/group sponsors are accountable for agent actions
4. **Fine-grained scopes** - OAuth2 scopes specific to agent operations
5. **Workload identity integration** - Compatible with AKS workload identity federation

#### Agent Identity Blueprint

The blueprint defines the template for creating agent identities:

| Property | Value |
|----------|-------|
| **Display Name** | `NextBestAction-Blueprint-{resourceToken}` |
| **Identifier URI** | `api://{blueprintAppId}` |
| **OAuth2 Scope** | `next_best_action` |
| **Federated Credential** | Linked to `id-mcp-*` managed identity |

**Created By:** `agentIdentityBlueprint.bicep`

#### Agent Identity

The actual identity used by MCP agent workloads:

| Property | Value |
|----------|-------|
| **Display Name** | `NextBestAction-Agent-{resourceToken}` |
| **Blueprint** | Linked to Agent Identity Blueprint |
| **Sponsors** | Configurable via `agentSponsorPrincipalId` parameter |

**Created By:** `agentIdentity.bicep`

#### AKS Workload Identity Federation

Enables Kubernetes pods to authenticate as the Agent Identity:

| Property | Value |
|----------|-------|
| **Service Account** | `mcp-agent-sa` in `mcp-agents` namespace |
| **OIDC Issuer** | AKS cluster's OIDC issuer URL |
| **Subject** | `system:serviceaccount:mcp-agents:mcp-agent-sa` |

**Created By:** `aksFederatedCredential.bicep`

#### Role Assignments

Comprehensive role assignments for the Agent Identity:

| Resource | Role | Purpose |
|----------|------|---------|
| **CosmosDB** | Data Contributor | Read/write tasks, plans, short-term memory |
| **Azure AI Search** | Index Data Contributor | Read/write search indexes (Foundry IQ) |
| **Azure AI Search** | Service Contributor | Manage indexes and knowledge bases |
| **Storage Account** | Blob Data Owner | Access ontologies and snippets |
| **Storage Account** | Queue Data Contributor | Message processing |
| **Azure AI Foundry** | OpenAI User | Access GPT-5.2-chat model |
| **Azure AI Foundry** | OpenAI Contributor | Manage model deployments |
| **Application Insights** | Metrics Publisher | Publish monitoring metrics |

**Created By:** `agent-RoleAssignments.bicep`

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Azure Subscription                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              rg-apim-mcp-aks (Primary Resource Group)               │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                     │    │
│  │  ┌──────────────────────┐    ┌──────────────────────┐              │    │
│  │  │  entra-app-user-     │    │    id-aks-...        │              │    │
│  │  │  assigned-identity   │    │  (AKS Cluster ID)    │              │    │
│  │  └──────────┬───────────┘    └──────────┬───────────┘              │    │
│  │             │                           │                          │    │
│  │             ▼                           ▼                          │    │
│  │  ┌──────────────────────┐    ┌──────────────────────┐              │    │
│  │  │   Azure API Mgmt    │    │     AKS Cluster      │              │    │
│  │  │   (OAuth/MCP)       │    │   (Control Plane)    │              │    │
│  │  └──────────────────────┘    └──────────────────────┘              │    │
│  │                                         │                          │    │
│  │                                         │                          │    │
│  │  ┌──────────────────────┐               │                          │    │
│  │  │    id-mcp-...        │───────────────┘                          │    │
│  │  │  (Deployment Scripts)│                                          │    │
│  │  └──────────┬───────────┘                                          │    │
│  │             │                                                      │    │
│  │             │ Creates via Graph API                                │    │
│  │             ▼                                                      │    │
│  │  ┌──────────────────────┐                                          │    │
│  │  │  Entra Agent Identity │  (Primary for AI Agents)               │    │
│  │  │  (NextBestAction-*)   │                                         │    │
│  │  └──────────┬───────────┘                                          │    │
│  │             │                                                      │    │
│  │             │ Workload Identity Federation                         │    │
│  │             ▼                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────┐     │    │
│  │  │                    MCP Server Pod                         │     │    │
│  │  │  ServiceAccount: mcp-agent-sa                            │     │    │
│  │  │  Authenticates to: CosmosDB, Storage, AI Search, Foundry │     │    │
│  │  └──────────────────────────────────────────────────────────┘     │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │     MC_rg-apim-mcp-aks_aks-..._eastus2 (AKS Managed RG)            │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │                                                                     │    │
│  │  ┌──────────────────────┐    ┌──────────────────────┐              │    │
│  │  │  aks-...-agentpool   │    │   omsagent-aks-...   │              │    │
│  │  │  (Kubelet Identity)  │    │  (Monitor Agent)     │              │    │
│  │  └──────────┬───────────┘    └──────────┬───────────┘              │    │
│  │             │                           │                          │    │
│  │             ▼                           ▼                          │    │
│  │  ┌──────────────────────┐    ┌──────────────────────┐              │    │
│  │  │   AKS Node Pool      │    │  Container Insights  │              │    │
│  │  │   (VMSS Nodes)       │    │  (Log Collection)    │              │    │
│  │  └──────────────────────┘    └──────────────────────┘              │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment

### Parameters

The following parameters configure the Agent Identity:

```json
{
  "agentIdentityEnabled": { "value": true },
  "agentBlueprintDisplayName": { "value": "" },
  "agentIdentityDisplayName": { "value": "" },
  "agentSponsorPrincipalId": { "value": "<user-object-id>" }
}
```

### Deployment Steps

1. **Provision Infrastructure**
   ```bash
   azd provision
   ```

2. **Apply Kubernetes Deployment**
   ```bash
   # Get outputs and substitute variables
   azd env get-values > .env
   
   # Apply the deployment
   envsubst < k8s/mcp-agents-deployment.yaml | kubectl apply -f -
   ```

3. **Verify Deployment**
   ```bash
   # Check the service account
   kubectl get pods -n mcp-agents -o jsonpath='{.items[*].spec.serviceAccountName}'
   # Expected: mcp-agent-sa
   
   # Check workload identity annotation
   kubectl get sa mcp-agent-sa -n mcp-agents -o yaml | grep azure.workload.identity
   ```

### Required Permissions

To create Agent Identity Blueprints, the deploying identity requires:

| Permission | Scope |
|------------|-------|
| `AgentIdentityBlueprint.Create` | Create blueprints |
| `AgentIdentityBlueprint.AddRemoveCreds.All` | Add federated credentials |
| `AgentIdentityBlueprint.ReadWrite.All` | Configure identifier URIs and scopes |

These Microsoft Graph API permissions must be granted by a Global Administrator or Privileged Role Administrator.

---

## Security Considerations

1. **Sponsorship** - Always configure a sponsor for agent identities to ensure accountability
2. **Least Privilege** - Role assignments follow least-privilege principles
3. **Scope Isolation** - Each agent identity has its own OAuth2 scope
4. **Audit Trail** - Agent identities provide enhanced audit capabilities
5. **Separation of Concerns** - Deployment scripts use `id-mcp-*`, runtime uses Agent Identity

---

## Future Considerations

1. **Per-Agent Identities** - Create separate agent identities for different agent types (customer churn, pipeline, etc.)
2. **Scope Expansion** - Add granular scopes as agent capabilities grow
3. **Sponsor Rotation** - Implement sponsor rotation for long-running agents
4. **Identity Lifecycle** - Implement automated cleanup of unused agent identities
5. **Remove Legacy** - Once fully validated, remove the legacy `mcp-agents-sa` ServiceAccount

---

## References

- [Entra Agent ID Documentation](https://learn.microsoft.com/en-us/entra/agent-id/)
- [Create Agent Identity Blueprint](https://learn.microsoft.com/en-us/entra/agent-id/identity-platform/create-blueprint)
- [Create Agent Identities](https://learn.microsoft.com/en-us/entra/agent-id/identity-platform/create-delete-agent-identities)
- [AKS Workload Identity](https://learn.microsoft.com/en-us/azure/aks/workload-identity-overview)

