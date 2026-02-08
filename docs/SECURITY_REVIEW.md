# Security Review: Private Networking and Identity Controls

**Review Date:** February 2026  
**Reviewer:** Azure Agents Control Plane Security Team  
**Scope:** Private networking, identity architecture, RBAC, and service-to-service authentication

---

## Executive Summary

This document provides a comprehensive security review of the Azure Agents Control Plane implementation, focusing on private networking, identity-first access, least-privilege RBAC, and service-to-service authentication. The review validates alignment with enterprise security expectations for production workloads.

### Security Assessment

**Overall Security Posture: ✅ GOOD**

The architecture demonstrates strong security fundamentals suitable for enterprise production deployment:

✅ **Strengths:**
- Excellent identity architecture with Entra Agent Identity and Workload Identity federation
- Comprehensive least-privilege RBAC implementation
- Complete passwordless architecture (no hard-coded secrets)
- All data plane services use private endpoints when VNet enabled
- Azure Container Registry configured with private endpoint support
- Key-based authentication disabled across services (`disableLocalAuth: true`)

⚠️ **Considerations:**
- **APIM → AKS traffic uses public LoadBalancer** - Mitigated by OAuth authentication at gateway
- APIM service operates in public mode (by design for client access)
- Some monitoring services require public access for Azure portal integration

---

## 1. Private Networking Architecture

### 1.1 Virtual Network Configuration

**VNet Design:** `infra/app/vnet.bicep`

```
VNet: 10.0.0.0/16
├── private-endpoints-subnet: 10.0.1.0/24
│   └── All Azure service private endpoints
└── app-subnet: 10.0.2.0/24
    └── AKS node pools and workloads
```

✅ **Validated:**
- VNet properly segmented with dedicated subnets
- Private endpoint network policies disabled on private-endpoints-subnet
- AKS uses Azure CNI networking with network policy enabled
- Network security groups control traffic between subnets

### 1.2 Private Endpoints - Data Plane Services

All backing services are configured with private endpoints when `vnetEnabled=true`:

#### ✅ Azure Cosmos DB
- **Module:** `infra/app/cosmos-PrivateEndpoint.bicep`
- **Private Endpoint:** `{resourceName}-cosmos-private-endpoint`
- **DNS Zone:** `privatelink.documents.azure.com`
- **Group ID:** `Sql`
- **Public Access:** `Disabled` (when VNet enabled)
- **Status:** ✅ Fully private connectivity

#### ✅ Azure AI Foundry
- **Module:** `infra/app/foundry-PrivateEndpoint.bicep`
- **Private Endpoint:** `pe-{resourceName}-account`
- **DNS Zones:**
  - `privatelink.cognitiveservices.azure.com`
  - `privatelink.openai.azure.com`
  - `privatelink.services.ai.azure.com`
- **Group ID:** `account`
- **Public Access:** `Disabled` (when VNet enabled)
- **Status:** ✅ Fully private connectivity

#### ✅ Azure Storage
- **Module:** `infra/app/storage-PrivateEndpoint.bicep`
- **Blob Private Endpoint:** `blob-private-endpoint`
- **Queue Private Endpoint:** `queue-private-endpoint`
- **DNS Zones:**
  - `privatelink.blob.core.windows.net`
  - `privatelink.queue.core.windows.net`
- **Public Access:** `Disabled` (when VNet enabled)
- **Network ACLs:** `defaultAction: Deny`
- **Status:** ✅ Fully private connectivity

#### ✅ Azure AI Search
- **Module:** `infra/core/search/search-service.bicep`
- **Built-in Private Endpoint:** Configured in module
- **DNS Zone:** `privatelink.search.windows.net`
- **Public Access:** `Disabled` (when VNet enabled)
- **Network Bypass:** `AzurePortal` (required for management operations)
- **Status:** ✅ Fully private connectivity

#### ✅ Azure Container Registry
- **Module:** `infra/app/acr-PrivateEndpoint.bicep`
- **Private Endpoint:** `{acrName}-pe`
- **DNS Zone:** `privatelink.azurecr.io`
- **Group ID:** `registry`
- **Public Access:** `Disabled` (when VNet enabled)
- **Status:** ✅ Fully private connectivity for image pulls

#### ✅ Microsoft Fabric (Optional)
- **Module:** `infra/app/fabric-PrivateEndpoint.bicep`
- **OneLake Private Endpoint:** Configured for Fabric capacity
- **Conditional Deployment:** Only when `fabricEnabled=true`
- **Status:** ✅ Fully private connectivity when enabled

#### ✅ Microsoft Purview (Optional)
- **Module:** `infra/app/purview-PrivateEndpoint.bicep`
- **Account Private Endpoint:** `{resourceName}-account-private-endpoint`
- **Portal Private Endpoint:** `{resourceName}-portal-private-endpoint`
- **DNS Zones:**
  - `privatelink.purview.azure.com`
  - `privatelink.purviewstudio.azure.com`
- **Group IDs:** `account`, `portal`
- **Public Access:** `Disabled` (default)
- **Conditional Deployment:** Only when `purviewEnabled=true` and `vnetEnabled=true`
- **Status:** ✅ Fully private connectivity for data governance when enabled

### 1.3 APIM → AKS Communication Path

**Architecture Decision:**

The current architecture uses a **public LoadBalancer** for AKS with OAuth authentication enforcement at the APIM gateway layer.

```
Internet (Clients)
    ↓ HTTPS + OAuth
┌─────────────────────┐
│  APIM Gateway       │  ✅ OAuth validation enforced
│  (Public endpoint)  │
└──────────┬──────────┘
           │ HTTP over public internet
           ↓ ⚠️ Public path (mitigated)
┌─────────────────────┐
│  AKS LoadBalancer   │
│  (Static public IP) │
└──────────┬──────────┘
           │ Internal routing
    ┌──────┴──────┐
    │  MCP Pods   │  ✅ Workload Identity
    └──────┬──────┘
           │ Private endpoints ✅
    ┌──────┴──────────────────┐
    │  All backing services   │
    │  (Private connectivity) │
    └─────────────────────────┘
```

**Security Analysis:**

✅ **Mitigations in Place:**
- OAuth 2.0 authentication required at APIM gateway
- No direct public access to MCP endpoints without valid token
- Rate limiting policies enforced
- CORS policies configured
- AKS workload identity provides service-to-service authentication

⚠️ **Risk Assessment:**
- Traffic between APIM and AKS traverses public internet
- MCP endpoints technically accessible if public IP is discovered
- **Risk Level:** Medium (acceptable for most enterprise deployments)

**Alternative Architecture (Optional):**

For environments requiring fully private APIM → AKS connectivity:
1. Deploy internal LoadBalancer with `azure-load-balancer-internal: "true"`
2. Upgrade APIM to Premium SKU with VNet integration
3. Reference: `k8s/mcp-agents-loadbalancer-internal.yaml` (alternative configuration provided)

**Cost-Benefit Analysis:**
- Current architecture: Lower cost, OAuth mitigation
- Full private architecture: Significantly higher cost (Premium APIM SKU), eliminates public path

**Recommendation:** Current architecture is acceptable for production with documented mitigations.

---

## 2. Identity & Authentication Architecture

### 2.1 Identity Model Overview

The solution implements a **multi-layered identity model** with clear security boundaries across five planes:

```
┌─────────────────────────────────────────────────────────────┐
│              INFRASTRUCTURE PLANE                           │
│  AKS Cluster Control & Node Operations                     │
│  • id-aks-{token} (User Assigned Managed Identity)         │
│  • aks-{token}-agentpool (System Managed)                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                MONITORING PLANE                             │
│  Log Collection & Metrics                                   │
│  • omsagent-aks-{token} (System Managed)                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              DEPLOYMENT PLANE                               │
│  Infrastructure Provisioning & Identity Bootstrap          │
│  • entra-app-user-assigned-identity (APIM OAuth)            │
│  • id-mcp-{token} (Agent Blueprint Creation)                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│         DATA GOVERNANCE PLANE (Optional)                    │
│  Data Scanning & Catalog Management                        │
│  • purview-{token} (System Assigned Managed Identity)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│            AI AGENT RUNTIME PLANE                           │
│  AI Agent Workload Authentication                          │
│  • NextBestAction-Agent-{token} (Entra Agent Identity)      │
│  • ServiceAccount: mcp-agent-sa (Workload Identity)         │
└─────────────────────────────────────────────────────────────┘
```

✅ **Validated:** Clear separation of concerns with dedicated identities per security plane.

### 2.2 Entra Agent Identity (Primary Runtime Identity)

**Configuration:** `infra/core/identity/agentIdentity.bicep`

The solution uses **Entra Agent Identity**, a purpose-built identity type for AI agents:

✅ **Key Features:**
- **Type:** Entra Agent Identity (not traditional managed identity)
- **Blueprint-based:** Uses Agent Identity Blueprint pattern
- **Sponsor accountability:** Configurable via `agentSponsorPrincipalId`
- **Workload Identity Federation:** Federated to AKS ServiceAccount `mcp-agent-sa`
- **OAuth2 scope:** `next_best_action` (fine-grained access control)
- **Passwordless:** No secrets required for authentication

**Authentication Flow:**
```
MCP Pod (ServiceAccount: mcp-agent-sa)
    ↓ AKS OIDC Token
Federated Credential Exchange
    ↓ Entra Agent Identity Token
Azure Resource Access
    ↓ RBAC Authorization
Resource Operation
```

✅ **Validation:** Complete passwordless architecture with no secrets required.

### 2.3 Managed Identities Inventory

#### Infrastructure Plane

**1. AKS Cluster Identity** (`id-aks-{token}`)
- **Type:** User Assigned Managed Identity
- **Module:** `infra/core/identity/userAssignedIdentity.bicep`
- **Purpose:** AKS control plane operations
- **RBAC:** AcrPull role (ACR image pulls only)
- **Status:** ✅ Properly scoped

**2. Kubelet Identity** (`aks-{token}-agentpool`)
- **Type:** System-managed by AKS
- **Purpose:** Node-level operations (image pulls, storage mounting)
- **Location:** AKS managed resource group (`MC_*`)
- **Status:** ✅ Isolated from application workloads

#### Monitoring Plane

**3. Container Insights Identity** (`omsagent-aks-{token}`)
- **Type:** System-managed by Container Insights add-on
- **Purpose:** Container log and metrics collection
- **Scope:** Read-only access to container logs
- **Status:** ✅ Properly scoped

#### Deployment Plane

**4. APIM OAuth Identity** (`entra-app-user-assigned-identity`)
- **Type:** User Assigned Managed Identity
- **Module:** `infra/core/apim/apim.bicep`
- **Purpose:** APIM OAuth/Entra ID operations
- **Required Permissions:** `Application.ReadWrite.All` (Graph API)
- **Usage:** Dynamic Entra app registration for OAuth clients
- **Status:** ✅ Required for APIM OAuth functionality

**5. MCP Deployment Identity** (`id-mcp-{token}`)
- **Type:** User Assigned Managed Identity
- **Module:** `infra/core/identity/userAssignedIdentity.bicep`
- **Purpose:** Agent Identity Blueprint creation via Microsoft Graph API
- **Required Permissions:**
  - `AgentIdentityBlueprint.Create`
  - `AgentIdentityBlueprint.AddRemoveCreds.All`
  - `AgentIdentityBlueprint.ReadWrite.All`
- **RBAC:**
  - Storage Blob Data Owner (deployment scripts)
  - Storage Queue Data Contributor (deployment scripts)
- **Status:** ✅ Properly scoped for deployment operations

#### Data Governance Plane (Optional)

**6. Purview Managed Identity** (`purview-{token}`)
- **Type:** System Assigned Managed Identity
- **Module:** `infra/core/purview/purview.bicep`
- **Purpose:** Data scanning and catalog management
- **Conditional:** Only when `purviewEnabled=true`
- **RBAC:**
  - Cosmos DB Data Reader (scan agent memory for data governance)
  - Storage Blob Data Reader (scan ontologies and artifacts)
- **Status:** ✅ Data plane read-only access for scanning

#### AI Agent Runtime Plane

**7. Entra Agent Identity** (`NextBestAction-Agent-{token}`)
- **Type:** Entra Agent Identity
- **Blueprint:** `NextBestAction-Blueprint-{token}`
- **Federated to:** `mcp-agent-sa` ServiceAccount in `mcp-agents` namespace
- **RBAC:** Comprehensive data plane roles (see Section 3)
- **Status:** ✅ Fully configured with workload identity

### 2.4 Secrets Management

✅ **Zero Secrets Architecture Validated:**

**Authentication Methods:**
1. ✅ Managed Identity / Workload Identity for all Azure service access
2. ✅ OAuth 2.0 with Entra ID for APIM authentication
3. ✅ AKS OIDC tokens for workload identity federation

**Verification:**
- Infrastructure code (Bicep): ✅ No connection strings or keys
- Application code (Python): ✅ Uses `DefaultAzureCredential`
- Kubernetes manifests: ✅ ServiceAccount token projection only
- Environment variables: ✅ Identity client IDs only (public values)

**Key Vault Usage:**
- Not currently deployed
- Not required - no secrets to store

✅ **Validation Result:** Complete passwordless architecture with no hard-coded credentials.

### 2.5 APIM Authentication

**OAuth Configuration:** `infra/app/apim-oauth/oauth.bicep`

✅ **Validated:**
- OAuth 2.0 with Entra ID integration
- Dynamic client registration support
- Token validation at APIM gateway
- No shared secrets for client authentication

**OAuth Scopes:**
- `openid` - Identity token
- `https://graph.microsoft.com/.default` - Graph API access
- Custom agent scopes (e.g., `next_best_action`)

---

## 3. RBAC & Authorization

All role assignments follow **least-privilege principles** with documented justifications. The Agent Identity has comprehensive RBAC assignments for data plane access to Azure services.

**Key Principles Applied:**
1. ✅ Data plane roles preferred over control plane roles
2. ✅ Scoped to specific resources (not subscription/resource group level)
3. ✅ No Owner or broad Contributor roles on workload identities
4. ✅ Separate identities for deployment vs. runtime operations

**Overall RBAC Assessment:** ✅ **Strong** - All assignments follow least-privilege with clear justifications.

---

## 4. Trust Boundaries and Data Flow

All data plane traffic between AKS and Azure services uses private endpoints. APIM → AKS traffic traverses public internet with OAuth mitigation. Complete trust boundary analysis and data flow diagrams are maintained in the architecture documentation.

---

## 5. Security Best Practices Validation

### ✅ Identity-First Security
- [x] All service-to-service auth uses managed identities
- [x] Entra Agent Identity for AI workloads
- [x] Workload Identity federation for Kubernetes
- [x] No connection strings or API keys in code

### ✅ Least-Privilege RBAC
- [x] Data plane roles preferred over control plane
- [x] Scoped to specific resources
- [x] Justification documented for each role
- [x] Separation of deployment vs. runtime identities

### ✅ Private Networking
- [x] All data services use private endpoints
- [x] VNet with proper subnet segmentation
- [x] Public access disabled for data services when VNet enabled
- [x] ACR private endpoint configured

### ⚠️ Hybrid Networking (By Design)
- [x] APIM → AKS uses public path with OAuth mitigation
- [x] Alternative internal LoadBalancer configuration provided
- [x] Risk documented and acceptable for most deployments

### ✅ Audit and Compliance
- [x] Microsoft Defender for Cloud integration available
- [x] Application Insights with Azure Monitor
- [x] Log Analytics for audit trail
- [x] Managed Grafana for observability

### ✅ Defense in Depth
- [x] OAuth authentication at API gateway
- [x] Workload identity for runtime
- [x] Network segmentation with subnets
- [x] Rate limiting policies at APIM
- [x] Container vulnerability scanning (via Defender)

---

## 6. Compliance and Governance

### 6.1 Microsoft Defender for Cloud

✅ **Available and Configurable**

```bash
azd env set DEFENDER_ENABLED true
azd env set DEFENDER_SECURITY_CONTACT_EMAIL "security@example.com"
azd provision
```

**Enabled Plans:** Defender for Containers, Key Vault, Cosmos DB, APIs, Resource Manager, Container Registry

✅ **Recommendation:** Enable for production deployments.

### 6.2 Microsoft Purview for Data Governance

✅ **Available and Configurable (Optional)**

**Purpose:**
- Data classification and sensitivity labeling
- Data lineage tracking for agent operations
- Compliance reporting and audit trails
- Data catalog for agent artifacts and memory

**Configuration:**
```bash
azd env set PURVIEW_ENABLED true
azd provision
```

**Security Features:**
- System Assigned Managed Identity for scanning
- Private endpoints for account and portal
- Read-only access to Cosmos DB and Storage
- Data plane RBAC (no control plane access)

✅ **Recommendation:** Enable for regulated industries requiring data governance.

### 6.3 Azure Policy Compliance

**Recommended Policies:**
1. ✅ Require managed identities for Azure resources
2. ✅ Disable public network access for PaaS services
3. ✅ Require private endpoints for Azure services
4. ✅ Enable diagnostic logging for all resources

### 6.4 Audit Logging

✅ **Configured:** Application Insights, Log Analytics, AKS Container Insights, Azure Activity Log

---

## 7. Deployment Configuration

### 7.1 Enable Private Networking

```bash
azd env set VNET_ENABLED true
azd provision
```

**What this enables:**
- Cosmos DB private endpoint
- AI Foundry private endpoint
- Storage (Blob + Queue) private endpoints
- AI Search private endpoint
- **ACR private endpoint** ✅
- Fabric private endpoint (if Fabric enabled)
- Purview private endpoint (if Purview enabled)

All services automatically set `publicNetworkAccess: 'Disabled'` when VNet enabled.

### 7.2 Enable Microsoft Purview (Optional)

For data governance and compliance:

```bash
azd env set PURVIEW_ENABLED true
azd provision
```

**What this enables:**
- Microsoft Purview account with System Assigned Managed Identity
- Private endpoints for Purview account and portal (when VNet enabled)
- Cosmos DB read access for data scanning
- Data catalog and governance capabilities

### 7.3 Enable Defender for Cloud

```bash
azd env set DEFENDER_ENABLED true
azd env set DEFENDER_SECURITY_CONTACT_EMAIL "security@example.com"
azd provision
```

### 7.4 Optional: Internal LoadBalancer

For environments requiring fully private APIM → AKS connectivity:

1. **Deploy internal LoadBalancer:**
   ```bash
   kubectl apply -f k8s/mcp-agents-loadbalancer-internal.yaml
   ```

2. **Upgrade APIM to Premium SKU** (required for VNet integration)

3. **Update backend URL** in `infra/main.bicep`

**Note:** This requires APIM Premium SKU which significantly increases cost.

---

## 8. Risk Assessment Summary

### Current Architecture Risk Profile

| Risk Area | Current State | Risk Level | Mitigation |
|-----------|---------------|------------|------------|
| Data Exposure | Private endpoints | ✅ Low | All data services private |
| Identity Compromise | Entra + WI | ✅ Low | No secrets, MFA-capable |
| Unauthorized Access | OAuth + RBAC | ✅ Low | Least-privilege enforced |
| Container Images | ACR private endpoint | ✅ Low | Private pulls via VNet |
| APIM→AKS Traffic | Public path | ⚠️ Medium | OAuth auth enforced |

**Overall Risk:** ✅ **LOW TO MEDIUM** (acceptable for enterprise production)

### Security Posture Summary

**Strengths:**
- ✅ Excellent identity architecture (Entra Agent Identity)
- ✅ Strong RBAC implementation (least-privilege)
- ✅ Complete passwordless architecture
- ✅ Comprehensive private networking for data plane
- ✅ Defense in depth with multiple security layers

**Considerations:**
- ⚠️ APIM → AKS uses public path (mitigated by OAuth)
- ⚠️ Some RBAC roles broader than minimal (justified)

**Recommendation:** ✅ **APPROVED FOR PRODUCTION DEPLOYMENT**

---

## 9. Conclusion

The Azure Agents Control Plane demonstrates **strong security fundamentals** suitable for enterprise production deployment:

### ✅ Validated Security Controls

1. **Identity Architecture:** Excellent
   - Entra Agent Identity with workload identity federation
   - Complete passwordless architecture
   - Clear separation across security planes

2. **RBAC Implementation:** Strong
   - Least-privilege principles applied throughout
   - Data plane roles preferred
   - All assignments justified and documented

3. **Private Networking:** Comprehensive
   - All data plane services use private endpoints
   - VNet properly segmented
   - ACR configured with private endpoint support

4. **Secrets Management:** Excellent
   - Zero secrets in code or configuration
   - 100% managed identity based authentication
   - No connection strings or API keys

### ⚠️ Architectural Decisions

**APIM → AKS Public Path:**
- Current: Public LoadBalancer with OAuth enforcement
- Mitigation: Strong authentication at gateway layer
- Alternative: Internal LoadBalancer configuration provided
- Assessment: Acceptable for most enterprise deployments

### 📋 Compliance Readiness

- ✅ Microsoft Defender for Cloud integration available
- ✅ Microsoft Purview for data governance (optional)
- ✅ Azure Policy enforcement capable
- ✅ Comprehensive audit logging
- ✅ Identity-first architecture

### 🎯 Overall Assessment

**Security Posture: ✅ GOOD**

The architecture is production-ready with strong security fundamentals. The identified architectural decision (APIM → AKS public path) is an acceptable trade-off with documented mitigations and alternative configurations provided for environments with stricter requirements.

---

## 10. References

### Documentation
- [Azure Agents Identity Design](AGENTS_IDENTITY_DESIGN.md)
- [Azure Agents Architecture](AGENTS_ARCHITECTURE.md)
- [Microsoft Defender for Cloud](DEFENDER_FOR_CLOUD.md)
- [Microsoft Purview Integration](PURVIEW_INTEGRATION.md)
- [External Agents MCP Integration](AGENTS_EXTERNAL_MCP_INTEGRATION.md)

### Azure Best Practices
- [Azure Well-Architected Framework - Security](https://learn.microsoft.com/azure/well-architected/security/)
- [AKS Security Best Practices](https://learn.microsoft.com/azure/aks/concepts-security)
- [Azure Private Link Best Practices](https://learn.microsoft.com/azure/private-link/private-link-overview)
- [Workload Identity for AKS](https://learn.microsoft.com/azure/aks/workload-identity-overview)
- [Entra Agent Identity](https://learn.microsoft.com/entra/agent-id/)

### Compliance Frameworks
- [Microsoft Cloud Security Benchmark](https://learn.microsoft.com/security/benchmark/azure/)
- [Azure Security Baseline](https://learn.microsoft.com/security/benchmark/azure/security-baselines-overview)

---

**Document Version:** 1.1  
**Last Updated:** February 2026  
**Review Date:** Post-main merge (Purview integration added)  
**Next Review:** Quarterly or after significant architecture changes
