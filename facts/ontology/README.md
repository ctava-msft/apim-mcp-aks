# Fabric IQ Ontology Data

This directory contains ontology definitions and sample data for integration with Microsoft Fabric IQ. These ontologies provide AI-grounded facts for the MCP agent to use in reasoning across three domains.

## Overview

[Microsoft Fabric IQ](https://learn.microsoft.com/en-us/fabric/iq/overview) unifies data across OneLake sources and organizes it according to your business language. The ontologies in this folder define:

- **Entity Types**: Business objects with properties (Customer, Pipeline, User, etc.)
- **Relationship Types**: Connections between entities (HAS_TRANSACTION, DEPLOYS_TO, etc.)
- **Facts**: AI-grounded observations, predictions, and derived insights

## Domain Ontologies

### 1. Customer Churn Analysis (`customer_churn_ontology.json`)

**Use Case**: Analyze customer churn data and create predictive models to identify at-risk customers

**Entity Types**:
- `Customer` - Customer account with subscription and engagement data
- `Transaction` - Purchase, renewal, upgrade, downgrade, refund records
- `EngagementEvent` - Login, feature usage, support ticket activities
- `ChurnPrediction` - AI-generated churn probability and recommendations

**Key Facts**:
- Customer churn risk predictions with confidence scores
- Segment-based retention insights
- High-risk customer identification with key indicators

**Sample Data**:
- 25 customers across enterprise, business, professional, starter, and trial segments
- Transaction history with various transaction types
- Engagement events tracking feature usage

### 2. CI/CD Pipeline (`cicd_pipeline_ontology.json`)

**Use Case**: Set up CI/CD pipeline for deploying microservices to Kubernetes

**Entity Types**:
- `Pipeline` - CI/CD pipeline definition with configuration
- `PipelineRun` - Individual execution with status and metrics
- `Deployment` - Kubernetes deployment event
- `Cluster` - Target Kubernetes cluster

**Failure Categories**:
- `build_error` - Compilation failures
- `test_failure` - Unit/integration test failures
- `security_scan` - Vulnerability detections
- `deployment_error` - Kubernetes deployment issues
- `infrastructure` - Connectivity/resource issues

**Key Facts**:
- Pipeline health observations with success rates
- Failure analysis with root cause categorization
- Deployment health recommendations

**Sample Data**:
- 6-8 pipelines for different microservices
- 20+ pipeline runs per pipeline with success/failure mix
- Deployment events with health check status

### 3. User Management (`user_management_ontology.json`)

**Use Case**: Design REST API for user management with authentication

**Entity Types**:
- `User` - User account with roles and permissions
- `Session` - Authentication session
- `AuthEvent` - Login, logout, password change events
- `AccessLog` - Resource access audit trail
- `Role` - Authorization role definition

**Security Features**:
- MFA status tracking
- Failed login attempt monitoring
- Risk score calculation
- Security alert generation

**Key Facts**:
- User activity observations
- Security alerts for suspicious patterns
- MFA adoption recommendations

**Sample Data**:
- 15-20 users with various roles
- Authentication events with success/failure
- Access logs with resource paths and response codes

## Integration with Fabric IQ

### Loading into OneLake

1. **Create Lakehouse**: Create a new Lakehouse in your Fabric workspace
2. **Upload Files**: Upload the JSON files to the Lakehouse files section
3. **Create Tables**: Convert JSON to Delta tables for querying

```python
# Example: Load ontology data into Spark DataFrame
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Load customer ontology
customer_df = spark.read.json("abfss://...@onelake.dfs.fabric.microsoft.com/.../customer_churn_ontology.json")

# Extract entities
entities_df = customer_df.select("sample_entities").explode("sample_entities")
```

### Creating Ontology in Fabric IQ

1. Navigate to Fabric IQ in your workspace
2. Create a new Ontology item
3. Define entity types matching the JSON schemas
4. Add relationship types for graph traversal
5. Bind to OneLake data sources

### Connecting to MCP Agent

The MCP agent uses the `FactsMemory` class to integrate with Fabric IQ:

```python
from memory import FactsMemory

facts_memory = FactsMemory(
    fabric_endpoint=os.getenv("FABRIC_ENDPOINT"),
    workspace_id=os.getenv("FABRIC_WORKSPACE_ID"),
    ontology_name="agent-ontology",
)
```

## MCP Tools for Facts Retrieval

The following MCP tools are available for querying facts:

| Tool | Description |
|------|-------------|
| `search_facts` | Semantic search across all domains |
| `get_customer_churn_facts` | Customer churn analysis facts |
| `get_pipeline_health_facts` | CI/CD pipeline health facts |
| `get_user_security_facts` | User security and access facts |
| `cross_domain_analysis` | Cross-domain reasoning |
| `get_facts_memory_stats` | Facts memory statistics |

## Environment Variables

Configure these environment variables for Fabric IQ integration:

```bash
FABRIC_ENDPOINT=https://<workspace>.fabric.microsoft.com
FABRIC_WORKSPACE_ID=<workspace-id>
FABRIC_ONTOLOGY_NAME=agent-ontology
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CompositeMemory                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │  Short-Term Memory  │  │   Long-Term Memory  │  │    Facts Memory     │ │
│  │    (CosmosDB)       │  │    (AI Search)      │  │   (Fabric IQ)       │ │
│  │  - Session context  │  │  - Task instructions│  │  - Ontology facts   │ │
│  │  - TTL-based        │  │  - Cross-session    │  │  - Cross-domain     │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Domain Ontologies                                 │
├──────────────────┬───────────────────────┬──────────────────────────────────┤
│    Customer      │       DevOps          │        User Management           │
│  Churn Analysis  │    CI/CD Pipeline     │       Authentication             │
├──────────────────┼───────────────────────┼──────────────────────────────────┤
│  - Customer      │  - Pipeline           │  - User                          │
│  - Transaction   │  - PipelineRun        │  - Session                       │
│  - Engagement    │  - Deployment         │  - AuthEvent                     │
│  - ChurnPredict  │  - Cluster            │  - AccessLog                     │
└──────────────────┴───────────────────────┴──────────────────────────────────┘
```

## References

- [Fabric IQ Overview](https://learn.microsoft.com/en-us/fabric/iq/overview)
- [Ontology in Fabric IQ](https://learn.microsoft.com/en-us/fabric/iq/ontology/overview)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
