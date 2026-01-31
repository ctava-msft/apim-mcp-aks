"""
Memory Provider Module for MCP Server
Provides short-term (CosmosDB), long-term (AI Search, FoundryIQ), and facts (Fabric IQ) memory abstractions

Architecture:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CompositeMemory                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐ │
│  │  Short-Term Memory  │  │   Long-Term Memory  │  │     Facts Memory        │ │
│  │    (CosmosDB)       │  │   (AI Search)       │  │    (Fabric IQ)          │ │
│  │  - Session-based    │  │  - Persistent       │  │  - Ontology-grounded    │ │
│  │  - TTL support      │  │  - Cross-session    │  │  - Cross-domain         │ │
│  │  - Fast access      │  │  - Hybrid search    │  │  - Entity relationships │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘

Domain Ontologies:
- Customer: Churn analysis, transactions, engagement
- DevOps: Pipelines, runs, deployments, clusters
- User Management: Users, sessions, auth events, access logs
"""

from .base import (
    MemoryProvider,
    MemoryEntry,
    MemorySearchResult,
    MemoryType,
    CompositeMemory,
)
from .cosmos_memory import ShortTermMemory
from .aisearch_memory import LongTermMemory, FoundryIQMemory
from .facts_memory import (
    FactsMemory,
    Fact,
    FactSearchResult,
    OntologyEntity,
    OntologyRelationship,
    EntityType,
    RelationshipType,
    CustomerEntity,
    PipelineEntity,
    UserAccessEntity,
)
from .ontology_data import (
    # Customer domain
    CustomerDataGenerator,
    CustomerProfile,
    CustomerTransaction,
    EngagementEvent,
    CustomerSegment,
    ChurnRiskLevel,
    # DevOps domain
    PipelineDataGenerator,
    Pipeline,
    PipelineRun,
    DeploymentEvent,
    PipelineStatus,
    FailureCategory,
    # User Management domain
    UserAccessDataGenerator,
    User,
    AuthEvent,
    AccessLog,
    UserStatus,
    AuthEventType,
    AccessAction,
)

__all__ = [
    # Base classes
    "MemoryProvider",
    "MemoryEntry",
    "MemorySearchResult",
    "MemoryType",
    "CompositeMemory",
    # Short-term memory (CosmosDB)
    "ShortTermMemory",
    # Long-term memory (AI Search / Foundry IQ)
    "LongTermMemory",
    "FoundryIQMemory",
    # Facts memory (Fabric IQ)
    "FactsMemory",
    "Fact",
    "FactSearchResult",
    "OntologyEntity",
    "OntologyRelationship",
    "EntityType",
    "RelationshipType",
    "CustomerEntity",
    "PipelineEntity",
    "UserAccessEntity",
    # Customer domain ontology
    "CustomerDataGenerator",
    "CustomerProfile",
    "CustomerTransaction",
    "EngagementEvent",
    "CustomerSegment",
    "ChurnRiskLevel",
    # DevOps domain ontology
    "PipelineDataGenerator",
    "Pipeline",
    "PipelineRun",
    "DeploymentEvent",
    "PipelineStatus",
    "FailureCategory",
    # User Management domain ontology
    "UserAccessDataGenerator",
    "User",
    "AuthEvent",
    "AccessLog",
    "UserStatus",
    "AuthEventType",
    "AccessAction",
]
