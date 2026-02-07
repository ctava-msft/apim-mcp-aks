"""
Fabric IQ Facts Memory Provider

Implements FactsMemory for Microsoft Fabric IQ integration, providing:
- Ontology-grounded facts retrieval for AI agents
- Cross-domain reasoning across Customer, DevOps, and User Management domains
- Integration with OneLake for unified data access (when Fabric is enabled)
- Fallback to Azure Blob Storage for ontology files (when Fabric is disabled)
- Entity-relationship traversal for complex queries

Based on Microsoft Fabric IQ:
https://learn.microsoft.com/en-us/fabric/iq/overview
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient

from .base import MemoryProvider, MemoryEntry, MemorySearchResult, MemoryType

logger = logging.getLogger(__name__)


# =============================================================================
# Ontology Entity Type Definitions
# =============================================================================

class EntityType(Enum):
    """Base entity types for Fabric IQ ontology"""
    # Customer Domain
    CUSTOMER = "customer"
    TRANSACTION = "transaction"
    ENGAGEMENT = "engagement"
    SUBSCRIPTION = "subscription"
    CHURN_PREDICTION = "churn_prediction"
    
    # DevOps Domain
    PIPELINE = "pipeline"
    PIPELINE_RUN = "pipeline_run"
    DEPLOYMENT = "deployment"
    SERVICE = "service"
    CLUSTER = "cluster"
    ARTIFACT = "artifact"
    
    # User Management Domain
    USER = "user"
    SESSION = "session"
    AUTH_EVENT = "auth_event"
    ROLE = "role"
    PERMISSION = "permission"
    ACCESS_LOG = "access_log"


class RelationshipType(Enum):
    """Relationship types for graph traversal"""
    # Customer relationships
    HAS_TRANSACTION = "has_transaction"
    HAS_ENGAGEMENT = "has_engagement"
    HAS_SUBSCRIPTION = "has_subscription"
    PREDICTED_CHURN = "predicted_churn"
    
    # DevOps relationships
    HAS_RUN = "has_run"
    DEPLOYS_TO = "deploys_to"
    CONTAINS_SERVICE = "contains_service"
    PRODUCES_ARTIFACT = "produces_artifact"
    TRIGGERED_BY = "triggered_by"
    
    # User Management relationships
    HAS_SESSION = "has_session"
    HAS_AUTH_EVENT = "has_auth_event"
    HAS_ROLE = "has_role"
    GRANTS_PERMISSION = "grants_permission"
    ACCESSED_RESOURCE = "accessed_resource"


# =============================================================================
# Ontology Entity Classes
# =============================================================================

@dataclass
class OntologyEntity:
    """Base class for all ontology entities"""
    id: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "entity_type": self.entity_type.value,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyEntity":
        return cls(
            id=data["id"],
            entity_type=EntityType(data["entity_type"]),
            properties=data.get("properties", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OntologyRelationship:
    """Represents a relationship between two entities"""
    id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship_type": self.relationship_type.value,
            "properties": self.properties,
            "created_at": self.created_at,
        }


@dataclass
class Fact:
    """
    Represents a grounded fact from Fabric IQ ontology.
    Facts are derived from entities, relationships, and reasoning.
    """
    id: str
    fact_type: str  # e.g., "observation", "prediction", "rule", "derived"
    domain: str  # e.g., "customer", "devops", "user_management"
    statement: str  # Natural language statement of the fact
    confidence: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)  # Entity IDs that support this fact
    context: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    valid_until: Optional[str] = None  # For time-sensitive facts
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "fact_type": self.fact_type,
            "domain": self.domain,
            "statement": self.statement,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "context": self.context,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "valid_until": self.valid_until,
        }


@dataclass
class FactSearchResult:
    """Result from a fact search operation"""
    fact: Fact
    score: float  # Relevance score
    related_entities: List[OntologyEntity] = field(default_factory=list)


# =============================================================================
# Domain-Specific Ontology Classes
# =============================================================================

@dataclass 
class CustomerEntity(OntologyEntity):
    """Customer domain entity"""
    def __init__(
        self,
        id: str,
        email: str,
        name: str,
        tenure_months: int,
        segment: str,
        monthly_spend: float,
        churn_risk: float = 0.0,
        **kwargs
    ):
        super().__init__(
            id=id,
            entity_type=EntityType.CUSTOMER,
            properties={
                "email": email,
                "name": name,
                "tenure_months": tenure_months,
                "segment": segment,
                "monthly_spend": monthly_spend,
                "churn_risk": churn_risk,
            },
            **kwargs
        )


@dataclass
class PipelineEntity(OntologyEntity):
    """CI/CD Pipeline domain entity"""
    def __init__(
        self,
        id: str,
        name: str,
        repository: str,
        target_cluster: str,
        status: str = "active",
        **kwargs
    ):
        super().__init__(
            id=id,
            entity_type=EntityType.PIPELINE,
            properties={
                "name": name,
                "repository": repository,
                "target_cluster": target_cluster,
                "status": status,
            },
            **kwargs
        )


@dataclass
class UserAccessEntity(OntologyEntity):
    """User Management domain entity"""
    def __init__(
        self,
        id: str,
        email: str,
        username: str,
        roles: List[str],
        status: str = "active",
        last_login: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            id=id,
            entity_type=EntityType.USER,
            properties={
                "email": email,
                "username": username,
                "roles": roles,
                "status": status,
                "last_login": last_login,
            },
            **kwargs
        )


# =============================================================================
# Fabric IQ Facts Memory Provider
# =============================================================================

class FactsMemory(MemoryProvider):
    """
    Facts Memory provider with dual-mode storage:
    
    1. Microsoft Fabric IQ mode (when fabric_enabled=True):
       - Uses OneLake for ontology storage
       - Full Fabric IQ integration for cross-domain reasoning
    
    2. Azure Blob Storage mode (when fabric_enabled=False, default):
       - Uses Azure Blob Storage container for ontology files
       - Accesses storage over private network when VNet is enabled
    
    Provides:
    - Ontology-grounded facts for AI agents
    - Cross-domain reasoning across Customer, DevOps, and User domains
    - Entity-relationship graph traversal
    - Semantic fact retrieval with embeddings
    """
    
    def __init__(
        self,
        storage_account_url: Optional[str] = None,
        ontology_container: str = "ontologies",
        fabric_enabled: bool = False,
        fabric_endpoint: Optional[str] = None,
        workspace_id: Optional[str] = None,
        ontology_name: str = "agent-ontology",
        credential: Optional[Any] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
    ):
        """
        Initialize Facts Memory provider.
        
        Args:
            storage_account_url: Azure Blob Storage account URL (used when fabric_enabled=False)
            ontology_container: Container name for ontology files (default: 'ontologies')
            fabric_enabled: Whether to use Fabric IQ (True) or Blob Storage (False)
            fabric_endpoint: Microsoft Fabric endpoint URL (used when fabric_enabled=True)
            workspace_id: Fabric workspace ID (used when fabric_enabled=True)
            ontology_name: Name of the ontology
            credential: Azure credential (defaults to DefaultAzureCredential)
            embedding_function: Function to generate embeddings from text
        """
        self._storage_account_url = storage_account_url or os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
        self._ontology_container = ontology_container
        self._fabric_enabled = fabric_enabled
        self._fabric_endpoint = fabric_endpoint
        self._workspace_id = workspace_id
        self._ontology_name = ontology_name
        self._credential = credential or DefaultAzureCredential()
        self._embedding_function = embedding_function
        
        # Initialize storage client for Blob Storage mode
        self._blob_service_client: Optional[BlobServiceClient] = None
        self._container_client: Optional[ContainerClient] = None
        if not self._fabric_enabled and self._storage_account_url:
            self._init_blob_storage()
        
        # In-memory storage for entities, relationships, and facts
        self._entities: Dict[str, OntologyEntity] = {}
        self._relationships: Dict[str, OntologyRelationship] = {}
        self._facts: Dict[str, Fact] = {}
        
        # Domain indices for fast lookup
        self._entities_by_type: Dict[EntityType, List[str]] = {et: [] for et in EntityType}
        self._entities_by_domain: Dict[str, List[str]] = {
            "customer": [],
            "devops": [],
            "user_management": [],
        }
        
        # Track loaded ontologies
        self._loaded_ontologies: List[str] = []
        
        mode = "Fabric IQ" if self._fabric_enabled else "Azure Blob Storage"
        logger.info(f"Facts Memory initialized: ontology={ontology_name}, mode={mode}")
    
    def _init_blob_storage(self) -> None:
        """Initialize Azure Blob Storage client for ontology access."""
        try:
            self._blob_service_client = BlobServiceClient(
                account_url=self._storage_account_url,
                credential=self._credential
            )
            self._container_client = self._blob_service_client.get_container_client(
                self._ontology_container
            )
            logger.info(f"Blob Storage initialized: {self._storage_account_url}/{self._ontology_container}")
        except Exception as e:
            logger.error(f"Failed to initialize Blob Storage: {e}")
            self._blob_service_client = None
            self._container_client = None
    
    async def load_ontology_from_storage(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """
        Load an ontology file from Azure Blob Storage.
        
        Args:
            blob_name: Name of the blob (e.g., 'customer_churn_ontology.json')
            
        Returns:
            Parsed ontology data or None if not found
        """
        if not self._container_client:
            logger.warning("Blob Storage not initialized, cannot load ontology")
            return None
        
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_data = blob_client.download_blob().readall()
            ontology_data = json.loads(blob_data.decode('utf-8'))
            
            logger.info(f"Loaded ontology from Blob Storage: {blob_name}")
            self._loaded_ontologies.append(blob_name)
            return ontology_data
        except Exception as e:
            logger.error(f"Failed to load ontology {blob_name}: {e}")
            return None
    
    async def load_all_ontologies(self) -> int:
        """
        Load all ontology files from the storage container.
        
        Returns:
            Number of ontologies loaded
        """
        if not self._container_client:
            logger.warning("Blob Storage not initialized")
            return 0
        
        loaded_count = 0
        try:
            blobs = self._container_client.list_blobs()
            for blob in blobs:
                if blob.name.endswith('.json'):
                    ontology = await self.load_ontology_from_storage(blob.name)
                    if ontology:
                        await self._process_ontology(ontology)
                        loaded_count += 1
        except Exception as e:
            logger.error(f"Failed to list ontologies: {e}")
        
        logger.info(f"Loaded {loaded_count} ontologies from storage")
        return loaded_count
    
    async def _process_ontology(self, ontology_data: Dict[str, Any]) -> None:
        """Process and store entities, relationships, and facts from ontology data."""
        # Process entities
        for entity_data in ontology_data.get("entities", []):
            try:
                entity = OntologyEntity.from_dict(entity_data)
                await self.store_entity(entity)
            except Exception as e:
                logger.warning(f"Failed to process entity: {e}")
        
        # Process relationships
        for rel_data in ontology_data.get("relationships", []):
            try:
                relationship = OntologyRelationship(
                    id=rel_data["id"],
                    source_id=rel_data["source_id"],
                    target_id=rel_data["target_id"],
                    relationship_type=RelationshipType(rel_data["relationship_type"]),
                    properties=rel_data.get("properties", {}),
                )
                await self.store_relationship(relationship)
            except Exception as e:
                logger.warning(f"Failed to process relationship: {e}")
        
        # Process facts
        for fact_data in ontology_data.get("facts", []):
            try:
                fact = Fact(
                    id=fact_data["id"],
                    fact_type=fact_data["fact_type"],
                    domain=fact_data["domain"],
                    statement=fact_data["statement"],
                    confidence=fact_data.get("confidence", 1.0),
                    evidence=fact_data.get("evidence", []),
                    context=fact_data.get("context", {}),
                )
                await self.store_fact(fact)
            except Exception as e:
                logger.warning(f"Failed to process fact: {e}")
    
    async def upload_ontology_to_storage(
        self, 
        ontology_data: Dict[str, Any], 
        blob_name: str
    ) -> bool:
        """
        Upload an ontology file to Azure Blob Storage.
        
        Args:
            ontology_data: The ontology data to upload
            blob_name: Name for the blob (e.g., 'customer_churn_ontology.json')
            
        Returns:
            True if upload succeeded, False otherwise
        """
        if not self._container_client:
            logger.warning("Blob Storage not initialized, cannot upload ontology")
            return False
        
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_data = json.dumps(ontology_data, indent=2).encode('utf-8')
            blob_client.upload_blob(blob_data, overwrite=True)
            
            logger.info(f"Uploaded ontology to Blob Storage: {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload ontology {blob_name}: {e}")
            return False
    
    @property
    def name(self) -> str:
        return "fabric_iq_facts"
    
    @property
    def is_short_term(self) -> bool:
        return False  # Facts are persistent knowledge
    
    @property
    def fabric_enabled(self) -> bool:
        """Whether Fabric IQ mode is enabled."""
        return self._fabric_enabled
    
    @property
    def loaded_ontologies(self) -> List[str]:
        """List of loaded ontology file names."""
        return self._loaded_ontologies.copy()
    
    def set_embedding_function(self, func: Callable[[str], List[float]]) -> None:
        """Set the embedding function for semantic fact retrieval."""
        self._embedding_function = func
    
    # =========================================================================
    # Entity Management
    # =========================================================================
    
    async def store_entity(self, entity: OntologyEntity) -> str:
        """Store an ontology entity."""
        self._entities[entity.id] = entity
        self._entities_by_type[entity.entity_type].append(entity.id)
        
        # Map to domain
        domain = self._get_entity_domain(entity.entity_type)
        self._entities_by_domain[domain].append(entity.id)
        
        logger.debug(f"Stored entity: {entity.id} ({entity.entity_type.value})")
        return entity.id
    
    async def get_entity(self, entity_id: str) -> Optional[OntologyEntity]:
        """Retrieve an entity by ID."""
        return self._entities.get(entity_id)
    
    async def query_entities(
        self,
        entity_type: Optional[EntityType] = None,
        domain: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[OntologyEntity]:
        """Query entities with optional filters."""
        results = []
        
        # Get candidate entity IDs
        if entity_type:
            candidate_ids = self._entities_by_type.get(entity_type, [])
        elif domain:
            candidate_ids = self._entities_by_domain.get(domain, [])
        else:
            candidate_ids = list(self._entities.keys())
        
        for entity_id in candidate_ids[:limit]:
            entity = self._entities.get(entity_id)
            if entity and self._matches_filters(entity, filters):
                results.append(entity)
        
        return results
    
    def _matches_filters(self, entity: OntologyEntity, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if entity matches the provided filters."""
        if not filters:
            return True
        
        for key, value in filters.items():
            entity_value = entity.properties.get(key)
            if entity_value != value:
                return False
        return True
    
    def _get_entity_domain(self, entity_type: EntityType) -> str:
        """Map entity type to domain."""
        customer_types = {EntityType.CUSTOMER, EntityType.TRANSACTION, EntityType.ENGAGEMENT, 
                        EntityType.SUBSCRIPTION, EntityType.CHURN_PREDICTION}
        devops_types = {EntityType.PIPELINE, EntityType.PIPELINE_RUN, EntityType.DEPLOYMENT,
                       EntityType.SERVICE, EntityType.CLUSTER, EntityType.ARTIFACT}
        user_types = {EntityType.USER, EntityType.SESSION, EntityType.AUTH_EVENT,
                     EntityType.ROLE, EntityType.PERMISSION, EntityType.ACCESS_LOG}
        
        if entity_type in customer_types:
            return "customer"
        elif entity_type in devops_types:
            return "devops"
        elif entity_type in user_types:
            return "user_management"
        return "unknown"
    
    # =========================================================================
    # Relationship Management
    # =========================================================================
    
    async def store_relationship(self, relationship: OntologyRelationship) -> str:
        """Store a relationship between entities."""
        self._relationships[relationship.id] = relationship
        logger.debug(f"Stored relationship: {relationship.source_id} -> {relationship.target_id}")
        return relationship.id
    
    async def get_related_entities(
        self,
        entity_id: str,
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "outgoing",  # "outgoing", "incoming", "both"
    ) -> List[OntologyEntity]:
        """Get entities related to the given entity."""
        related_ids = []
        
        for rel in self._relationships.values():
            if relationship_type and rel.relationship_type != relationship_type:
                continue
            
            if direction in ("outgoing", "both") and rel.source_id == entity_id:
                related_ids.append(rel.target_id)
            if direction in ("incoming", "both") and rel.target_id == entity_id:
                related_ids.append(rel.source_id)
        
        return [self._entities[rid] for rid in related_ids if rid in self._entities]
    
    # =========================================================================
    # Facts Management
    # =========================================================================
    
    async def store_fact(self, fact: Fact) -> str:
        """Store a derived fact."""
        # Generate embedding if function available
        if self._embedding_function and not fact.embedding:
            try:
                fact.embedding = self._embedding_function(fact.statement)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for fact: {e}")
        
        self._facts[fact.id] = fact
        logger.debug(f"Stored fact: {fact.id} ({fact.domain})")
        return fact.id
    
    async def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by ID."""
        return self._facts.get(fact_id)
    
    async def search_facts(
        self,
        query: str,
        domain: Optional[str] = None,
        fact_type: Optional[str] = None,
        min_confidence: float = 0.5,
        limit: int = 10,
    ) -> List[FactSearchResult]:
        """
        Search for relevant facts using semantic similarity.
        
        Args:
            query: Natural language query
            domain: Filter by domain (customer, devops, user_management)
            fact_type: Filter by fact type (observation, prediction, rule, derived)
            min_confidence: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            List of FactSearchResult ordered by relevance
        """
        import numpy as np
        
        results = []
        
        # Generate query embedding
        query_embedding = None
        if self._embedding_function:
            try:
                query_embedding = self._embedding_function(query)
            except Exception as e:
                logger.warning(f"Failed to generate query embedding: {e}")
        
        for fact in self._facts.values():
            # Apply filters
            if domain and fact.domain != domain:
                continue
            if fact_type and fact.fact_type != fact_type:
                continue
            if fact.confidence < min_confidence:
                continue
            
            # Calculate relevance score
            score = 0.0
            if query_embedding and fact.embedding:
                # Cosine similarity
                arr1 = np.array(query_embedding)
                arr2 = np.array(fact.embedding)
                dot_product = np.dot(arr1, arr2)
                norm1 = np.linalg.norm(arr1)
                norm2 = np.linalg.norm(arr2)
                if norm1 > 0 and norm2 > 0:
                    score = float(dot_product / (norm1 * norm2))
            else:
                # Fallback: keyword matching
                query_lower = query.lower()
                statement_lower = fact.statement.lower()
                matching_words = sum(1 for word in query_lower.split() if word in statement_lower)
                score = matching_words / max(len(query_lower.split()), 1)
            
            # Get related entities for context
            related_entities = []
            for evidence_id in fact.evidence[:3]:  # Limit for performance
                entity = await self.get_entity(evidence_id)
                if entity:
                    related_entities.append(entity)
            
            results.append(FactSearchResult(
                fact=fact,
                score=score,
                related_entities=related_entities,
            ))
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    async def derive_facts(
        self,
        entity_id: str,
        reasoning_type: str = "basic",
    ) -> List[Fact]:
        """
        Derive new facts from an entity and its relationships.
        
        Args:
            entity_id: The entity to analyze
            reasoning_type: Type of reasoning (basic, aggregate, predictive)
            
        Returns:
            List of derived facts
        """
        entity = await self.get_entity(entity_id)
        if not entity:
            return []
        
        derived_facts = []
        domain = self._get_entity_domain(entity.entity_type)
        
        if reasoning_type == "basic":
            # Basic fact derivation from entity properties
            if entity.entity_type == EntityType.CUSTOMER:
                churn_risk = entity.properties.get("churn_risk", 0)
                if churn_risk > 0.7:
                    derived_facts.append(Fact(
                        id=f"fact-{entity_id}-high-churn",
                        fact_type="observation",
                        domain=domain,
                        statement=f"Customer {entity.properties.get('name')} has high churn risk ({churn_risk:.0%})",
                        confidence=churn_risk,
                        evidence=[entity_id],
                        context={"churn_risk": churn_risk, "segment": entity.properties.get("segment")},
                    ))
            
            elif entity.entity_type == EntityType.PIPELINE:
                status = entity.properties.get("status")
                derived_facts.append(Fact(
                    id=f"fact-{entity_id}-status",
                    fact_type="observation",
                    domain=domain,
                    statement=f"Pipeline {entity.properties.get('name')} is {status}",
                    confidence=1.0,
                    evidence=[entity_id],
                    context={"repository": entity.properties.get("repository")},
                ))
        
        # Store derived facts
        for fact in derived_facts:
            await self.store_fact(fact)
        
        return derived_facts
    
    # =========================================================================
    # Fabric Lakehouse Integration
    # =========================================================================
    
    async def load_entities_from_lakehouse(
        self,
        lakehouse_id: str,
        table_name: str,
        entity_type: EntityType,
        id_column: str = "id",
        property_columns: Optional[List[str]] = None,
    ) -> int:
        """
        Load entities from a Fabric Lakehouse table.
        
        This method queries a Fabric Lakehouse table and creates ontology entities
        from the results. Useful for pulling real-time data from Fabric into facts memory.
        
        Args:
            lakehouse_id: ID of the Fabric lakehouse
            table_name: Name of the table to query
            entity_type: Type of entities to create
            id_column: Column name to use as entity ID (default: "id")
            property_columns: Specific columns to include (default: all columns)
        
        Returns:
            Number of entities loaded
        """
        if not self._fabric_enabled:
            logger.warning("Fabric is not enabled, cannot load from lakehouse")
            return 0
        
        try:
            # Import fabric_tools dynamically to avoid circular dependency
            from fabric_tools import get_fabric_client
            
            client = get_fabric_client()
            
            # Build Spark SQL query
            columns = "*" if not property_columns else ", ".join(property_columns)
            query = f"SELECT {columns} FROM {table_name}"
            
            result = client.query_lakehouse(lakehouse_id, query, table_name)
            
            if not result.get("results"):
                logger.warning(f"No results from lakehouse table {table_name}")
                return 0
            
            # Parse results and create entities
            count = 0
            rows = result.get("results", {}).get("rows", [])
            
            for row in rows:
                try:
                    entity_id = str(row.get(id_column, f"lakehouse-{count}"))
                    
                    # Extract properties (exclude id_column)
                    properties = {k: v for k, v in row.items() if k != id_column}
                    
                    entity = OntologyEntity(
                        id=entity_id,
                        entity_type=entity_type,
                        properties=properties,
                        metadata={
                            "source": "fabric_lakehouse",
                            "lakehouse_id": lakehouse_id,
                            "table_name": table_name,
                        }
                    )
                    
                    await self.store_entity(entity)
                    count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to create entity from row: {e}")
            
            logger.info(f"Loaded {count} entities from Fabric lakehouse table {table_name}")
            return count
        
        except Exception as e:
            logger.error(f"Error loading entities from lakehouse: {e}")
            return 0
    
    async def sync_facts_from_warehouse(
        self,
        warehouse_id: str,
        fact_table: str,
        domain: str,
        statement_column: str = "statement",
        confidence_column: str = "confidence",
    ) -> int:
        """
        Synchronize facts from a Fabric Data Warehouse table.
        
        This method queries a Fabric Warehouse table containing facts and loads them
        into facts memory. Useful for maintaining facts in a centralized warehouse.
        
        Args:
            warehouse_id: ID of the Fabric warehouse
            fact_table: Name of the facts table
            domain: Domain for the facts (customer, devops, user_management)
            statement_column: Column containing fact statements
            confidence_column: Column containing confidence scores
        
        Returns:
            Number of facts loaded
        """
        if not self._fabric_enabled:
            logger.warning("Fabric is not enabled, cannot sync from warehouse")
            return 0
        
        try:
            # Import fabric_tools dynamically to avoid circular dependency
            from fabric_tools import get_fabric_client
            
            client = get_fabric_client()
            
            # Query facts from warehouse
            query = f"SELECT * FROM {fact_table}"
            result = client.query_warehouse(warehouse_id, query, fact_table)
            
            if not result.get("results"):
                logger.warning(f"No results from warehouse table {fact_table}")
                return 0
            
            # Parse results and create facts
            count = 0
            rows = result.get("results", {}).get("rows", [])
            
            for row in rows:
                try:
                    fact_id = row.get("id", f"warehouse-fact-{count}")
                    statement = row.get(statement_column, "")
                    confidence = float(row.get(confidence_column, 0.8))
                    
                    if not statement:
                        continue
                    
                    fact = Fact(
                        id=fact_id,
                        fact_type=row.get("fact_type", "derived"),
                        domain=domain,
                        statement=statement,
                        confidence=confidence,
                        evidence=row.get("evidence", []),
                        source="fabric_warehouse",
                        context={
                            "warehouse_id": warehouse_id,
                            "table": fact_table,
                            **{k: v for k, v in row.items() if k not in [statement_column, confidence_column, "id"]}
                        }
                    )
                    
                    await self.store_fact(fact)
                    count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to create fact from row: {e}")
            
            logger.info(f"Loaded {count} facts from Fabric warehouse table {fact_table}")
            return count
        
        except Exception as e:
            logger.error(f"Error syncing facts from warehouse: {e}")
            return 0
    
    def get_fabric_sync_status(self) -> Dict[str, Any]:
        """
        Get status of Fabric data synchronization.
        
        Returns:
            Dictionary with sync status information
        """
        return {
            "fabric_enabled": self._fabric_enabled,
            "fabric_endpoint": self._fabric_endpoint,
            "workspace_id": self._workspace_id,
            "total_entities": len(self._entities),
            "total_facts": len(self._facts),
            "entities_by_domain": {
                domain: len(entity_ids)
                for domain, entity_ids in self._entities_by_domain.items()
            },
            "loaded_ontologies": self._loaded_ontologies,
        }
    
    # =========================================================================
    # Cross-Domain Reasoning
    # =========================================================================
    
    async def cross_domain_query(
        self,
        query: str,
        source_domain: str,
        target_domain: str,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Perform cross-domain reasoning to find connections.
        
        Example: Find pipeline failures that affected high-value customers
        
        Args:
            query: Natural language query describing the connection to find
            source_domain: Starting domain
            target_domain: Target domain to connect to
            max_hops: Maximum relationship hops to traverse
            
        Returns:
            List of connection paths with entities and relationships
        """
        # This would use Fabric IQ's graph capabilities in production
        # For now, return related facts across domains
        
        source_facts = await self.search_facts(query, domain=source_domain, limit=5)
        target_facts = await self.search_facts(query, domain=target_domain, limit=5)
        
        connections = []
        for sf in source_facts:
            for tf in target_facts:
                # Check for common context or evidence
                common_context = set(sf.fact.evidence) & set(tf.fact.evidence)
                if common_context or (sf.score > 0.5 and tf.score > 0.5):
                    connections.append({
                        "source_fact": sf.fact.to_dict(),
                        "target_fact": tf.fact.to_dict(),
                        "connection_strength": (sf.score + tf.score) / 2,
                        "common_evidence": list(common_context),
                    })
        
        return sorted(connections, key=lambda x: x["connection_strength"], reverse=True)
    
    # =========================================================================
    # MemoryProvider Interface Implementation
    # =========================================================================
    
    async def store(self, entry: MemoryEntry) -> str:
        """Store as a fact with memory entry wrapper."""
        fact = Fact(
            id=entry.id,
            fact_type="stored",
            domain=entry.metadata.get("domain", "general"),
            statement=entry.content,
            confidence=1.0,
            embedding=entry.embedding,
            context=entry.metadata,
        )
        return await self.store_fact(fact)
    
    async def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a fact as memory entry."""
        fact = await self.get_fact(entry_id)
        if not fact:
            return None
        
        return MemoryEntry(
            id=fact.id,
            content=fact.statement,
            memory_type=MemoryType.CONTEXT,
            embedding=fact.embedding,
            metadata={"fact_type": fact.fact_type, "domain": fact.domain, **fact.context},
        )
    
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7,
        memory_type: Optional[MemoryType] = None,
        session_id: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        """Search facts by embedding."""
        import numpy as np
        
        results = []
        
        for fact in self._facts.values():
            if not fact.embedding:
                continue
            
            # Calculate cosine similarity
            arr1 = np.array(query_embedding)
            arr2 = np.array(fact.embedding)
            dot_product = np.dot(arr1, arr2)
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            
            if norm1 > 0 and norm2 > 0:
                score = float(dot_product / (norm1 * norm2))
                if score >= threshold:
                    entry = MemoryEntry(
                        id=fact.id,
                        content=fact.statement,
                        memory_type=MemoryType.CONTEXT,
                        embedding=fact.embedding,
                        metadata={"domain": fact.domain, "fact_type": fact.fact_type},
                    )
                    results.append(MemorySearchResult(entry=entry, score=score, source=self.name))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    async def search_by_text(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        session_id: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        """Search facts by text query."""
        fact_results = await self.search_facts(query, limit=limit)
        
        return [
            MemorySearchResult(
                entry=MemoryEntry(
                    id=fr.fact.id,
                    content=fr.fact.statement,
                    memory_type=MemoryType.CONTEXT,
                    embedding=fr.fact.embedding,
                    metadata={"domain": fr.fact.domain, "fact_type": fr.fact.fact_type},
                ),
                score=fr.score,
                source=self.name,
            )
            for fr in fact_results
        ]
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a fact."""
        if entry_id in self._facts:
            del self._facts[entry_id]
            return True
        return False
    
    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        memory_type: Optional[MemoryType] = None,
    ) -> List[MemoryEntry]:
        """List facts (not session-based for FactsMemory)."""
        return []
    
    async def clear_session(self, session_id: str) -> int:
        """Clear session (not applicable for FactsMemory)."""
        return 0
    
    async def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get conversation history (not applicable for FactsMemory)."""
        return []
    
    async def health_check(self) -> bool:
        """Check if the facts memory provider is healthy."""
        return True
    
    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the facts memory."""
        return {
            "storage_mode": "fabric_iq" if self._fabric_enabled else "blob_storage",
            "storage_url": self._fabric_endpoint if self._fabric_enabled else self._storage_account_url,
            "ontology_container": self._ontology_container,
            "loaded_ontologies": self._loaded_ontologies,
            "total_entities": len(self._entities),
            "total_relationships": len(self._relationships),
            "total_facts": len(self._facts),
            "entities_by_type": {et.value: len(ids) for et, ids in self._entities_by_type.items() if ids},
            "entities_by_domain": {d: len(ids) for d, ids in self._entities_by_domain.items() if ids},
            "facts_by_domain": self._count_facts_by_domain(),
        }
    
    def _count_facts_by_domain(self) -> Dict[str, int]:
        """Count facts by domain."""
        counts = {"customer": 0, "devops": 0, "user_management": 0}
        for fact in self._facts.values():
            if fact.domain in counts:
                counts[fact.domain] += 1
        return counts
