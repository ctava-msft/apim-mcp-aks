"""
Cosmos DB RL Ledger - Persistent storage for reinforcement learning artifacts.

This module provides the authoritative system of record for:
- Episodes (prompt → tool calls → response)
- Rewards/labels (approval, scores, safety flags)
- Datasets (manifests for fine-tuning)
- Training runs (hyperparams, input/output)
- Deployments (active tuned models, rollback history)

Aligned with architecture goal: Cosmos DB for durable long-term memory and lineage/traces.
"""

import logging
import hashlib
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Callable

from azure.cosmos import CosmosClient, ContainerProxy, PartitionKey
from azure.cosmos import exceptions as cosmos_exceptions
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)


class RewardSource(Enum):
    """Source of a reward/label."""
    EVAL_SCORE = "eval_score"
    HUMAN_APPROVAL = "human_approval"
    SAFETY_CHECK = "safety_check"
    TEST_RESULT = "test_result"
    COST_PENALTY = "cost_penalty"
    LATENCY_PENALTY = "latency_penalty"
    GOLDEN_CONVERSATION = "golden_conversation"


class TrainingStatus(Enum):
    """Status of a training run."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EpisodeToolCall:
    """A single tool call within an episode."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None  # Redacted result
    duration_ms: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeToolCall":
        return cls(
            tool_name=data.get("tool_name", ""),
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            duration_ms=data.get("duration_ms"),
            error=data.get("error"),
        )


@dataclass
class Episode:
    """
    An episode represents a complete agent interaction:
    user input → tool calls → final response.
    """
    id: str
    agent_id: str
    user_input: str
    assistant_output: str
    tool_calls: List[EpisodeToolCall] = field(default_factory=list)
    instructions_hash: Optional[str] = None  # Hash of agent instructions
    model_deployment: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    request_latency_ms: Optional[int] = None
    token_usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "user_input": self.user_input,
            "assistant_output": self.assistant_output,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "instructions_hash": self.instructions_hash,
            "model_deployment": self.model_deployment,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "request_latency_ms": self.request_latency_ms,
            "token_usage": self.token_usage,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        tool_calls = [
            EpisodeToolCall.from_dict(tc) for tc in data.get("tool_calls", [])
        ]
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            user_input=data.get("user_input", ""),
            assistant_output=data.get("assistant_output", ""),
            tool_calls=tool_calls,
            instructions_hash=data.get("instructions_hash"),
            model_deployment=data.get("model_deployment"),
            correlation_id=data.get("correlation_id"),
            session_id=data.get("session_id"),
            request_latency_ms=data.get("request_latency_ms"),
            token_usage=data.get("token_usage"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )


@dataclass
class Reward:
    """A reward/label attached to an episode."""
    id: str
    episode_id: str
    agent_id: str
    source: RewardSource
    value: float  # -1.0 to 1.0 normalized
    raw_value: Optional[Any] = None  # Original value (score, boolean, etc.)
    rubric: Optional[str] = None  # Evaluation rubric used
    evaluator: Optional[str] = None  # Who/what evaluated
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "source": self.source.value,
            "value": self.value,
            "raw_value": self.raw_value,
            "rubric": self.rubric,
            "evaluator": self.evaluator,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reward":
        return cls(
            id=data["id"],
            episode_id=data["episode_id"],
            agent_id=data["agent_id"],
            source=RewardSource(data.get("source", "eval_score")),
            value=data.get("value", 0.0),
            raw_value=data.get("raw_value"),
            rubric=data.get("rubric"),
            evaluator=data.get("evaluator"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )


@dataclass
class Dataset:
    """Manifest for a fine-tuning dataset built from episodes."""
    id: str
    agent_id: str
    name: str
    description: Optional[str] = None
    query_filter: Optional[str] = None  # How episodes were selected
    episode_ids: List[str] = field(default_factory=list)
    training_count: int = 0
    validation_count: int = 0
    local_path: Optional[str] = None  # Local JSONL path
    blob_path: Optional[str] = None  # Azure Blob path
    reward_threshold: Optional[float] = None  # Min reward for inclusion
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "query_filter": self.query_filter,
            "episode_ids": self.episode_ids,
            "training_count": self.training_count,
            "validation_count": self.validation_count,
            "local_path": self.local_path,
            "blob_path": self.blob_path,
            "reward_threshold": self.reward_threshold,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            name=data.get("name", ""),
            description=data.get("description"),
            query_filter=data.get("query_filter"),
            episode_ids=data.get("episode_ids", []),
            training_count=data.get("training_count", 0),
            validation_count=data.get("validation_count", 0),
            local_path=data.get("local_path"),
            blob_path=data.get("blob_path"),
            reward_threshold=data.get("reward_threshold"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )


@dataclass
class TrainingRun:
    """Record of a fine-tuning training run."""
    id: str
    agent_id: str
    dataset_id: str
    base_model: str
    tuned_model_name: Optional[str] = None  # Result model deployment name
    status: TrainingStatus = TrainingStatus.PENDING
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    aoai_job_id: Optional[str] = None  # Azure OpenAI fine-tune job ID
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "dataset_id": self.dataset_id,
            "base_model": self.base_model,
            "tuned_model_name": self.tuned_model_name,
            "status": self.status.value,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "aoai_job_id": self.aoai_job_id,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingRun":
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            dataset_id=data["dataset_id"],
            base_model=data.get("base_model", ""),
            tuned_model_name=data.get("tuned_model_name"),
            status=TrainingStatus(data.get("status", "pending")),
            hyperparameters=data.get("hyperparameters", {}),
            metrics=data.get("metrics", {}),
            aoai_job_id=data.get("aoai_job_id"),
            error_message=data.get("error_message"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )


@dataclass
class Deployment:
    """Record of an active tuned model deployment."""
    id: str
    agent_id: str
    training_run_id: str
    tuned_model_name: str
    is_active: bool = True
    promoted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    promoted_by: Optional[str] = None
    rollback_from: Optional[str] = None  # Previous deployment ID if rollback
    rollback_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "training_run_id": self.training_run_id,
            "tuned_model_name": self.tuned_model_name,
            "is_active": self.is_active,
            "promoted_at": self.promoted_at,
            "promoted_by": self.promoted_by,
            "rollback_from": self.rollback_from,
            "rollback_reason": self.rollback_reason,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Deployment":
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            training_run_id=data["training_run_id"],
            tuned_model_name=data.get("tuned_model_name", ""),
            is_active=data.get("is_active", True),
            promoted_at=data.get("promoted_at", datetime.utcnow().isoformat()),
            promoted_by=data.get("promoted_by"),
            rollback_from=data.get("rollback_from"),
            rollback_reason=data.get("rollback_reason"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )


class RLLedgerCosmos:
    """
    Cosmos DB-backed RL Ledger for Agent Lightning.
    
    Persists all reinforcement learning artifacts with AAD or key auth.
    Partition key: agent_id (configurable via COSMOS_PARTITION_KEY_FIELD).
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        database_name: Optional[str] = None,
        auth_mode: Optional[str] = None,
        account_key: Optional[str] = None,
        container_episodes: Optional[str] = None,
        container_rewards: Optional[str] = None,
        container_datasets: Optional[str] = None,
        container_runs: Optional[str] = None,
        container_deployments: Optional[str] = None,
        partition_key_field: Optional[str] = None,
        credential: Optional[Any] = None,
    ):
        """
        Initialize the RL Ledger with Cosmos DB.
        
        Args:
            endpoint: Cosmos DB endpoint (or COSMOS_ACCOUNT_URI env var)
            database_name: Database name (or COSMOS_DATABASE_NAME env var)
            auth_mode: 'aad' or 'key' (or COSMOS_AUTH_MODE env var)
            account_key: Account key if auth_mode='key' (or COSMOS_ACCOUNT_KEY)
            container_*: Container names (or COSMOS_CONTAINER_* env vars)
            partition_key_field: Field for partitioning (default: agent_id)
            credential: Azure credential (uses DefaultAzureCredential if not provided)
        """
        # Load from env vars with defaults
        self._endpoint = endpoint or os.getenv("COSMOS_ACCOUNT_URI", "")
        self._database_name = database_name or os.getenv("COSMOS_DATABASE_NAME", "agent_rl")
        self._auth_mode = auth_mode or os.getenv("COSMOS_AUTH_MODE", "aad")
        self._account_key = account_key or os.getenv("COSMOS_ACCOUNT_KEY", "")
        
        self._container_names = {
            "episodes": container_episodes or os.getenv("COSMOS_CONTAINER_EPISODES", "rl_episodes"),
            "rewards": container_rewards or os.getenv("COSMOS_CONTAINER_REWARDS", "rl_rewards"),
            "datasets": container_datasets or os.getenv("COSMOS_CONTAINER_DATASETS", "rl_datasets"),
            "runs": container_runs or os.getenv("COSMOS_CONTAINER_RUNS", "rl_training_runs"),
            "deployments": container_deployments or os.getenv("COSMOS_CONTAINER_DEPLOYMENTS", "rl_deployments"),
        }
        
        self._partition_key_field = partition_key_field or os.getenv("COSMOS_PARTITION_KEY_FIELD", "agent_id")
        
        self._client: Optional[CosmosClient] = None
        self._database = None
        self._containers: Dict[str, ContainerProxy] = {}
        self._initialized = False
        self._credential = credential
        
        # Cache for active deployments (TTL-based)
        self._deployment_cache: Dict[str, tuple] = {}  # agent_id -> (deployment, timestamp)
        self._cache_ttl_seconds = int(os.getenv("LIGHTNING_CACHE_TTL", "60"))

    def _ensure_initialized(self) -> bool:
        """Lazily initialize Cosmos client and containers."""
        if self._initialized:
            return True
        
        if not self._endpoint:
            logger.warning("COSMOS_ACCOUNT_URI not configured - RL Ledger disabled")
            return False
        
        try:
            # Choose auth method
            if self._auth_mode.lower() == "key" and self._account_key:
                self._client = CosmosClient(self._endpoint, credential=self._account_key)
                logger.info("RL Ledger: Connected to Cosmos with key auth")
            else:
                # Use AAD auth
                if self._credential is None:
                    self._credential = DefaultAzureCredential()
                self._client = CosmosClient(self._endpoint, credential=self._credential)
                logger.info("RL Ledger: Connected to Cosmos with AAD auth")
            
            # Get or create database
            self._database = self._client.create_database_if_not_exists(self._database_name)
            
            # Get or create containers
            partition_key = PartitionKey(path=f"/{self._partition_key_field}")
            for key, name in self._container_names.items():
                try:
                    self._containers[key] = self._database.create_container_if_not_exists(
                        id=name,
                        partition_key=partition_key,
                    )
                    logger.info(f"RL Ledger: Container '{name}' ready")
                except cosmos_exceptions.CosmosHttpResponseError as e:
                    # Container might already exist with different settings
                    self._containers[key] = self._database.get_container_client(name)
                    logger.info(f"RL Ledger: Connected to existing container '{name}'")
            
            self._initialized = True
            logger.info(f"RL Ledger initialized: database={self._database_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RL Ledger: {e}")
            return False

    # ==========================================
    # Episode Operations
    # ==========================================

    def store_episode(self, episode: Episode) -> Optional[str]:
        """Store an episode in Cosmos. Returns episode ID or None on failure."""
        if not self._ensure_initialized():
            return None
        
        try:
            doc = episode.to_dict()
            self._containers["episodes"].upsert_item(doc)
            logger.debug(f"Stored episode: {episode.id}")
            return episode.id
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to store episode {episode.id}: {e.message}")
            return None

    def get_episode(self, episode_id: str, agent_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID."""
        if not self._ensure_initialized():
            return None
        
        try:
            doc = self._containers["episodes"].read_item(
                item=episode_id,
                partition_key=agent_id
            )
            return Episode.from_dict(doc)
        except cosmos_exceptions.CosmosResourceNotFoundError:
            return None
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to get episode {episode_id}: {e.message}")
            return None

    def query_episodes(
        self,
        agent_id: Optional[str] = None,
        min_reward: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Episode]:
        """Query episodes with optional filters."""
        if not self._ensure_initialized():
            return []
        
        try:
            query_parts = ["SELECT * FROM c WHERE 1=1"]
            parameters = []
            
            if agent_id:
                query_parts.append("AND c.agent_id = @agent_id")
                parameters.append({"name": "@agent_id", "value": agent_id})
            
            if start_date:
                query_parts.append("AND c.created_at >= @start_date")
                parameters.append({"name": "@start_date", "value": start_date})
            
            if end_date:
                query_parts.append("AND c.created_at <= @end_date")
                parameters.append({"name": "@end_date", "value": end_date})
            
            query_parts.append("ORDER BY c.created_at DESC")
            query_parts.append(f"OFFSET 0 LIMIT {limit}")
            
            query = " ".join(query_parts)
            
            items = list(self._containers["episodes"].query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=agent_id is None,
            ))
            
            return [Episode.from_dict(item) for item in items]
            
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to query episodes: {e.message}")
            return []

    # ==========================================
    # Reward Operations
    # ==========================================

    def store_reward(self, reward: Reward) -> Optional[str]:
        """Store a reward/label in Cosmos."""
        if not self._ensure_initialized():
            return None
        
        try:
            doc = reward.to_dict()
            self._containers["rewards"].upsert_item(doc)
            logger.debug(f"Stored reward: {reward.id} for episode {reward.episode_id}")
            return reward.id
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to store reward {reward.id}: {e.message}")
            return None

    def get_rewards_for_episode(self, episode_id: str, agent_id: str) -> List[Reward]:
        """Get all rewards for an episode."""
        if not self._ensure_initialized():
            return []
        
        try:
            query = "SELECT * FROM c WHERE c.episode_id = @episode_id"
            items = list(self._containers["rewards"].query_items(
                query=query,
                parameters=[{"name": "@episode_id", "value": episode_id}],
                partition_key=agent_id,
            ))
            return [Reward.from_dict(item) for item in items]
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to get rewards for episode {episode_id}: {e.message}")
            return []

    def query_rewards(
        self,
        agent_id: str,
        episode_id: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[Reward]:
        """
        Query rewards with optional filters.
        
        Args:
            agent_id: Agent ID to filter by
            episode_id: Optional episode ID to filter by
            min_value: Optional minimum reward value
            max_value: Optional maximum reward value
            source: Optional reward source to filter by
            limit: Maximum number of rewards to return
            
        Returns:
            List of matching rewards
        """
        if not self._ensure_initialized():
            return []
        
        try:
            query_parts = ["SELECT TOP @limit * FROM c WHERE 1=1"]
            parameters = [{"name": "@limit", "value": limit}]
            
            if episode_id:
                query_parts.append("AND c.episode_id = @episode_id")
                parameters.append({"name": "@episode_id", "value": episode_id})
            
            if min_value is not None:
                query_parts.append("AND c.value >= @min_value")
                parameters.append({"name": "@min_value", "value": min_value})
            
            if max_value is not None:
                query_parts.append("AND c.value <= @max_value")
                parameters.append({"name": "@max_value", "value": max_value})
            
            if source:
                query_parts.append("AND c.source = @source")
                parameters.append({"name": "@source", "value": source})
            
            query_parts.append("ORDER BY c.created_at DESC")
            query = " ".join(query_parts)
            
            items = list(self._containers["rewards"].query_items(
                query=query,
                parameters=parameters,
                partition_key=agent_id,
            ))
            return [Reward.from_dict(item) for item in items]
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to query rewards: {e.message}")
            return []

    def query_episodes_with_rewards(
        self,
        agent_id: str,
        min_reward: Optional[float] = None,
        sources: Optional[List[RewardSource]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query episodes that have rewards, returning both episode and reward data.
        Returns list of {episode: Episode, rewards: List[Reward], avg_reward: float}
        """
        if not self._ensure_initialized():
            return []
        
        try:
            # Get rewards first
            query_parts = ["SELECT * FROM c WHERE c.agent_id = @agent_id"]
            parameters = [{"name": "@agent_id", "value": agent_id}]
            
            if sources:
                source_values = [s.value for s in sources]
                query_parts.append("AND ARRAY_CONTAINS(@sources, c.source)")
                parameters.append({"name": "@sources", "value": source_values})
            
            query = " ".join(query_parts)
            
            reward_items = list(self._containers["rewards"].query_items(
                query=query,
                parameters=parameters,
                partition_key=agent_id,
            ))
            
            # Group rewards by episode
            episode_rewards: Dict[str, List[Reward]] = {}
            for item in reward_items:
                reward = Reward.from_dict(item)
                if reward.episode_id not in episode_rewards:
                    episode_rewards[reward.episode_id] = []
                episode_rewards[reward.episode_id].append(reward)
            
            # Filter by min_reward if specified
            results = []
            for episode_id, rewards in episode_rewards.items():
                avg_reward = sum(r.value for r in rewards) / len(rewards)
                if min_reward is not None and avg_reward < min_reward:
                    continue
                
                episode = self.get_episode(episode_id, agent_id)
                if episode:
                    results.append({
                        "episode": episode,
                        "rewards": rewards,
                        "avg_reward": avg_reward,
                    })
            
            # Sort by avg reward descending and limit
            results.sort(key=lambda x: x["avg_reward"], reverse=True)
            return results[:limit]
            
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to query episodes with rewards: {e.message}")
            return []

    # ==========================================
    # Dataset Operations
    # ==========================================

    def store_dataset(self, dataset: Dataset) -> Optional[str]:
        """Store a dataset manifest in Cosmos."""
        if not self._ensure_initialized():
            return None
        
        try:
            doc = dataset.to_dict()
            self._containers["datasets"].upsert_item(doc)
            logger.info(f"Stored dataset: {dataset.id} ({dataset.name})")
            return dataset.id
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to store dataset {dataset.id}: {e.message}")
            return None

    def get_dataset(self, dataset_id: str, agent_id: str) -> Optional[Dataset]:
        """Retrieve a dataset by ID."""
        if not self._ensure_initialized():
            return None
        
        try:
            doc = self._containers["datasets"].read_item(
                item=dataset_id,
                partition_key=agent_id
            )
            return Dataset.from_dict(doc)
        except cosmos_exceptions.CosmosResourceNotFoundError:
            return None
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to get dataset {dataset_id}: {e.message}")
            return None

    def list_datasets(self, agent_id: str, limit: int = 50) -> List[Dataset]:
        """List all datasets for an agent."""
        if not self._ensure_initialized():
            return []
        
        try:
            query = f"SELECT * FROM c WHERE c.agent_id = @agent_id ORDER BY c.created_at DESC OFFSET 0 LIMIT {limit}"
            items = list(self._containers["datasets"].query_items(
                query=query,
                parameters=[{"name": "@agent_id", "value": agent_id}],
                partition_key=agent_id,
            ))
            return [Dataset.from_dict(item) for item in items]
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to list datasets: {e.message}")
            return []

    # ==========================================
    # Training Run Operations
    # ==========================================

    def store_training_run(self, run: TrainingRun) -> Optional[str]:
        """Store a training run in Cosmos."""
        if not self._ensure_initialized():
            return None
        
        try:
            doc = run.to_dict()
            self._containers["runs"].upsert_item(doc)
            logger.info(f"Stored training run: {run.id} (status={run.status.value})")
            return run.id
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to store training run {run.id}: {e.message}")
            return None

    def get_training_run(self, run_id: str, agent_id: str) -> Optional[TrainingRun]:
        """Retrieve a training run by ID."""
        if not self._ensure_initialized():
            return None
        
        try:
            doc = self._containers["runs"].read_item(
                item=run_id,
                partition_key=agent_id
            )
            return TrainingRun.from_dict(doc)
        except cosmos_exceptions.CosmosResourceNotFoundError:
            return None
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to get training run {run_id}: {e.message}")
            return None

    def update_training_run_status(
        self,
        run_id: str,
        agent_id: str,
        status: TrainingStatus,
        tuned_model_name: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update a training run's status and results."""
        run = self.get_training_run(run_id, agent_id)
        if not run:
            return False
        
        run.status = status
        if tuned_model_name:
            run.tuned_model_name = tuned_model_name
        if metrics:
            run.metrics.update(metrics)
        if error_message:
            run.error_message = error_message
        
        if status == TrainingStatus.RUNNING:
            run.started_at = datetime.utcnow().isoformat()
        elif status in (TrainingStatus.SUCCEEDED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
            run.completed_at = datetime.utcnow().isoformat()
        
        return self.store_training_run(run) is not None

    def list_training_runs(
        self,
        agent_id: str,
        status: Optional[TrainingStatus] = None,
        limit: int = 50,
    ) -> List[TrainingRun]:
        """List training runs for an agent."""
        if not self._ensure_initialized():
            return []
        
        try:
            query_parts = ["SELECT * FROM c WHERE c.agent_id = @agent_id"]
            parameters = [{"name": "@agent_id", "value": agent_id}]
            
            if status:
                query_parts.append("AND c.status = @status")
                parameters.append({"name": "@status", "value": status.value})
            
            query_parts.append(f"ORDER BY c.created_at DESC OFFSET 0 LIMIT {limit}")
            query = " ".join(query_parts)
            
            items = list(self._containers["runs"].query_items(
                query=query,
                parameters=parameters,
                partition_key=agent_id,
            ))
            return [TrainingRun.from_dict(item) for item in items]
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to list training runs: {e.message}")
            return []

    # ==========================================
    # Deployment Operations
    # ==========================================

    def store_deployment(self, deployment: Deployment) -> Optional[str]:
        """Store a deployment record in Cosmos."""
        if not self._ensure_initialized():
            return None
        
        try:
            doc = deployment.to_dict()
            self._containers["deployments"].upsert_item(doc)
            logger.info(f"Stored deployment: {deployment.id} (model={deployment.tuned_model_name})")
            
            # Invalidate cache
            if deployment.agent_id in self._deployment_cache:
                del self._deployment_cache[deployment.agent_id]
            
            return deployment.id
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to store deployment {deployment.id}: {e.message}")
            return None

    def get_active_deployment(self, agent_id: str) -> Optional[Deployment]:
        """
        Get the currently active deployment for an agent.
        Uses caching with TTL to avoid hot-loop reads.
        """
        # Check cache first
        if agent_id in self._deployment_cache:
            deployment, cached_at = self._deployment_cache[agent_id]
            age_seconds = (datetime.utcnow() - datetime.fromisoformat(cached_at)).total_seconds()
            if age_seconds < self._cache_ttl_seconds:
                return deployment
        
        if not self._ensure_initialized():
            return None
        
        try:
            query = "SELECT * FROM c WHERE c.agent_id = @agent_id AND c.is_active = true ORDER BY c.promoted_at DESC OFFSET 0 LIMIT 1"
            items = list(self._containers["deployments"].query_items(
                query=query,
                parameters=[{"name": "@agent_id", "value": agent_id}],
                partition_key=agent_id,
            ))
            
            if items:
                deployment = Deployment.from_dict(items[0])
                self._deployment_cache[agent_id] = (deployment, datetime.utcnow().isoformat())
                return deployment
            return None
            
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to get active deployment for {agent_id}: {e.message}")
            return None

    def promote_deployment(
        self,
        agent_id: str,
        training_run_id: str,
        tuned_model_name: str,
        promoted_by: Optional[str] = None,
    ) -> Optional[Deployment]:
        """
        Promote a tuned model to active deployment.
        Deactivates the previous deployment.
        """
        if not self._ensure_initialized():
            return None
        
        try:
            # Deactivate current active deployment
            current = self.get_active_deployment(agent_id)
            if current:
                current.is_active = False
                self.store_deployment(current)
            
            # Create new active deployment
            import uuid
            deployment = Deployment(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                training_run_id=training_run_id,
                tuned_model_name=tuned_model_name,
                is_active=True,
                promoted_by=promoted_by,
            )
            
            if self.store_deployment(deployment):
                logger.info(f"Promoted deployment: {deployment.id} for agent {agent_id}")
                return deployment
            return None
            
        except Exception as e:
            logger.error(f"Failed to promote deployment: {e}")
            return None

    def rollback_deployment(
        self,
        agent_id: str,
        target_deployment_id: str,
        reason: Optional[str] = None,
        rolled_back_by: Optional[str] = None,
    ) -> Optional[Deployment]:
        """
        Rollback to a previous deployment.
        Creates a new deployment record referencing the rollback.
        """
        if not self._ensure_initialized():
            return None
        
        try:
            # Get the target deployment
            target = None
            query = "SELECT * FROM c WHERE c.id = @id AND c.agent_id = @agent_id"
            items = list(self._containers["deployments"].query_items(
                query=query,
                parameters=[
                    {"name": "@id", "value": target_deployment_id},
                    {"name": "@agent_id", "value": agent_id},
                ],
                partition_key=agent_id,
            ))
            if items:
                target = Deployment.from_dict(items[0])
            
            if not target:
                logger.error(f"Target deployment {target_deployment_id} not found")
                return None
            
            # Deactivate current
            current = self.get_active_deployment(agent_id)
            current_id = current.id if current else None
            if current:
                current.is_active = False
                self.store_deployment(current)
            
            # Create rollback deployment
            import uuid
            rollback = Deployment(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                training_run_id=target.training_run_id,
                tuned_model_name=target.tuned_model_name,
                is_active=True,
                promoted_by=rolled_back_by,
                rollback_from=current_id,
                rollback_reason=reason,
            )
            
            if self.store_deployment(rollback):
                logger.info(f"Rolled back to deployment: {rollback.id} for agent {agent_id}")
                return rollback
            return None
            
        except Exception as e:
            logger.error(f"Failed to rollback deployment: {e}")
            return None

    def list_deployments(
        self,
        agent_id: str,
        limit: int = 50,
    ) -> List[Deployment]:
        """List all deployments for an agent (deployment history)."""
        if not self._ensure_initialized():
            return []
        
        try:
            query = f"SELECT * FROM c WHERE c.agent_id = @agent_id ORDER BY c.promoted_at DESC OFFSET 0 LIMIT {limit}"
            items = list(self._containers["deployments"].query_items(
                query=query,
                parameters=[{"name": "@agent_id", "value": agent_id}],
                partition_key=agent_id,
            ))
            return [Deployment.from_dict(item) for item in items]
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to list deployments: {e.message}")
            return []

    # ==========================================
    # Utility Methods
    # ==========================================

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the RL Ledger connection."""
        result = {
            "initialized": self._initialized,
            "endpoint": self._endpoint[:50] + "..." if len(self._endpoint) > 50 else self._endpoint,
            "database": self._database_name,
            "auth_mode": self._auth_mode,
            "containers": {},
        }
        
        if self._ensure_initialized():
            for key, container in self._containers.items():
                try:
                    # Simple read to check container is accessible
                    list(container.query_items("SELECT TOP 1 c.id FROM c", enable_cross_partition_query=True))
                    result["containers"][key] = "healthy"
                except Exception as e:
                    result["containers"][key] = f"unhealthy: {str(e)[:50]}"
        
        return result


# Singleton instance (initialized lazily)
_rl_ledger_instance: Optional[RLLedgerCosmos] = None


def get_rl_ledger() -> RLLedgerCosmos:
    """Get the singleton RL Ledger instance."""
    global _rl_ledger_instance
    if _rl_ledger_instance is None:
        _rl_ledger_instance = RLLedgerCosmos()
    return _rl_ledger_instance
