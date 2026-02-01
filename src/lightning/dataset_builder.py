"""
Dataset Builder - Builds fine-tuning datasets from Cosmos episodes.

Queries episodes with rewards from the RL ledger and produces JSONL
files compatible with Azure OpenAI fine-tuning.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from .rl_ledger_cosmos import (
    RLLedgerCosmos,
    Dataset,
    Episode,
    Reward,
    RewardSource,
    get_rl_ledger,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset building."""
    output_dir: str = "./data/finetune"
    train_split: float = 0.8  # 80% training, 20% validation
    min_reward_threshold: float = 0.0  # Minimum avg reward for inclusion
    max_examples: int = 10000
    include_tool_calls: bool = True
    system_prompt: Optional[str] = None  # Optional system prompt to include
    
    @classmethod
    def from_env(cls) -> "DatasetConfig":
        """Create config from environment variables."""
        return cls(
            output_dir=os.getenv("LIGHTNING_DATA_DIR", "./data/finetune"),
            train_split=float(os.getenv("LIGHTNING_TRAIN_SPLIT", "0.8")),
            min_reward_threshold=float(os.getenv("LIGHTNING_MIN_REWARD", "0.0")),
            max_examples=int(os.getenv("LIGHTNING_MAX_EXAMPLES", "10000")),
            include_tool_calls=os.getenv("LIGHTNING_INCLUDE_TOOLS", "true").lower() == "true",
        )


class DatasetBuilder:
    """
    Builds fine-tuning datasets from Cosmos episodes.
    
    Produces JSONL files in the format expected by Azure OpenAI fine-tuning:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Usage:
        builder = DatasetBuilder.from_env()
        
        # Build dataset from rewarded episodes
        dataset = builder.build_dataset(
            agent_id="mcp-agents",
            name="customer-service-v1",
            min_reward=0.5,
        )
        
        # Get file paths
        train_file = dataset.local_path  # training.jsonl
        val_file = dataset.metadata.get("validation_path")  # validation.jsonl
    """
    
    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        ledger: Optional[RLLedgerCosmos] = None,
    ):
        """
        Initialize the dataset builder.
        
        Args:
            config: Dataset configuration
            ledger: Cosmos RL ledger
        """
        self.config = config or DatasetConfig.from_env()
        self._ledger = ledger
        self._output_dir_created = False

    def _ensure_output_dir(self):
        """Lazily create output directory when actually needed."""
        if not self._output_dir_created:
            try:
                os.makedirs(self.config.output_dir, exist_ok=True)
                self._output_dir_created = True
            except OSError as e:
                logger.warning(f"Could not create output directory {self.config.output_dir}: {e}")

    @property
    def ledger(self) -> RLLedgerCosmos:
        """Get the RL ledger (lazy initialization)."""
        if self._ledger is None:
            self._ledger = get_rl_ledger()
        return self._ledger

    @classmethod
    def from_env(cls) -> "DatasetBuilder":
        """Create a dataset builder from environment variables."""
        return cls(config=DatasetConfig.from_env())

    def _episode_to_messages(self, episode: Episode) -> List[Dict[str, str]]:
        """Convert an episode to the messages format for fine-tuning."""
        messages = []
        
        # Add system prompt if configured
        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt,
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": episode.user_input,
        })
        
        # Add tool calls if enabled and present
        if self.config.include_tool_calls and episode.tool_calls:
            # For fine-tuning, we represent tool calls in the assistant message
            # This creates a "trace" of the reasoning/action
            tool_trace = []
            for tc in episode.tool_calls:
                trace_entry = f"[Tool: {tc.tool_name}]"
                if tc.arguments:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    if len(args_str) <= 200:
                        trace_entry += f" Args: {args_str}"
                if tc.result and len(tc.result) <= 500:
                    trace_entry += f" → {tc.result}"
                elif tc.error:
                    trace_entry += f" → Error: {tc.error}"
                tool_trace.append(trace_entry)
            
            if tool_trace:
                # Include tool trace as part of the assistant response
                full_response = "\n".join(tool_trace) + "\n\n" + episode.assistant_output
            else:
                full_response = episode.assistant_output
        else:
            full_response = episode.assistant_output
        
        # Add assistant message
        messages.append({
            "role": "assistant",
            "content": full_response,
        })
        
        return messages

    def _split_data(
        self,
        data: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into training and validation sets."""
        import random
        random.shuffle(data)
        
        split_idx = int(len(data) * self.config.train_split)
        return data[:split_idx], data[split_idx:]

    def build_dataset(
        self,
        agent_id: str,
        name: str,
        description: Optional[str] = None,
        min_reward: Optional[float] = None,
        sources: Optional[List[RewardSource]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[Dataset]:
        """
        Build a fine-tuning dataset from rewarded episodes.
        
        Args:
            agent_id: ID of the agent
            name: Name for the dataset
            description: Optional description
            min_reward: Minimum average reward for inclusion (default from config)
            sources: Filter by reward sources
            start_date: Filter episodes after this date
            end_date: Filter episodes before this date
        
        Returns:
            Dataset manifest with file paths
        """
        min_reward = min_reward if min_reward is not None else self.config.min_reward_threshold
        
        logger.info(f"Building dataset '{name}' for agent {agent_id} with min_reward={min_reward}")
        
        # Query episodes with rewards
        episode_data = self.ledger.query_episodes_with_rewards(
            agent_id=agent_id,
            min_reward=min_reward,
            sources=sources,
            limit=self.config.max_examples,
        )
        
        if not episode_data:
            logger.warning(f"No episodes found matching criteria for dataset '{name}'")
            return None
        
        logger.info(f"Found {len(episode_data)} episodes with rewards")
        
        # Convert to training format
        training_examples = []
        episode_ids = []
        
        for item in episode_data:
            episode = item["episode"]
            avg_reward = item["avg_reward"]
            
            # Skip if below threshold
            if avg_reward < min_reward:
                continue
            
            messages = self._episode_to_messages(episode)
            training_examples.append({"messages": messages})
            episode_ids.append(episode.id)
        
        if not training_examples:
            logger.warning(f"No training examples after filtering for dataset '{name}'")
            return None
        
        logger.info(f"Prepared {len(training_examples)} training examples")
        
        # Split into train/val
        train_data, val_data = self._split_data(training_examples)
        
        # Ensure output directory exists (lazy creation)
        self._ensure_output_dir()
        
        # Generate file paths
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dataset_id = str(uuid.uuid4())
        
        train_filename = f"{name}_train_{timestamp}.jsonl"
        val_filename = f"{name}_val_{timestamp}.jsonl"
        
        train_path = os.path.join(self.config.output_dir, train_filename)
        val_path = os.path.join(self.config.output_dir, val_filename)
        
        # Write training file
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in train_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Write validation file
        with open(val_path, 'w', encoding='utf-8') as f:
            for example in val_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Wrote {len(train_data)} training examples to {train_path}")
        logger.info(f"Wrote {len(val_data)} validation examples to {val_path}")
        
        # Build query filter description
        query_filter = f"min_reward >= {min_reward}"
        if sources:
            query_filter += f", sources in [{', '.join(s.value for s in sources)}]"
        if start_date:
            query_filter += f", created_at >= {start_date}"
        if end_date:
            query_filter += f", created_at <= {end_date}"
        
        # Create dataset manifest
        dataset = Dataset(
            id=dataset_id,
            agent_id=agent_id,
            name=name,
            description=description,
            query_filter=query_filter,
            episode_ids=episode_ids,
            training_count=len(train_data),
            validation_count=len(val_data),
            local_path=train_path,
            reward_threshold=min_reward,
            metadata={
                "validation_path": val_path,
                "include_tool_calls": self.config.include_tool_calls,
                "train_split": self.config.train_split,
                "total_episodes": len(episode_data),
            },
        )
        
        # Store dataset manifest in Cosmos
        try:
            result = self.ledger.store_dataset(dataset)
            if result:
                logger.info(f"Dataset manifest stored: {dataset.id}")
        except Exception as e:
            logger.warning(f"Failed to store dataset manifest in Cosmos: {e}")
        
        return dataset

    def build_from_golden_conversations(
        self,
        agent_id: str,
        name: str,
        golden_file: str,
        description: Optional[str] = None,
    ) -> Optional[Dataset]:
        """
        Build a dataset from a file of golden conversations.
        
        The golden file should be a JSONL file with the same format as fine-tuning:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        
        Args:
            agent_id: ID of the agent
            name: Name for the dataset
            golden_file: Path to golden conversations JSONL file
            description: Optional description
        
        Returns:
            Dataset manifest
        """
        if not os.path.exists(golden_file):
            logger.error(f"Golden file not found: {golden_file}")
            return None
        
        # Load golden conversations
        golden_examples = []
        with open(golden_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        golden_examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid line: {e}")
        
        if not golden_examples:
            logger.error("No valid golden examples found")
            return None
        
        logger.info(f"Loaded {len(golden_examples)} golden conversations")
        
        # Split into train/val
        train_data, val_data = self._split_data(golden_examples)
        
        # Ensure output directory exists (lazy creation)
        self._ensure_output_dir()
        
        # Generate file paths
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        dataset_id = str(uuid.uuid4())
        
        train_filename = f"{name}_golden_train_{timestamp}.jsonl"
        val_filename = f"{name}_golden_val_{timestamp}.jsonl"
        
        train_path = os.path.join(self.config.output_dir, train_filename)
        val_path = os.path.join(self.config.output_dir, val_filename)
        
        # Write files
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in train_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        with open(val_path, 'w', encoding='utf-8') as f:
            for example in val_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Wrote {len(train_data)} training examples to {train_path}")
        logger.info(f"Wrote {len(val_data)} validation examples to {val_path}")
        
        # Create dataset manifest
        dataset = Dataset(
            id=dataset_id,
            agent_id=agent_id,
            name=name,
            description=description or f"Golden conversations from {golden_file}",
            query_filter="golden_conversations",
            training_count=len(train_data),
            validation_count=len(val_data),
            local_path=train_path,
            metadata={
                "validation_path": val_path,
                "source_file": golden_file,
                "total_golden": len(golden_examples),
            },
        )
        
        # Store manifest
        try:
            self.ledger.store_dataset(dataset)
        except Exception as e:
            logger.warning(f"Failed to store dataset manifest: {e}")
        
        return dataset

    def list_datasets(self, agent_id: str) -> List[Dataset]:
        """List all datasets for an agent."""
        return self.ledger.list_datasets(agent_id)

    def get_dataset(self, dataset_id: str, agent_id: str) -> Optional[Dataset]:
        """Get a dataset by ID."""
        return self.ledger.get_dataset(dataset_id, agent_id)


# Singleton instance
_dataset_builder_instance: Optional[DatasetBuilder] = None


def get_dataset_builder() -> DatasetBuilder:
    """Get the singleton dataset builder instance."""
    global _dataset_builder_instance
    if _dataset_builder_instance is None:
        _dataset_builder_instance = DatasetBuilder.from_env()
    return _dataset_builder_instance
