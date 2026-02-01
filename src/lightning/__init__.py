"""
Agent Lightning - Reinforcement Learning and Fine-Tuning for MCP Agents

This module provides:
- Episode capture for training data collection
- Reward/label tracking for RLHF
- Dataset building for fine-tuning
- Training run management
- Deployment registry for tuned models
- Cosmos DB persistence for all RL artifacts
"""

from .rl_ledger_cosmos import (
    RLLedgerCosmos,
    Episode,
    EpisodeToolCall,
    Reward,
    RewardSource,
    Dataset,
    TrainingRun,
    TrainingStatus,
    Deployment,
    get_rl_ledger,
)
from .episode_capture import EpisodeCaptureHook, CaptureConfig, get_capture_hook
from .reward_writer import RewardWriter, RewardConfig, get_reward_writer
from .dataset_builder import DatasetBuilder, DatasetConfig, get_dataset_builder
from .training_runner import TrainingRunner, TrainingConfig, get_training_runner
from .deployment_registry import DeploymentRegistry, get_deployment_registry

__all__ = [
    # Core entities
    "Episode",
    "EpisodeToolCall",
    "Reward",
    "RewardSource",
    "Dataset",
    "TrainingRun",
    "TrainingStatus",
    "Deployment",
    # Cosmos ledger
    "RLLedgerCosmos",
    "get_rl_ledger",
    # Components
    "EpisodeCaptureHook",
    "CaptureConfig",
    "get_capture_hook",
    "RewardWriter",
    "RewardConfig",
    "get_reward_writer",
    "DatasetBuilder",
    "DatasetConfig",
    "get_dataset_builder",
    "TrainingRunner",
    "TrainingConfig",
    "get_training_runner",
    "DeploymentRegistry",
    "get_deployment_registry",
]
