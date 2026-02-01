"""
Reward Writer - Attaches rewards/labels to episodes for RLHF.

Supports multiple reward sources:
- Evaluation scores (numeric)
- Human approval (boolean)
- Safety check outcomes
- Test results
- Cost/latency penalties
- Golden conversation matching
"""

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

from .rl_ledger_cosmos import (
    RLLedgerCosmos,
    Reward,
    RewardSource,
    get_rl_ledger,
)

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward writing."""
    default_agent_id: str = "mcp-agents"
    
    # Thresholds for auto-reward rules (configurable via env)
    eval_score_threshold: float = 0.7  # Below this = negative reward
    latency_penalty_threshold_ms: int = 10000  # Above this = penalty
    latency_penalty_value: float = -0.1
    
    # Normalization ranges
    eval_score_min: float = 0.0
    eval_score_max: float = 1.0
    
    @classmethod
    def from_env(cls) -> "RewardConfig":
        """Create config from environment variables."""
        return cls(
            default_agent_id=os.getenv("LIGHTNING_AGENT_ID", "mcp-agents"),
            eval_score_threshold=float(os.getenv("LIGHTNING_EVAL_THRESHOLD", "0.7")),
            latency_penalty_threshold_ms=int(os.getenv("LIGHTNING_LATENCY_THRESHOLD_MS", "10000")),
            latency_penalty_value=float(os.getenv("LIGHTNING_LATENCY_PENALTY", "-0.1")),
        )


class RewardWriter:
    """
    Writes rewards/labels to the RL ledger.
    
    Usage:
        writer = RewardWriter.from_env()
        
        # Human approval
        writer.record_human_approval(episode_id, agent_id, approved=True)
        
        # Evaluation score
        writer.record_eval_score(episode_id, agent_id, score=0.85, rubric="accuracy")
        
        # Test result
        writer.record_test_result(episode_id, agent_id, passed=False, test_name="intent_match")
        
        # Apply auto-reward rules
        writer.apply_auto_rewards(episode_id, agent_id, latency_ms=5000, eval_score=0.6)
    """
    
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        ledger: Optional[RLLedgerCosmos] = None,
    ):
        """
        Initialize the reward writer.
        
        Args:
            config: Reward configuration
            ledger: Cosmos RL ledger
        """
        self.config = config or RewardConfig.from_env()
        self._ledger = ledger

    @property
    def ledger(self) -> RLLedgerCosmos:
        """Get the RL ledger (lazy initialization)."""
        if self._ledger is None:
            self._ledger = get_rl_ledger()
        return self._ledger

    @classmethod
    def from_env(cls) -> "RewardWriter":
        """Create a reward writer from environment variables."""
        return cls(config=RewardConfig.from_env())

    def _normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize a score to [-1, 1] range."""
        if max_val == min_val:
            return 0.0
        # First normalize to [0, 1]
        normalized = (score - min_val) / (max_val - min_val)
        # Then scale to [-1, 1]
        return (normalized * 2) - 1

    def record_reward(
        self,
        episode_id: str,
        agent_id: str,
        source: RewardSource,
        value: float,
        raw_value: Optional[Any] = None,
        rubric: Optional[str] = None,
        evaluator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Reward]:
        """
        Record a reward for an episode.
        
        Args:
            episode_id: ID of the episode
            agent_id: ID of the agent
            source: Source of the reward
            value: Normalized reward value (-1 to 1)
            raw_value: Original value before normalization
            rubric: Evaluation rubric used
            evaluator: Who/what evaluated
            metadata: Additional metadata
        
        Returns:
            The stored Reward, or None on failure
        """
        reward = Reward(
            id=str(uuid.uuid4()),
            episode_id=episode_id,
            agent_id=agent_id,
            source=source,
            value=max(-1.0, min(1.0, value)),  # Clamp to [-1, 1]
            raw_value=raw_value,
            rubric=rubric,
            evaluator=evaluator,
            metadata=metadata or {},
        )
        
        try:
            result = self.ledger.store_reward(reward)
            if result:
                logger.info(f"Reward recorded: {reward.id} for episode {episode_id} (value={value:.3f})")
                return reward
        except Exception as e:
            logger.error(f"Failed to record reward: {e}")
        
        return None

    def record_human_approval(
        self,
        episode_id: str,
        agent_id: str,
        approved: bool,
        reviewer: Optional[str] = None,
        comments: Optional[str] = None,
    ) -> Optional[Reward]:
        """
        Record a human approval decision.
        
        Args:
            episode_id: ID of the episode
            agent_id: ID of the agent
            approved: Whether the response was approved
            reviewer: Name/ID of the reviewer
            comments: Optional review comments
        """
        return self.record_reward(
            episode_id=episode_id,
            agent_id=agent_id,
            source=RewardSource.HUMAN_APPROVAL,
            value=1.0 if approved else -1.0,
            raw_value=approved,
            evaluator=reviewer,
            metadata={"comments": comments} if comments else {},
        )

    def record_eval_score(
        self,
        episode_id: str,
        agent_id: str,
        score: float,
        rubric: Optional[str] = None,
        evaluator: Optional[str] = None,
        min_score: float = 0.0,
        max_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Reward]:
        """
        Record an evaluation score.
        
        Args:
            episode_id: ID of the episode
            agent_id: ID of the agent
            score: Raw evaluation score
            rubric: Evaluation rubric/criteria
            evaluator: Evaluation system/person
            min_score: Minimum possible score
            max_score: Maximum possible score
            metadata: Additional metadata
        """
        normalized = self._normalize_score(score, min_score, max_score)
        
        return self.record_reward(
            episode_id=episode_id,
            agent_id=agent_id,
            source=RewardSource.EVAL_SCORE,
            value=normalized,
            raw_value=score,
            rubric=rubric,
            evaluator=evaluator,
            metadata=metadata,
        )

    def record_test_result(
        self,
        episode_id: str,
        agent_id: str,
        passed: bool,
        test_name: Optional[str] = None,
        test_suite: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Reward]:
        """
        Record a test result.
        
        Args:
            episode_id: ID of the episode
            agent_id: ID of the agent
            passed: Whether the test passed
            test_name: Name of the test
            test_suite: Test suite name
            error_message: Error message if failed
        """
        return self.record_reward(
            episode_id=episode_id,
            agent_id=agent_id,
            source=RewardSource.TEST_RESULT,
            value=1.0 if passed else -1.0,
            raw_value=passed,
            rubric=test_name,
            evaluator=test_suite,
            metadata={"error_message": error_message} if error_message else {},
        )

    def record_safety_check(
        self,
        episode_id: str,
        agent_id: str,
        passed: bool,
        safety_category: Optional[str] = None,
        severity: Optional[str] = None,
        details: Optional[str] = None,
    ) -> Optional[Reward]:
        """
        Record a safety check result.
        
        Args:
            episode_id: ID of the episode
            agent_id: ID of the agent
            passed: Whether safety check passed
            safety_category: Category of safety check
            severity: Severity if failed
            details: Additional details
        """
        # Failed safety checks get severe negative reward
        value = 1.0 if passed else -1.0
        if not passed and severity == "critical":
            value = -1.0  # Maximum penalty for critical safety issues
        
        return self.record_reward(
            episode_id=episode_id,
            agent_id=agent_id,
            source=RewardSource.SAFETY_CHECK,
            value=value,
            raw_value=passed,
            rubric=safety_category,
            metadata={"severity": severity, "details": details} if not passed else {},
        )

    def record_golden_match(
        self,
        episode_id: str,
        agent_id: str,
        similarity_score: float,
        golden_id: Optional[str] = None,
        match_criteria: Optional[str] = None,
    ) -> Optional[Reward]:
        """
        Record a match against a golden conversation.
        
        Args:
            episode_id: ID of the episode
            agent_id: ID of the agent
            similarity_score: Similarity to golden (0-1)
            golden_id: ID of the golden conversation
            match_criteria: How similarity was measured
        """
        return self.record_reward(
            episode_id=episode_id,
            agent_id=agent_id,
            source=RewardSource.GOLDEN_CONVERSATION,
            value=self._normalize_score(similarity_score),
            raw_value=similarity_score,
            rubric=match_criteria,
            metadata={"golden_id": golden_id} if golden_id else {},
        )

    def apply_auto_rewards(
        self,
        episode_id: str,
        agent_id: str,
        latency_ms: Optional[int] = None,
        eval_score: Optional[float] = None,
        test_passed: Optional[bool] = None,
        safety_passed: Optional[bool] = None,
    ) -> List[Reward]:
        """
        Apply automatic reward rules based on episode metrics.
        
        This is a convenience method that applies configurable rules:
        - Latency penalty if above threshold
        - Eval score converted to reward
        - Test pass/fail converted to reward
        - Safety check pass/fail converted to reward
        
        Args:
            episode_id: ID of the episode
            agent_id: ID of the agent
            latency_ms: Request latency in milliseconds
            eval_score: Evaluation score (0-1)
            test_passed: Whether associated test passed
            safety_passed: Whether safety check passed
        
        Returns:
            List of rewards that were recorded
        """
        rewards = []
        
        # Latency penalty
        if latency_ms is not None and latency_ms > self.config.latency_penalty_threshold_ms:
            reward = self.record_reward(
                episode_id=episode_id,
                agent_id=agent_id,
                source=RewardSource.LATENCY_PENALTY,
                value=self.config.latency_penalty_value,
                raw_value=latency_ms,
                rubric=f"latency > {self.config.latency_penalty_threshold_ms}ms",
            )
            if reward:
                rewards.append(reward)
        
        # Eval score
        if eval_score is not None:
            reward = self.record_eval_score(
                episode_id=episode_id,
                agent_id=agent_id,
                score=eval_score,
                evaluator="auto",
            )
            if reward:
                rewards.append(reward)
        
        # Test result
        if test_passed is not None:
            reward = self.record_test_result(
                episode_id=episode_id,
                agent_id=agent_id,
                passed=test_passed,
                test_suite="auto",
            )
            if reward:
                rewards.append(reward)
        
        # Safety check
        if safety_passed is not None:
            reward = self.record_safety_check(
                episode_id=episode_id,
                agent_id=agent_id,
                passed=safety_passed,
            )
            if reward:
                rewards.append(reward)
        
        return rewards

    def get_episode_rewards(self, episode_id: str, agent_id: str) -> List[Reward]:
        """Get all rewards for an episode."""
        return self.ledger.get_rewards_for_episode(episode_id, agent_id)

    def get_average_reward(self, episode_id: str, agent_id: str) -> Optional[float]:
        """Calculate the average reward for an episode."""
        rewards = self.get_episode_rewards(episode_id, agent_id)
        if not rewards:
            return None
        return sum(r.value for r in rewards) / len(rewards)


# Singleton instance
_reward_writer_instance: Optional[RewardWriter] = None


def get_reward_writer() -> RewardWriter:
    """Get the singleton reward writer instance."""
    global _reward_writer_instance
    if _reward_writer_instance is None:
        _reward_writer_instance = RewardWriter.from_env()
    return _reward_writer_instance
