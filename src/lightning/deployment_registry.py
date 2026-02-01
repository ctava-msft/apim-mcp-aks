"""
Deployment Registry - Manages active tuned model deployments.

Provides the mechanism for:
- Promoting a tuned model to active
- Selecting the active model for an agent
- Rolling back to previous deployments
- Deployment history tracking
"""

import logging
import os
from typing import Optional, List

from .rl_ledger_cosmos import (
    RLLedgerCosmos,
    Deployment,
    TrainingRun,
    TrainingStatus,
    get_rl_ledger,
)

logger = logging.getLogger(__name__)


class DeploymentRegistry:
    """
    Manages tuned model deployments.
    
    This registry is queried by next_best_action_agent.py to determine which model
    deployment to use for a given agent.
    
    Usage:
        registry = DeploymentRegistry.from_env()
        
        # Promote a tuned model
        deployment = registry.promote(
            agent_id="mcp-agents",
            training_run_id="run-123",
            promoted_by="admin@example.com",
        )
        
        # Get active model for agent
        model_name = registry.get_active_model(agent_id="mcp-agents")
        
        # Rollback to previous deployment
        registry.rollback(agent_id="mcp-agents", reason="Performance regression")
    """
    
    def __init__(
        self,
        ledger: Optional[RLLedgerCosmos] = None,
        fallback_model: Optional[str] = None,
    ):
        """
        Initialize the deployment registry.
        
        Args:
            ledger: Cosmos RL ledger
            fallback_model: Fallback model if no active deployment
        """
        self._ledger = ledger
        self._fallback_model = fallback_model or os.getenv("TUNED_MODEL_DEPLOYMENT_NAME")

    @property
    def ledger(self) -> RLLedgerCosmos:
        """Get the RL ledger (lazy initialization)."""
        if self._ledger is None:
            self._ledger = get_rl_ledger()
        return self._ledger

    @classmethod
    def from_env(cls) -> "DeploymentRegistry":
        """Create a deployment registry from environment variables."""
        return cls()

    def promote(
        self,
        agent_id: str,
        training_run_id: str,
        promoted_by: Optional[str] = None,
    ) -> Optional[Deployment]:
        """
        Promote a tuned model to active deployment.
        
        Args:
            agent_id: ID of the agent
            training_run_id: ID of the training run with the tuned model
            promoted_by: Who is promoting (for audit trail)
        
        Returns:
            The new active Deployment, or None on failure
        """
        # Get training run
        run = self.ledger.get_training_run(training_run_id, agent_id)
        if not run:
            logger.error(f"Training run {training_run_id} not found")
            return None
        
        if run.status != TrainingStatus.SUCCEEDED:
            logger.error(f"Training run {training_run_id} has not succeeded (status={run.status.value})")
            return None
        
        if not run.tuned_model_name:
            logger.error(f"Training run {training_run_id} has no tuned model name")
            return None
        
        # Promote deployment
        deployment = self.ledger.promote_deployment(
            agent_id=agent_id,
            training_run_id=training_run_id,
            tuned_model_name=run.tuned_model_name,
            promoted_by=promoted_by,
        )
        
        if deployment:
            logger.info(f"Promoted deployment {deployment.id} for agent {agent_id}: {deployment.tuned_model_name}")
        
        return deployment

    def get_active_deployment(self, agent_id: str) -> Optional[Deployment]:
        """
        Get the currently active deployment for an agent.
        
        Args:
            agent_id: ID of the agent
        
        Returns:
            Active Deployment, or None if no active deployment
        """
        return self.ledger.get_active_deployment(agent_id)

    def get_active_model(self, agent_id: str) -> Optional[str]:
        """
        Get the active tuned model name for an agent.
        
        This is the primary method called by next_best_action_agent.py to determine
        which model to use.
        
        Selection order:
        1. Active deployment in Cosmos
        2. TUNED_MODEL_DEPLOYMENT_NAME env var
        3. None (use base model)
        
        Args:
            agent_id: ID of the agent
        
        Returns:
            Tuned model deployment name, or None to use base model
        """
        # Check if tuned models are enabled
        if os.getenv("USE_TUNED_MODEL", "false").lower() != "true":
            return None
        
        # Try to get active deployment from Cosmos
        try:
            deployment = self.ledger.get_active_deployment(agent_id)
            if deployment:
                logger.debug(f"Using tuned model from Cosmos: {deployment.tuned_model_name}")
                return deployment.tuned_model_name
        except Exception as e:
            logger.warning(f"Failed to get active deployment from Cosmos: {e}")
        
        # Fallback to env var
        if self._fallback_model:
            logger.debug(f"Using fallback tuned model: {self._fallback_model}")
            return self._fallback_model
        
        # No tuned model available
        return None

    def rollback(
        self,
        agent_id: str,
        target_deployment_id: Optional[str] = None,
        reason: Optional[str] = None,
        rolled_back_by: Optional[str] = None,
    ) -> Optional[Deployment]:
        """
        Rollback to a previous deployment.
        
        If no target is specified, rolls back to the most recent previous deployment.
        
        Args:
            agent_id: ID of the agent
            target_deployment_id: Specific deployment to roll back to (optional)
            reason: Reason for rollback
            rolled_back_by: Who is rolling back (for audit trail)
        
        Returns:
            The new active Deployment, or None on failure
        """
        # If no target specified, find the most recent previous deployment
        if not target_deployment_id:
            deployments = self.ledger.list_deployments(agent_id, limit=10)
            
            # Find a non-active deployment to roll back to
            for dep in deployments:
                if not dep.is_active:
                    target_deployment_id = dep.id
                    break
            
            if not target_deployment_id:
                logger.error("No previous deployment found to roll back to")
                return None
        
        # Perform rollback
        deployment = self.ledger.rollback_deployment(
            agent_id=agent_id,
            target_deployment_id=target_deployment_id,
            reason=reason,
            rolled_back_by=rolled_back_by,
        )
        
        if deployment:
            logger.info(f"Rolled back to deployment {deployment.id}: {deployment.tuned_model_name}")
        
        return deployment

    def deactivate(self, agent_id: str, reason: Optional[str] = None) -> bool:
        """
        Deactivate the current deployment (revert to base model).
        
        Args:
            agent_id: ID of the agent
            reason: Reason for deactivation
        
        Returns:
            True if deactivated successfully
        """
        current = self.ledger.get_active_deployment(agent_id)
        if not current:
            logger.info(f"No active deployment for agent {agent_id}")
            return True
        
        current.is_active = False
        current.metadata["deactivation_reason"] = reason
        
        result = self.ledger.store_deployment(current)
        if result:
            logger.info(f"Deactivated deployment {current.id} for agent {agent_id}")
            return True
        
        return False

    def list_deployments(self, agent_id: str, limit: int = 50) -> List[Deployment]:
        """
        List deployment history for an agent.
        
        Args:
            agent_id: ID of the agent
            limit: Maximum number to return
        
        Returns:
            List of Deployments, most recent first
        """
        return self.ledger.list_deployments(agent_id, limit)

    def get_deployment_lineage(self, agent_id: str) -> List[dict]:
        """
        Get the full lineage of a deployment chain.
        
        Returns a list showing the progression of deployments, including
        which training runs and datasets were used.
        
        Args:
            agent_id: ID of the agent
        
        Returns:
            List of lineage entries with deployment, training run, and dataset info
        """
        deployments = self.list_deployments(agent_id)
        lineage = []
        
        for dep in deployments:
            entry = {
                "deployment_id": dep.id,
                "tuned_model": dep.tuned_model_name,
                "is_active": dep.is_active,
                "promoted_at": dep.promoted_at,
                "promoted_by": dep.promoted_by,
                "rollback_from": dep.rollback_from,
                "rollback_reason": dep.rollback_reason,
            }
            
            # Get training run info
            run = self.ledger.get_training_run(dep.training_run_id, agent_id)
            if run:
                entry["training_run"] = {
                    "id": run.id,
                    "base_model": run.base_model,
                    "dataset_id": run.dataset_id,
                    "status": run.status.value,
                    "started_at": run.started_at,
                    "completed_at": run.completed_at,
                }
                
                # Get dataset info
                dataset = self.ledger.get_dataset(run.dataset_id, agent_id)
                if dataset:
                    entry["dataset"] = {
                        "id": dataset.id,
                        "name": dataset.name,
                        "training_count": dataset.training_count,
                        "validation_count": dataset.validation_count,
                        "reward_threshold": dataset.reward_threshold,
                    }
            
            lineage.append(entry)
        
        return lineage


# Singleton instance
_deployment_registry_instance: Optional[DeploymentRegistry] = None


def get_deployment_registry() -> DeploymentRegistry:
    """Get the singleton deployment registry instance."""
    global _deployment_registry_instance
    if _deployment_registry_instance is None:
        _deployment_registry_instance = DeploymentRegistry.from_env()
    return _deployment_registry_instance
