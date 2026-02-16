"""
Training Runner - Runs Agent Lightning fine-tuning via Azure OpenAI.

Manages the fine-tuning lifecycle:
1. Upload dataset files to Azure OpenAI
2. Create and monitor fine-tuning job
3. Track progress in Cosmos
4. Create deployments for tuned models
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

from azure.identity import DefaultAzureCredential

from .rl_ledger_cosmos import (
    RLLedgerCosmos,
    Dataset,
    TrainingRun,
    TrainingStatus,
    get_rl_ledger,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    base_model: str = "gpt-4o-mini-2024-07-18"  # Base model for fine-tuning
    n_epochs: int = 3
    batch_size: str = "auto"
    learning_rate_multiplier: float = 1.0
    suffix: Optional[str] = None  # Custom suffix for model name
    poll_interval_seconds: int = 30
    max_wait_minutes: int = 120
    
    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Create config from environment variables."""
        return cls(
            base_model=os.getenv("LIGHTNING_BASE_MODEL", "gpt-4o-mini-2024-07-18"),
            n_epochs=int(os.getenv("LIGHTNING_EPOCHS", "3")),
            learning_rate_multiplier=float(os.getenv("LIGHTNING_LR_MULTIPLIER", "1.0")),
            suffix=os.getenv("LIGHTNING_MODEL_SUFFIX"),
            poll_interval_seconds=int(os.getenv("LIGHTNING_POLL_INTERVAL", "30")),
            max_wait_minutes=int(os.getenv("LIGHTNING_MAX_WAIT_MINUTES", "120")),
        )


class TrainingRunner:
    """
    Runs fine-tuning jobs via Azure OpenAI.
    
    Usage:
        runner = TrainingRunner.from_env()
        
        # Start training
        run = runner.start_training(
            dataset_id="dataset-123",
            agent_id="mcp-agents",
        )
        
        # Wait for completion
        run = runner.wait_for_completion(run.id, run.agent_id)
        
        # Or use the all-in-one method
        run = runner.run_training(dataset_id="dataset-123", agent_id="mcp-agents")
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        ledger: Optional[RLLedgerCosmos] = None,
        aoai_endpoint: Optional[str] = None,
        credential: Optional[Any] = None,
    ):
        """
        Initialize the training runner.
        
        Args:
            config: Training configuration
            ledger: Cosmos RL ledger
            aoai_endpoint: Azure OpenAI endpoint
            credential: Azure credential
        """
        self.config = config or TrainingConfig.from_env()
        self._ledger = ledger
        self._aoai_endpoint = aoai_endpoint or os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")
        self._credential = credential
        self._client = None

    @property
    def ledger(self) -> RLLedgerCosmos:
        """Get the RL ledger (lazy initialization)."""
        if self._ledger is None:
            self._ledger = get_rl_ledger()
        return self._ledger

    @property
    def client(self):
        """Get the Azure OpenAI client (lazy initialization)."""
        if self._client is None:
            if not self._aoai_endpoint:
                raise ValueError("Azure OpenAI endpoint not configured. Set FOUNDRY_PROJECT_ENDPOINT.")
            
            from openai import AzureOpenAI
            
            if self._credential is None:
                self._credential = DefaultAzureCredential()
            
            # Get token for Azure Cognitive Services
            token = self._credential.get_token("https://cognitiveservices.azure.com/.default")
            
            # Extract base endpoint
            base_endpoint = self._aoai_endpoint.split('/api/projects')[0] if '/api/projects' in self._aoai_endpoint else self._aoai_endpoint
            
            self._client = AzureOpenAI(
                azure_endpoint=base_endpoint,
                api_key=token.token,
                api_version="2025-04-01-preview",  # Fine-tuning API version (supports GlobalStandard training)
            )
        
        return self._client

    @classmethod
    def from_env(cls) -> "TrainingRunner":
        """Create a training runner from environment variables."""
        return cls(config=TrainingConfig.from_env())

    def _upload_file(self, file_path: str, purpose: str = "fine-tune") -> str:
        """Upload a file to Azure OpenAI, wait for processing, and return the file ID."""
        with open(file_path, "rb") as f:
            response = self.client.files.create(file=f, purpose=purpose)
        
        file_id = response.id
        logger.info(f"Uploaded file: {file_path} -> {file_id}, status={response.status}")
        
        # Wait for file processing to complete
        max_wait = 120  # seconds
        poll_interval = 5
        elapsed = 0
        while elapsed < max_wait:
            file_info = self.client.files.retrieve(file_id)
            status = file_info.status
            if status == "processed":
                logger.info(f"File {file_id} processing complete")
                return file_id
            elif status in ("error", "deleting", "deleted"):
                raise RuntimeError(f"File {file_id} processing failed with status: {status}")
            logger.info(f"File {file_id} status: {status}, waiting...")
            time.sleep(poll_interval)
            elapsed += poll_interval
        
        raise RuntimeError(f"File {file_id} processing timed out after {max_wait}s (status: {status})")
        return file_id

    def start_training(
        self,
        dataset_id: str,
        agent_id: str,
        base_model: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[TrainingRun]:
        """
        Start a fine-tuning job.
        
        Args:
            dataset_id: ID of the dataset to use
            agent_id: ID of the agent
            base_model: Base model to fine-tune (default from config)
            hyperparameters: Custom hyperparameters
        
        Returns:
            TrainingRun record, or None on failure
        """
        # Get dataset
        dataset = self.ledger.get_dataset(dataset_id, agent_id)
        if not dataset:
            logger.error(f"Dataset {dataset_id} not found")
            return None
        
        if not dataset.local_path or not os.path.exists(dataset.local_path):
            logger.error(f"Training file not found: {dataset.local_path}")
            return None
        
        base_model = base_model or self.config.base_model
        
        # Build hyperparameters (exclude batch_size="auto" — Azure OpenAI rejects string values)
        hp = {
            "n_epochs": self.config.n_epochs,
            "learning_rate_multiplier": self.config.learning_rate_multiplier,
        }
        if self.config.batch_size != "auto":
            hp["batch_size"] = int(self.config.batch_size)
        if hyperparameters:
            # Filter out invalid batch_size
            clean_hp = {k: v for k, v in hyperparameters.items() if not (k == "batch_size" and v == "auto")}
            hp.update(clean_hp)
        
        # Create training run record
        run = TrainingRun(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            dataset_id=dataset_id,
            base_model=base_model,
            hyperparameters=hp,
            status=TrainingStatus.PENDING,
        )
        
        # Store initial record
        self.ledger.store_training_run(run)
        
        try:
            # Upload training file
            logger.info(f"Uploading training file: {dataset.local_path}")
            train_file_id = self._upload_file(dataset.local_path)
            
            # Upload validation file if available
            val_file_id = None
            val_path = dataset.metadata.get("validation_path")
            if val_path and os.path.exists(val_path):
                logger.info(f"Uploading validation file: {val_path}")
                val_file_id = self._upload_file(val_path)
            
            # Create fine-tuning job
            logger.info(f"Creating fine-tuning job with base model: {base_model}")
            
            job_params = {
                "training_file": train_file_id,
                "model": base_model,
                "hyperparameters": hp,
            }
            
            if val_file_id:
                job_params["validation_file"] = val_file_id
            
            if self.config.suffix:
                job_params["suffix"] = self.config.suffix
            
            # Use GlobalStandard training tier for cross-region fine-tuning
            job_params["extra_body"] = {"trainingType": "GlobalStandard"}
            
            job = self.client.fine_tuning.jobs.create(**job_params)
            
            logger.info(f"Fine-tuning job created: {job.id}")
            
            # Update run with job ID
            run.aoai_job_id = job.id
            run.status = TrainingStatus.RUNNING
            run.started_at = datetime.utcnow().isoformat()
            run.metadata["train_file_id"] = train_file_id
            if val_file_id:
                run.metadata["val_file_id"] = val_file_id
            
            self.ledger.store_training_run(run)
            
            return run
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            run.status = TrainingStatus.FAILED
            run.error_message = str(e)
            run.completed_at = datetime.utcnow().isoformat()
            self.ledger.store_training_run(run)
            return run

    def check_status(self, run_id: str, agent_id: str) -> Optional[TrainingRun]:
        """
        Check the status of a training run.
        
        Args:
            run_id: ID of the training run
            agent_id: ID of the agent
        
        Returns:
            Updated TrainingRun, or None if not found
        """
        run = self.ledger.get_training_run(run_id, agent_id)
        if not run:
            return None
        
        if not run.aoai_job_id:
            return run
        
        if run.status in (TrainingStatus.SUCCEEDED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
            # Already terminal state
            return run
        
        try:
            job = self.client.fine_tuning.jobs.retrieve(run.aoai_job_id)
            
            # Map AOAI status to our status
            status_map = {
                "validating_files": TrainingStatus.RUNNING,
                "queued": TrainingStatus.RUNNING,
                "running": TrainingStatus.RUNNING,
                "succeeded": TrainingStatus.SUCCEEDED,
                "failed": TrainingStatus.FAILED,
                "cancelled": TrainingStatus.CANCELLED,
            }
            
            new_status = status_map.get(job.status, TrainingStatus.RUNNING)
            
            # Update run
            run.status = new_status
            
            if job.status == "succeeded":
                run.tuned_model_name = job.fine_tuned_model
                run.completed_at = datetime.utcnow().isoformat()
                
                # Extract metrics if available
                if hasattr(job, 'result_files') and job.result_files:
                    run.metadata["result_files"] = job.result_files
                
                logger.info(f"Training succeeded! Tuned model: {run.tuned_model_name}")
            
            elif job.status == "failed":
                run.error_message = getattr(job, 'error', {}).get('message', 'Unknown error')
                run.completed_at = datetime.utcnow().isoformat()
                logger.error(f"Training failed: {run.error_message}")
            
            elif job.status == "cancelled":
                run.completed_at = datetime.utcnow().isoformat()
                logger.info("Training was cancelled")
            
            # Store updated run
            self.ledger.store_training_run(run)
            
            return run
            
        except Exception as e:
            logger.error(f"Failed to check training status: {e}")
            return run

    def wait_for_completion(
        self,
        run_id: str,
        agent_id: str,
        poll_interval: Optional[int] = None,
        max_wait_minutes: Optional[int] = None,
    ) -> Optional[TrainingRun]:
        """
        Wait for a training run to complete.
        
        Args:
            run_id: ID of the training run
            agent_id: ID of the agent
            poll_interval: Seconds between status checks
            max_wait_minutes: Maximum minutes to wait
        
        Returns:
            Final TrainingRun, or None if not found
        """
        poll_interval = poll_interval or self.config.poll_interval_seconds
        max_wait_minutes = max_wait_minutes or self.config.max_wait_minutes
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        
        logger.info(f"Waiting for training run {run_id} to complete (max {max_wait_minutes} minutes)...")
        
        while True:
            run = self.check_status(run_id, agent_id)
            if not run:
                return None
            
            if run.status in (TrainingStatus.SUCCEEDED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
                return run
            
            elapsed = time.time() - start_time
            if elapsed > max_wait_seconds:
                logger.warning(f"Timeout waiting for training run {run_id}")
                run.status = TrainingStatus.FAILED
                run.error_message = f"Timeout after {max_wait_minutes} minutes"
                self.ledger.store_training_run(run)
                return run
            
            logger.info(f"Training status: {run.status.value} (elapsed: {int(elapsed)}s)")
            time.sleep(poll_interval)

    def run_training(
        self,
        dataset_id: str,
        agent_id: str,
        base_model: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        wait: bool = True,
    ) -> Optional[TrainingRun]:
        """
        Run a complete fine-tuning job.
        
        Args:
            dataset_id: ID of the dataset to use
            agent_id: ID of the agent
            base_model: Base model to fine-tune
            hyperparameters: Custom hyperparameters
            wait: Whether to wait for completion
        
        Returns:
            Final TrainingRun
        """
        run = self.start_training(
            dataset_id=dataset_id,
            agent_id=agent_id,
            base_model=base_model,
            hyperparameters=hyperparameters,
        )
        
        if not run or run.status == TrainingStatus.FAILED:
            return run
        
        if wait:
            run = self.wait_for_completion(run.id, agent_id)
        
        return run

    def cancel_training(self, run_id: str, agent_id: str) -> bool:
        """
        Cancel a training run.
        
        Args:
            run_id: ID of the training run
            agent_id: ID of the agent
        
        Returns:
            True if cancelled successfully
        """
        run = self.ledger.get_training_run(run_id, agent_id)
        if not run or not run.aoai_job_id:
            return False
        
        try:
            self.client.fine_tuning.jobs.cancel(run.aoai_job_id)
            run.status = TrainingStatus.CANCELLED
            run.completed_at = datetime.utcnow().isoformat()
            self.ledger.store_training_run(run)
            logger.info(f"Training run {run_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel training: {e}")
            return False

    def list_runs(
        self,
        agent_id: str,
        status: Optional[TrainingStatus] = None,
    ) -> List[TrainingRun]:
        """List training runs for an agent."""
        return self.ledger.list_training_runs(agent_id, status)

    def get_run(self, run_id: str, agent_id: str) -> Optional[TrainingRun]:
        """Get a training run by ID."""
        return self.ledger.get_training_run(run_id, agent_id)


# Singleton instance
_training_runner_instance: Optional[TrainingRunner] = None


def get_training_runner() -> TrainingRunner:
    """Get the singleton training runner instance."""
    global _training_runner_instance
    if _training_runner_instance is None:
        _training_runner_instance = TrainingRunner.from_env()
    return _training_runner_instance
