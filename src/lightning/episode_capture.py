"""
Episode Capture Hook - Captures agent interactions for training data.

This hook intercepts agent requests and responses to build training datasets.
Enabled via ENABLE_LIGHTNING_CAPTURE=true environment variable.
"""

import logging
import hashlib
import time
import os
import re
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

from .rl_ledger_cosmos import (
    RLLedgerCosmos,
    Episode,
    EpisodeToolCall,
    get_rl_ledger,
)

logger = logging.getLogger(__name__)


# Patterns for redacting sensitive data
REDACT_PATTERNS = [
    (re.compile(r'(bearer\s+)[a-zA-Z0-9\-_\.]+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(api[_-]?key["\s:=]+)[a-zA-Z0-9\-_]+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(password["\s:=]+)[^\s"]+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(secret["\s:=]+)[^\s"]+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(token["\s:=]+)[a-zA-Z0-9\-_\.]+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(connection[_-]?string["\s:=]+)[^\s"]+', re.IGNORECASE), r'\1[REDACTED]'),
    # Azure-specific
    (re.compile(r'(AccountKey=)[^;]+', re.IGNORECASE), r'\1[REDACTED]'),
    (re.compile(r'(SharedAccessSignature=)[^;]+', re.IGNORECASE), r'\1[REDACTED]'),
]


def redact_sensitive_data(text: str) -> str:
    """Redact sensitive patterns from text."""
    if not text:
        return text
    
    result = text
    for pattern, replacement in REDACT_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def hash_instructions(instructions: str) -> str:
    """Generate a hash of agent instructions for tracking."""
    if not instructions:
        return ""
    return hashlib.sha256(instructions.encode()).hexdigest()[:16]


@dataclass
class CaptureConfig:
    """Configuration for episode capture."""
    enabled: bool = False
    agent_id: str = "default"
    local_fallback_dir: Optional[str] = None
    max_output_length: int = 10000  # Max chars for tool results
    redact_secrets: bool = True
    capture_embeddings: bool = False  # Include embeddings in episodes
    
    @classmethod
    def from_env(cls) -> "CaptureConfig":
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("ENABLE_LIGHTNING_CAPTURE", "false").lower() == "true",
            agent_id=os.getenv("LIGHTNING_AGENT_ID", "mcp-agents"),
            local_fallback_dir=os.getenv("LIGHTNING_DATA_DIR", "./data/finetune"),
            max_output_length=int(os.getenv("LIGHTNING_MAX_OUTPUT_LENGTH", "10000")),
            redact_secrets=os.getenv("LIGHTNING_REDACT_SECRETS", "true").lower() == "true",
            capture_embeddings=os.getenv("LIGHTNING_CAPTURE_EMBEDDINGS", "false").lower() == "true",
        )


@dataclass
class CaptureContext:
    """Context for an ongoing capture."""
    episode_id: str
    agent_id: str
    start_time: float
    user_input: str
    instructions_hash: Optional[str] = None
    model_deployment: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    tool_calls: List[EpisodeToolCall] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodeCaptureHook:
    """
    Hook for capturing agent interactions.
    
    Usage:
        hook = EpisodeCaptureHook.from_env()
        
        # Start capture before processing
        ctx = hook.start_capture(user_input, model_deployment, correlation_id)
        
        # Record tool calls during processing
        hook.record_tool_call(ctx, tool_name, arguments, result, duration_ms)
        
        # End capture after response
        episode = hook.end_capture(ctx, assistant_output, token_usage)
    """
    
    def __init__(
        self,
        config: Optional[CaptureConfig] = None,
        ledger: Optional[RLLedgerCosmos] = None,
    ):
        """
        Initialize the capture hook.
        
        Args:
            config: Capture configuration (uses env vars if not provided)
            ledger: Cosmos RL ledger (uses singleton if not provided)
        """
        self.config = config or CaptureConfig.from_env()
        self._ledger = ledger
        self._local_fallback_file = None
        self._local_fallback_dir_created = False

    def _ensure_fallback_dir(self):
        """Lazily create fallback directory when actually needed."""
        if not self._local_fallback_dir_created and self.config.local_fallback_dir:
            try:
                os.makedirs(self.config.local_fallback_dir, exist_ok=True)
                self._local_fallback_dir_created = True
            except OSError as e:
                logger.warning(f"Could not create fallback directory {self.config.local_fallback_dir}: {e}")

    def _get_fallback_file(self) -> Optional[str]:
        """Get the fallback file path, creating directory if needed."""
        if not self.config.local_fallback_dir:
            return None
        self._ensure_fallback_dir()
        return os.path.join(
            self.config.local_fallback_dir,
            f"episodes_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        )

    @property
    def ledger(self) -> RLLedgerCosmos:
        """Get the RL ledger (lazy initialization)."""
        if self._ledger is None:
            self._ledger = get_rl_ledger()
        return self._ledger

    @classmethod
    def from_env(cls) -> "EpisodeCaptureHook":
        """Create a capture hook from environment variables."""
        return cls(config=CaptureConfig.from_env())

    def is_enabled(self) -> bool:
        """Check if capture is enabled."""
        return self.config.enabled

    def start_capture(
        self,
        user_input: str,
        model_deployment: Optional[str] = None,
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        instructions: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CaptureContext:
        """
        Start capturing an episode.
        
        Args:
            user_input: The user's input message
            model_deployment: Model deployment name
            correlation_id: Request correlation ID
            session_id: Session ID
            instructions: Agent instructions (will be hashed)
            metadata: Additional metadata
        
        Returns:
            CaptureContext for tracking the episode
        """
        return CaptureContext(
            episode_id=str(uuid.uuid4()),
            agent_id=self.config.agent_id,
            start_time=time.time(),
            user_input=user_input,
            instructions_hash=hash_instructions(instructions) if instructions else None,
            model_deployment=model_deployment,
            correlation_id=correlation_id,
            session_id=session_id,
            metadata=metadata or {},
        )

    def record_tool_call(
        self,
        ctx: CaptureContext,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Optional[str] = None,
        duration_ms: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Record a tool call within an episode.
        
        Args:
            ctx: Capture context
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result (will be redacted and truncated)
            duration_ms: Duration in milliseconds
            error: Error message if tool failed
        """
        if not self.is_enabled():
            return
        
        # Redact sensitive data from arguments and result
        safe_args = arguments.copy()
        if self.config.redact_secrets:
            for key, value in safe_args.items():
                if isinstance(value, str):
                    safe_args[key] = redact_sensitive_data(value)
        
        safe_result = None
        if result:
            safe_result = result
            if self.config.redact_secrets:
                safe_result = redact_sensitive_data(safe_result)
            # Truncate long results
            if len(safe_result) > self.config.max_output_length:
                safe_result = safe_result[:self.config.max_output_length] + "...[TRUNCATED]"
        
        tool_call = EpisodeToolCall(
            tool_name=tool_name,
            arguments=safe_args,
            result=safe_result,
            duration_ms=duration_ms,
            error=error,
        )
        ctx.tool_calls.append(tool_call)

    def end_capture(
        self,
        ctx: CaptureContext,
        assistant_output: str,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> Optional[Episode]:
        """
        End capture and store the episode.
        
        Args:
            ctx: Capture context
            assistant_output: The assistant's final response
            token_usage: Token usage stats
        
        Returns:
            The stored Episode, or None if disabled/failed
        """
        if not self.is_enabled():
            return None
        
        # Calculate request latency
        latency_ms = int((time.time() - ctx.start_time) * 1000)
        
        # Build episode
        episode = Episode(
            id=ctx.episode_id,
            agent_id=ctx.agent_id,
            user_input=ctx.user_input,
            assistant_output=assistant_output,
            tool_calls=ctx.tool_calls,
            instructions_hash=ctx.instructions_hash,
            model_deployment=ctx.model_deployment,
            correlation_id=ctx.correlation_id,
            session_id=ctx.session_id,
            request_latency_ms=latency_ms,
            token_usage=token_usage,
            metadata=ctx.metadata,
        )
        
        # Try to store in Cosmos
        stored = False
        try:
            result = self.ledger.store_episode(episode)
            if result:
                stored = True
                logger.info(f"Episode {episode.id} stored in Cosmos")
        except Exception as e:
            logger.warning(f"Failed to store episode in Cosmos: {e}")
        
        # Fallback to local file if Cosmos failed
        fallback_file = self._get_fallback_file()
        if not stored and fallback_file:
            try:
                with open(fallback_file, 'a') as f:
                    f.write(json.dumps(episode.to_dict()) + '\n')
                logger.info(f"Episode {episode.id} stored in local fallback")
            except Exception as e:
                logger.error(f"Failed to store episode in local fallback: {e}")
        
        return episode

    def capture_from_tool_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        user_input: str,
        model_deployment: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> Optional[Episode]:
        """
        Convenience method to capture a single tool call as an episode.
        
        This is useful for capturing MCP tool calls directly.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result
            user_input: The user's input
            model_deployment: Model deployment name
            duration_ms: Duration in milliseconds
        
        Returns:
            The stored Episode, or None if disabled/failed
        """
        if not self.is_enabled():
            return None
        
        ctx = self.start_capture(
            user_input=user_input,
            model_deployment=model_deployment,
        )
        
        self.record_tool_call(
            ctx=ctx,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            duration_ms=duration_ms,
        )
        
        return self.end_capture(ctx, assistant_output=result)


# Singleton instance
_capture_hook_instance: Optional[EpisodeCaptureHook] = None


def get_capture_hook() -> EpisodeCaptureHook:
    """Get the singleton capture hook instance."""
    global _capture_hook_instance
    if _capture_hook_instance is None:
        _capture_hook_instance = EpisodeCaptureHook.from_env()
    return _capture_hook_instance
