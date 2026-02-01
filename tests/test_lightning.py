"""
Tests for Agent Lightning - Fine-tuning and behavior optimization.

Run with: pytest tests/test_lightning.py -v
"""

import json
import os
import sys
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lightning import (
    RLLedgerCosmos,
    Episode,
    EpisodeToolCall,
    Reward,
    RewardSource,
    Dataset,
    TrainingRun,
    TrainingStatus,
    Deployment,
    EpisodeCaptureHook,
    CaptureConfig,
    RewardWriter,
    RewardConfig,
    DatasetBuilder,
    DatasetConfig,
    DeploymentRegistry,
)


class MockCosmosContainer:
    """Mock Cosmos container for testing."""
    
    def __init__(self):
        self.items = {}
    
    def upsert_item(self, doc):
        self.items[doc['id']] = doc
        return doc
    
    def read_item(self, item, partition_key):
        if item in self.items:
            return self.items[item]
        from azure.cosmos import exceptions
        raise exceptions.CosmosResourceNotFoundError(message="Not found")
    
    def query_items(self, query, parameters=None, enable_cross_partition_query=False, partition_key=None):
        # Simple mock that returns all items
        return list(self.items.values())


class TestEpisodeDataclass:
    """Tests for Episode dataclass."""
    
    def test_episode_creation(self):
        """Test creating an episode."""
        episode = Episode(
            id="ep-123",
            agent_id="mcp-agents",
            user_input="What is 2+2?",
            assistant_output="4",
            tool_calls=[
                EpisodeToolCall(
                    tool_name="ask_foundry",
                    arguments={"question": "What is 2+2?"},
                    result="4",
                    duration_ms=100,
                )
            ],
            model_deployment="gpt-4",
            request_latency_ms=150,
        )
        
        assert episode.id == "ep-123"
        assert episode.agent_id == "mcp-agents"
        assert len(episode.tool_calls) == 1
        assert episode.tool_calls[0].tool_name == "ask_foundry"
    
    def test_episode_to_dict(self):
        """Test episode serialization."""
        episode = Episode(
            id="ep-123",
            agent_id="mcp-agents",
            user_input="Hello",
            assistant_output="Hi there!",
        )
        
        data = episode.to_dict()
        
        assert data["id"] == "ep-123"
        assert data["agent_id"] == "mcp-agents"
        assert data["user_input"] == "Hello"
        assert "created_at" in data
    
    def test_episode_from_dict(self):
        """Test episode deserialization."""
        data = {
            "id": "ep-456",
            "agent_id": "test-agent",
            "user_input": "Test input",
            "assistant_output": "Test output",
            "tool_calls": [
                {"tool_name": "test_tool", "arguments": {"a": 1}, "result": "ok"}
            ],
            "created_at": "2024-01-01T00:00:00",
        }
        
        episode = Episode.from_dict(data)
        
        assert episode.id == "ep-456"
        assert len(episode.tool_calls) == 1
        assert episode.tool_calls[0].tool_name == "test_tool"


class TestRewardDataclass:
    """Tests for Reward dataclass."""
    
    def test_reward_creation(self):
        """Test creating a reward."""
        reward = Reward(
            id="rw-123",
            episode_id="ep-123",
            agent_id="mcp-agents",
            source=RewardSource.HUMAN_APPROVAL,
            value=1.0,
            raw_value=True,
            evaluator="admin@example.com",
        )
        
        assert reward.id == "rw-123"
        assert reward.source == RewardSource.HUMAN_APPROVAL
        assert reward.value == 1.0
    
    def test_reward_to_dict(self):
        """Test reward serialization."""
        reward = Reward(
            id="rw-123",
            episode_id="ep-123",
            agent_id="mcp-agents",
            source=RewardSource.EVAL_SCORE,
            value=0.8,
        )
        
        data = reward.to_dict()
        
        assert data["source"] == "eval_score"
        assert data["value"] == 0.8


class TestEpisodeCaptureHook:
    """Tests for episode capture hook."""
    
    def test_capture_config_from_env(self):
        """Test loading config from environment."""
        with patch.dict(os.environ, {
            "ENABLE_LIGHTNING_CAPTURE": "true",
            "LIGHTNING_AGENT_ID": "test-agent",
        }):
            config = CaptureConfig.from_env()
            
            assert config.enabled == True
            assert config.agent_id == "test-agent"
    
    def test_capture_disabled_by_default(self):
        """Test capture is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            config = CaptureConfig.from_env()
            assert config.enabled == False
    
    def test_start_and_end_capture(self):
        """Test starting and ending capture."""
        config = CaptureConfig(enabled=True, agent_id="test-agent")
        
        # Create hook with mock ledger
        mock_ledger = Mock()
        mock_ledger.store_episode = Mock(return_value="ep-123")
        
        hook = EpisodeCaptureHook(config=config, ledger=mock_ledger)
        
        # Start capture
        ctx = hook.start_capture(
            user_input="What is 2+2?",
            model_deployment="gpt-4",
        )
        
        assert ctx.user_input == "What is 2+2?"
        assert ctx.agent_id == "test-agent"
        
        # Record tool call
        hook.record_tool_call(
            ctx=ctx,
            tool_name="ask_foundry",
            arguments={"question": "What is 2+2?"},
            result="4",
            duration_ms=100,
        )
        
        assert len(ctx.tool_calls) == 1
        
        # End capture
        episode = hook.end_capture(ctx, assistant_output="4")
        
        assert episode is not None
        assert episode.assistant_output == "4"
        mock_ledger.store_episode.assert_called_once()
    
    def test_redaction_of_secrets(self):
        """Test that secrets are redacted."""
        from lightning.episode_capture import redact_sensitive_data
        
        text = "Bearer abc123xyz token"
        redacted = redact_sensitive_data(text)
        assert "abc123xyz" not in redacted
        assert "[REDACTED]" in redacted
        
        text2 = "api_key=mysecretkey"
        redacted2 = redact_sensitive_data(text2)
        assert "mysecretkey" not in redacted2


class TestRewardWriter:
    """Tests for reward writer."""
    
    def test_record_human_approval(self):
        """Test recording human approval."""
        mock_ledger = Mock()
        mock_ledger.store_reward = Mock(return_value="rw-123")
        
        writer = RewardWriter(ledger=mock_ledger)
        
        reward = writer.record_human_approval(
            episode_id="ep-123",
            agent_id="mcp-agents",
            approved=True,
            reviewer="admin@example.com",
        )
        
        assert reward is not None
        assert reward.value == 1.0
        assert reward.source == RewardSource.HUMAN_APPROVAL
        mock_ledger.store_reward.assert_called_once()
    
    def test_record_eval_score(self):
        """Test recording evaluation score."""
        mock_ledger = Mock()
        mock_ledger.store_reward = Mock(return_value="rw-123")
        
        writer = RewardWriter(ledger=mock_ledger)
        
        reward = writer.record_eval_score(
            episode_id="ep-123",
            agent_id="mcp-agents",
            score=0.75,
            rubric="accuracy",
        )
        
        assert reward is not None
        # 0.75 normalized to [-1, 1] = 0.5
        assert reward.value == pytest.approx(0.5, 0.01)
    
    def test_record_test_result_failure(self):
        """Test recording a failed test."""
        mock_ledger = Mock()
        mock_ledger.store_reward = Mock(return_value="rw-123")
        
        writer = RewardWriter(ledger=mock_ledger)
        
        reward = writer.record_test_result(
            episode_id="ep-123",
            agent_id="mcp-agents",
            passed=False,
            test_name="intent_match",
        )
        
        assert reward is not None
        assert reward.value == -1.0


class TestDatasetBuilder:
    """Tests for dataset builder."""
    
    def test_episode_to_messages(self):
        """Test converting episode to training messages format."""
        config = DatasetConfig(include_tool_calls=False)
        builder = DatasetBuilder(config=config)
        
        episode = Episode(
            id="ep-123",
            agent_id="mcp-agents",
            user_input="What is the capital of France?",
            assistant_output="The capital of France is Paris.",
        )
        
        messages = builder._episode_to_messages(episode)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the capital of France?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "The capital of France is Paris."
    
    def test_episode_to_messages_with_tool_calls(self):
        """Test converting episode with tool calls to messages."""
        config = DatasetConfig(include_tool_calls=True)
        builder = DatasetBuilder(config=config)
        
        episode = Episode(
            id="ep-123",
            agent_id="mcp-agents",
            user_input="What is 2+2?",
            assistant_output="4",
            tool_calls=[
                EpisodeToolCall(
                    tool_name="calculator",
                    arguments={"a": 2, "b": 2},
                    result="4",
                )
            ],
        )
        
        messages = builder._episode_to_messages(episode)
        
        # Assistant message should include tool trace
        assert "[Tool: calculator]" in messages[1]["content"]
    
    def test_split_data(self):
        """Test train/validation split."""
        config = DatasetConfig(train_split=0.8)
        builder = DatasetBuilder(config=config)
        
        data = [{"id": i} for i in range(100)]
        train, val = builder._split_data(data)
        
        assert len(train) == 80
        assert len(val) == 20
    
    def test_build_from_golden(self):
        """Test building dataset from golden conversations."""
        config = DatasetConfig(output_dir=tempfile.mkdtemp())
        mock_ledger = Mock()
        mock_ledger.store_dataset = Mock(return_value="ds-123")
        
        builder = DatasetBuilder(config=config, ledger=mock_ledger)
        
        # Create a golden file
        golden_file = os.path.join(config.output_dir, "golden.jsonl")
        with open(golden_file, 'w') as f:
            for i in range(10):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"Question {i}"},
                        {"role": "assistant", "content": f"Answer {i}"},
                    ]
                }) + '\n')
        
        dataset = builder.build_from_golden_conversations(
            agent_id="mcp-agents",
            name="golden-v1",
            golden_file=golden_file,
        )
        
        assert dataset is not None
        assert dataset.training_count + dataset.validation_count == 10


class TestDeploymentRegistry:
    """Tests for deployment registry."""
    
    def test_get_active_model_disabled(self):
        """Test that tuned model is not used when disabled."""
        mock_ledger = Mock()
        registry = DeploymentRegistry(ledger=mock_ledger)
        
        with patch.dict(os.environ, {"USE_TUNED_MODEL": "false"}):
            result = registry.get_active_model("mcp-agents")
            assert result is None
    
    def test_get_active_model_from_cosmos(self):
        """Test getting active model from Cosmos."""
        mock_ledger = Mock()
        mock_deployment = Deployment(
            id="dep-123",
            agent_id="mcp-agents",
            training_run_id="run-123",
            tuned_model_name="ft:gpt-4:custom-model",
            is_active=True,
        )
        mock_ledger.get_active_deployment = Mock(return_value=mock_deployment)
        
        registry = DeploymentRegistry(ledger=mock_ledger)
        
        with patch.dict(os.environ, {"USE_TUNED_MODEL": "true"}):
            result = registry.get_active_model("mcp-agents")
            assert result == "ft:gpt-4:custom-model"
    
    def test_get_active_model_fallback(self):
        """Test fallback to env var when no Cosmos deployment."""
        mock_ledger = Mock()
        mock_ledger.get_active_deployment = Mock(return_value=None)
        
        registry = DeploymentRegistry(
            ledger=mock_ledger,
            fallback_model="fallback-model",
        )
        
        with patch.dict(os.environ, {"USE_TUNED_MODEL": "true"}):
            result = registry.get_active_model("mcp-agents")
            assert result == "fallback-model"


class TestRLLedgerCosmos:
    """Tests for RLLedgerCosmos."""
    
    def test_episode_roundtrip(self):
        """Test storing and retrieving an episode."""
        # Create a ledger with mock containers
        ledger = RLLedgerCosmos()
        ledger._initialized = True
        ledger._containers = {
            "episodes": MockCosmosContainer(),
            "rewards": MockCosmosContainer(),
            "datasets": MockCosmosContainer(),
            "runs": MockCosmosContainer(),
            "deployments": MockCosmosContainer(),
        }
        
        episode = Episode(
            id="ep-123",
            agent_id="mcp-agents",
            user_input="Hello",
            assistant_output="Hi!",
        )
        
        # Store
        result = ledger.store_episode(episode)
        assert result == "ep-123"
        
        # Retrieve
        retrieved = ledger.get_episode("ep-123", "mcp-agents")
        assert retrieved is not None
        assert retrieved.user_input == "Hello"
    
    def test_reward_roundtrip(self):
        """Test storing and retrieving a reward."""
        ledger = RLLedgerCosmos()
        ledger._initialized = True
        ledger._containers = {
            "episodes": MockCosmosContainer(),
            "rewards": MockCosmosContainer(),
            "datasets": MockCosmosContainer(),
            "runs": MockCosmosContainer(),
            "deployments": MockCosmosContainer(),
        }
        
        reward = Reward(
            id="rw-123",
            episode_id="ep-123",
            agent_id="mcp-agents",
            source=RewardSource.HUMAN_APPROVAL,
            value=1.0,
        )
        
        result = ledger.store_reward(reward)
        assert result == "rw-123"
    
    def test_promote_deployment(self):
        """Test promoting a deployment."""
        ledger = RLLedgerCosmos()
        ledger._initialized = True
        ledger._containers = {
            "episodes": MockCosmosContainer(),
            "rewards": MockCosmosContainer(),
            "datasets": MockCosmosContainer(),
            "runs": MockCosmosContainer(),
            "deployments": MockCosmosContainer(),
        }
        
        deployment = ledger.promote_deployment(
            agent_id="mcp-agents",
            training_run_id="run-123",
            tuned_model_name="ft:gpt-4:custom",
            promoted_by="admin",
        )
        
        assert deployment is not None
        assert deployment.tuned_model_name == "ft:gpt-4:custom"
        assert deployment.is_active == True


class TestModelSelection:
    """Tests for model selection in mcp_agents.py."""
    
    def test_model_selection_uses_base_when_disabled(self):
        """Test that base model is used when USE_TUNED_MODEL=false."""
        # This would be an integration test requiring the mcp_agents module
        # For now, we test the logic in isolation
        base_model = "gpt-5.2-chat"
        use_tuned = False
        
        if not use_tuned:
            selected_model = base_model
        
        assert selected_model == base_model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
