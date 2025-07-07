"""
Test framework for Organic Intelligence components
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import redis

from core.dispatcher.action_dispatcher import ActionDispatcher
from core.agents.scheduling_agent import SchedulingAgent
from core.agents.london_killzone_agent import LondonKillzoneAgent
from core.command_processor import CommandProcessor, OrganicIntelligenceOrchestrator


class TestActionDispatcher:
    """Test the Action Dispatcher component."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return Mock(spec=redis.Redis)

    @pytest.fixture
    def dispatcher(self, mock_redis):
        """Create a dispatcher instance."""
        return ActionDispatcher(mock_redis, {})

    @pytest.mark.asyncio
    async def test_process_valid_command(self, dispatcher):
        """Test processing a valid command."""
        command = {
            "request_id": "test_123",
            "action_type": "LOG_JOURNAL_ENTRY",
            "payload": {
                "type": "test",
                "content": "Test entry"
            }
        }

        result = await dispatcher.process_command(command)

        assert result["status"] == "success"
        assert "entry_id" in result

    @pytest.mark.asyncio
    async def test_invalid_command_structure(self, dispatcher):
        """Test handling of invalid command structure."""
        command = {
            "action_type": "LOG_JOURNAL_ENTRY"
            # Missing required fields
        }

        result = await dispatcher.process_command(command)

        assert result["status"] == "error"
        assert "Invalid command structure" in result["message"]

    @pytest.mark.asyncio
    async def test_unknown_action_type(self, dispatcher):
        """Test handling of unknown action type."""
        command = {
            "request_id": "test_123",
            "action_type": "UNKNOWN_ACTION",
            "payload": {}
        }

        result = await dispatcher.process_command(command)

        assert result["status"] == "error"
        assert "Unknown action type" in result["message"]


class TestLondonKillzoneAgent:
    """Test the London Killzone specialist agent."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = Mock(spec=redis.Redis)
        mock.get.return_value = None
        mock.setex.return_value = True
        return mock

    @pytest.fixture
    def manifest(self):
        """Create a test manifest."""
        return {
            "strategy_id": "test_london",
            "risk_params": {
                "allowed_pairs": ["EURUSD"],
                "max_risk_per_trade": 0.01
            },
            "workflow": [
                {
                    "name": "identify_asian_range",
                    "params": {"lookback_hours": 8}
                }
            ]
        }

    @pytest.fixture
    def agent(self, mock_redis, manifest):
        """Create an agent instance."""
        return LondonKillzoneAgent(mock_redis, manifest)

    @pytest.mark.asyncio
    async def test_identify_asian_range(self, agent):
        """Test Asian range identification."""
        # Mock price data retrieval
        with patch.object(agent, '_get_price_data') as mock_get_data:
            import pandas as pd
            mock_data = pd.DataFrame({
                'high': [1.0860, 1.0865, 1.0870],
                'low': [1.0850, 1.0855, 1.0858],
                'close': [1.0855, 1.0862, 1.0865]
            })
            mock_get_data.return_value = mock_data

            step = {"name": "identify_asian_range", "params": {}}
            result = await agent._identify_asian_range("EURUSD", step)

            assert result is True
            assert "asia_high" in agent.workflow_state
            assert "asia_low" in agent.workflow_state
            assert agent.workflow_state["asia_high"] == 1.0870
            assert agent.workflow_state["asia_low"] == 1.0850


class TestSchedulingAgent:
    """Test the Scheduling Agent."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return Mock(spec=redis.Redis)

    @pytest.fixture
    def agent(self, mock_redis, tmp_path):
        """Create a scheduling agent."""
        # Create a temporary manifest directory
        manifest_dir = tmp_path / "strategies"
        manifest_dir.mkdir()

        # Create a test manifest
        manifest = {
            "strategy_id": "test_strategy",
            "enabled": True,
            "schedule": {
                "trigger_type": "time_based",
                "utc_time_window": {
                    "start": "06:00",
                    "end": "09:00"
                },
                "frequency": "every_1_minute"
            }
        }

        import yaml
        with open(manifest_dir / "test_strategy.yml", "w") as f:
            yaml.dump(manifest, f)

        return SchedulingAgent(mock_redis, str(manifest_dir))

    @pytest.mark.asyncio
    async def test_load_manifests(self, agent):
        """Test loading strategy manifests."""
        await agent.initialize()

        assert "test_strategy" in agent.active_strategies
        assert agent.active_strategies["test_strategy"]["enabled"] is True


class TestCommandProcessor:
    """Test the Command Processor."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = Mock(spec=redis.Redis)
        mock.brpop.return_value = None
        mock.rpop.return_value = None
        return mock

    @pytest.fixture
    def processor(self, mock_redis):
        """Create a command processor."""
        return CommandProcessor(mock_redis, {})

    @pytest.mark.asyncio
    async def test_process_command_from_queue(self, processor, mock_redis):
        """Test processing commands from queue."""
        # Mock a command in queue
        command = {
            "request_id": "test_123",
            "action_type": "LOG_JOURNAL_ENTRY",
            "payload": {"content": "Test"}
        }

        mock_redis.brpop.return_value = ("queue", json.dumps(command))

        # Start processor briefly
        await processor.start()
        await asyncio.sleep(0.1)
        await processor.stop()

        # Verify command was attempted to be processed
        mock_redis.brpop.assert_called()


class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_end_to_end_command_flow(self):
        """Test complete command flow from trigger to execution."""
        # This would be a more complex integration test
        # involving multiple components working together
        pass


# Pytest configuration
pytest_plugins = ["pytest_asyncio"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
