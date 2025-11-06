"""Basic tests for the AEC compliance agent."""

from src.core.config import AppConfig
from src.agents.compliance_agent import ComplianceAgent


def test_config_loads():
    """Test configuration loads correctly."""
    config = AppConfig()
    assert config.default_model == "gpt-4"


def test_agent_initializes():
    """Test agent can be created."""
    agent = ComplianceAgent()
    assert agent is not None


def test_agent_processes_data():
    """Test agent can process sample data (without API key)."""
    agent = ComplianceAgent()
    
    sample_data = {
        "metadata": {"project_name": "Test"},
        "rooms": [{"id": "R001", "area": 25.0}]
    }
    
    result = agent.process(sample_data)
    # Should handle gracefully even without API key
    assert "status" in result