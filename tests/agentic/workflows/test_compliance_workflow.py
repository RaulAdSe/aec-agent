"""Test agent workflow decision-making."""

from unittest.mock import patch
from src.agents.compliance_agent import ComplianceAgent


def test_agent_tool_selection():
    """Test agent selects appropriate tools."""
    agent = ComplianceAgent()
    
    # Agent should have initialized tools
    tools = agent.toolkit.get_tools()
    assert len(tools) > 0
    
    # Should have data conversion tool
    tool_names = [tool["name"] for tool in tools]
    assert "convert_data_format" in tool_names


def test_agent_memory_workflow():
    """Test agent memory management during workflow."""
    agent = ComplianceAgent()
    
    # Initial state
    assert agent.get_memory_summary()["total_entries"] == 0
    
    # Process data
    sample_data = {"metadata": {"project_name": "Memory Test"}}
    agent.process(sample_data)
    
    # Memory should be updated
    memory = agent.get_memory_summary()
    assert memory["total_entries"] > 0
    assert memory["latest_building"]["project_name"] == "Memory Test"
    
    # Clear memory
    agent.clear_memory()
    assert agent.get_memory_summary()["total_entries"] == 0


def test_agent_decision_flow():
    """Test agent decision-making process."""
    with patch('src.services.ai_client.AIClient.analyze_compliance') as mock_ai:
        mock_ai.return_value = {
            "status": "success", 
            "analysis": "Test analysis result"
        }
        
        agent = ComplianceAgent()
        result = agent.process({"test": "data"})
        
        # Should have made decisions and processed
        assert "status" in result
        mock_ai.assert_called_once()