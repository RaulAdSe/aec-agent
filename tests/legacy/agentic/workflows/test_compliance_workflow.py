"""Test reasoning agent workflow decision-making."""

from unittest.mock import patch, MagicMock
from aec_agent import ReasoningAgent


def test_agent_tool_registry():
    """Test agent has initialized tool registry."""
    agent = ReasoningAgent(enable_memory=False)
    
    # Agent should have initialized tools
    assert len(agent.tool_registry) > 0
    
    # Should have key tools
    tool_names = list(agent.tool_registry.keys())
    assert "load_building_data" in tool_names
    assert "query_elements" in tool_names
    assert "search_compliance_documents" in tool_names


def test_agent_memory_workflow():
    """Test agent memory management during workflow."""
    agent = ReasoningAgent(enable_memory=True)
    
    # Initial state
    initial_summary = agent.get_session_summary()
    assert initial_summary is not None
    
    # Set session goal
    agent.set_session_goal("Test workflow goal")
    
    # Memory should be updated
    memory = agent.get_session_summary()
    assert memory["goal"]["description"] == "Test workflow goal"
    
    # Clear memory
    agent.clear_memory()
    updated_summary = agent.get_session_summary()
    assert updated_summary["memory_stats"]["total_turns"] == 0


def test_agent_reasoning_flow():
    """Test agent reasoning process."""
    # Mock the LLM responses for controlled testing
    with patch('aec_agent.core.goal_decomposer.GoalDecomposer.decompose') as mock_decompose:
        mock_decompose.return_value = {
            "success": True,
            "tasks": [],
            "method": "test"
        }
        
        agent = ReasoningAgent(enable_memory=False)
        result = agent.process_goal("Test goal")
        
        # Should have processed the goal
        assert "status" in result
        assert result["goal"] == "Test goal"
        mock_decompose.assert_called_once()