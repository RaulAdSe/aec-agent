"""Performance tests for reasoning agent processes."""

import time
from unittest.mock import patch
from aec_agent import ReasoningAgent


def test_agent_initialization_speed():
    """Test agent initializes within reasonable time."""
    start_time = time.time()
    agent = ReasoningAgent(enable_memory=False)
    initialization_time = time.time() - start_time
    
    # Should initialize in under 2 seconds (more components now)
    assert initialization_time < 2.0
    assert agent is not None


def test_memory_performance():
    """Test memory operations performance."""
    agent = ReasoningAgent(enable_memory=True)
    
    # Add multiple session goals
    start_time = time.time()
    for i in range(10):
        agent.set_session_goal(f"Test goal {i}")
    
    memory_time = time.time() - start_time
    
    # Should handle multiple operations quickly
    assert memory_time < 0.5
    summary = agent.get_session_summary()
    assert summary is not None


def test_reasoning_component_initialization():
    """Test reasoning component initialization performance."""
    start_time = time.time()
    agent = ReasoningAgent(enable_memory=False)
    
    # Check all reasoning components are initialized
    assert agent.goal_decomposer is not None
    assert agent.tool_planner is not None
    assert agent.executor is not None
    assert agent.validator is not None
    assert agent.reasoning_controller is not None
    
    init_time = time.time() - start_time
    assert init_time < 2.0  # Should initialize quickly