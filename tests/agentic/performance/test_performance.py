"""Performance tests for agentic processes."""

import time
from unittest.mock import patch
from src.agents.compliance_agent import ComplianceAgent


def test_agent_initialization_speed():
    """Test agent initializes within reasonable time."""
    start_time = time.time()
    agent = ComplianceAgent()
    initialization_time = time.time() - start_time
    
    # Should initialize in under 1 second
    assert initialization_time < 1.0
    assert agent is not None


def test_memory_performance():
    """Test memory operations performance."""
    agent = ComplianceAgent()
    
    # Add multiple entries
    start_time = time.time()
    for i in range(10):
        sample_data = {"metadata": {"project_name": f"Test {i}"}}
        agent.memory.add_building_data(sample_data)
    
    memory_time = time.time() - start_time
    
    # Should handle 10 entries quickly
    assert memory_time < 0.1
    assert agent.get_memory_summary()["total_entries"] == 10


def test_mock_ai_response_time():
    """Test simulated AI response performance."""
    with patch('src.services.ai_client.AIClient.analyze_compliance') as mock_ai:
        # Simulate fast response
        mock_ai.return_value = {"status": "success", "analysis": "Fast test"}
        
        agent = ComplianceAgent()
        
        start_time = time.time()
        result = agent.process({"test": "data"})
        process_time = time.time() - start_time
        
        # Should process quickly when AI is mocked
        assert process_time < 0.5
        assert result["status"] == "success"