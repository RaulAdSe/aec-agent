"""Integration tests for agent workflows."""

from src.agents.compliance_agent import ComplianceAgent


def test_agent_end_to_end_without_api():
    """Test complete agent workflow without API key."""
    agent = ComplianceAgent()
    
    sample_data = {
        "metadata": {"project_name": "Integration Test Building"},
        "rooms": [
            {"id": "R001", "name": "Office", "area": 25.0},
            {"id": "R002", "name": "Corridor", "area": 10.0}
        ]
    }
    
    # Should handle gracefully even without API
    result = agent.process(sample_data)
    assert "status" in result
    
    # Check memory was updated
    memory_summary = agent.get_memory_summary()
    assert memory_summary["total_entries"] > 0


def test_agent_specific_compliance():
    """Test specific compliance analysis."""
    agent = ComplianceAgent()
    
    sample_data = {
        "metadata": {"project_name": "Fire Safety Test"},
        "doors": [{"id": "D001", "width": 0.9, "fire_rated": True}]
    }
    
    result = agent.analyze_specific_compliance(sample_data, "fire_safety")
    assert result["compliance_type"] == "fire_safety"