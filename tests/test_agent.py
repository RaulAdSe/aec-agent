"""
Basic tests for the clean AEC compliance agent.

Minimal test suite focused on core functionality.
"""

import pytest
from unittest.mock import Mock, patch
import json

from src.agent import ComplianceAgent, AgentConfig
from src.utils import ToonConverter


class TestComplianceAgent:
    """Test the core compliance agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized with default config."""
        config = AgentConfig(verbose=False)
        agent = ComplianceAgent(config)
        
        assert agent.config.model_name == "gemini-2.0-flash-exp"
        assert agent.config.use_toon == True
        assert agent.toon_converter is not None
        assert agent.memory is not None
        assert agent.toolkit is not None
    
    def test_agent_memory_operations(self):
        """Test agent memory functionality."""
        agent = ComplianceAgent(AgentConfig(verbose=False))
        
        # Test storing building data
        building_data = {
            "metadata": {"project_name": "Test Building"},
            "rooms": [{"id": "R001", "area": 25.0}]
        }
        
        agent.memory.add_building_data(building_data)
        stored_data = agent.memory.get_latest_building_data()
        
        assert stored_data == building_data
        assert stored_data["metadata"]["project_name"] == "Test Building"
    
    def test_memory_summary(self):
        """Test memory summary functionality."""
        agent = ComplianceAgent(AgentConfig(verbose=False))
        
        # Add some test data
        building_data = {"metadata": {"project_name": "Test", "total_area": 100.0}}
        agent.memory.add_building_data(building_data)
        
        summary = agent.get_memory_summary()
        
        assert "memory_stats" in summary
        assert summary["memory_stats"]["building_data_entries"] == 1
        assert summary["latest_building"]["project_name"] == "Test"
    
    @patch('src.agent.core.ChatGoogleGenerativeAI')
    @patch('src.agent.core.AgentExecutor')
    def test_agent_analysis_mock(self, mock_executor, mock_llm):
        """Test agent analysis with mocked LLM."""
        # Mock the agent executor response
        mock_executor.from_agent_and_tools.return_value.invoke.return_value = {
            "output": "Building is compliant with fire safety regulations."
        }
        
        agent = ComplianceAgent(AgentConfig(verbose=False))
        
        building_data = {
            "metadata": {"project_name": "Test Building"},
            "rooms": [{"id": "R001", "area": 25.0, "use": "office"}],
            "doors": [{"id": "D001", "width": 0.9}]
        }
        
        # This would normally fail without API key, but we're mocking it
        result = agent.analyze_building(building_data)
        
        # Verify the building data was stored in memory
        stored_data = agent.memory.get_latest_building_data()
        assert stored_data == building_data


class TestToonConverter:
    """Test the TOON format converter."""
    
    def test_toon_converter_initialization(self):
        """Test TOON converter can be initialized."""
        converter = ToonConverter()
        assert converter.config.use_toon == True
        assert converter.config.validate_conversion == True
    
    def test_token_savings_calculation(self):
        """Test token savings analysis."""
        converter = ToonConverter()
        
        test_data = {
            "rooms": [
                {"id": "R001", "area": 25.0, "type": "office"},
                {"id": "R002", "area": 30.0, "type": "office"}
            ]
        }
        
        try:
            savings = converter.get_token_savings(test_data)
            
            # Check that savings analysis returns expected fields
            assert "json_size" in savings
            assert "toon_size" in savings
            assert "savings_percent" in savings
            assert "compression_ratio" in savings
            
            # TOON should be smaller than JSON for this structured data
            assert savings["toon_size"] <= savings["json_size"]
            
        except ValueError as e:
            # If TOON library not available, that's expected
            assert "TOON library not available" in str(e)
    
    def test_normalize_input(self):
        """Test input normalization."""
        converter = ToonConverter()
        
        # Test dict input
        test_dict = {"test": "data"}
        result = converter._normalize_input(test_dict)
        assert result == test_dict
        
        # Test JSON string input
        json_string = '{"test": "data"}'
        result = converter._normalize_input(json_string)
        assert result == {"test": "data"}


class TestComplianceTools:
    """Test the compliance toolkit."""
    
    def test_toolkit_initialization(self):
        """Test toolkit can be initialized."""
        from src.agent.tools import ComplianceToolkit
        
        toolkit = ComplianceToolkit()
        tools = toolkit.get_tools()
        
        assert len(tools) > 0
        
        # Check expected tools are present
        tool_names = [tool.name for tool in tools]
        assert "calculate_clearance" in tool_names
        assert "analyze_door_width" in tool_names
        assert "check_evacuation_routes" in tool_names
    
    def test_door_analysis_tool(self):
        """Test door width analysis tool."""
        from src.agent.tools import ComplianceToolkit
        
        toolkit = ComplianceToolkit()
        
        # Test door data
        door_data = {
            "doors": [
                {"id": "D001", "width": 0.9},  # Compliant
                {"id": "D002", "width": 0.7},  # Non-compliant
            ]
        }
        
        result = toolkit._analyze_door_width_tool(str(door_data))
        
        assert "D001" in result
        assert "D002" in result
        assert "COMPLIANT" in result
        assert "NON-COMPLIANT" in result
    
    def test_occupancy_calculation_tool(self):
        """Test occupancy load calculation."""
        from src.agent.tools import ComplianceToolkit
        
        toolkit = ComplianceToolkit()
        
        room_data = {
            "rooms": [
                {"id": "R001", "area": 50.0, "use": "office", "occupancy_load": 3},
                {"id": "R002", "area": 100.0, "use": "classroom"}
            ]
        }
        
        result = toolkit._calculate_occupancy_load_tool(str(room_data))
        
        assert "R001" in result
        assert "R002" in result
        assert "Total building occupancy" in result


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])