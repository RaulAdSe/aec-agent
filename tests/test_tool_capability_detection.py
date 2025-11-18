"""
Tests for Fix 4: Tool Capability Detection

This module tests that the tool planner can detect unsupported capabilities
and select appropriate alternatives.
"""

import pytest
from unittest.mock import Mock

from aec_agent.core.tool_planner import ToolPlanner
from aec_agent.core.reasoning_utils import Task, Priority
from aec_agent.config import AgentConfig


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    return llm


@pytest.fixture
def planner(mock_llm):
    """Create a ToolPlanner with mocked LLM."""
    return ToolPlanner(llm=mock_llm)


def test_spatial_relationship_blocked(planner):
    """Test that unsupported spatial relationships are detected."""
    planned_input = {"relationship_type": "spatial"}
    capable = planner._check_tool_capability("find_related", planned_input)
    
    assert capable == False


def test_supported_relationship_allowed(planner):
    """Test that supported relationships are allowed."""
    planned_input = {"relationship_type": "adjacent"}
    capable = planner._check_tool_capability("find_related", planned_input)
    
    assert capable == True


def test_unsupported_compliance_rule_blocked(planner):
    """Test that unsupported compliance rules are detected."""
    planned_input = {"rule_type": "stair_compliance"}
    capable = planner._check_tool_capability("validate_rule", planned_input)
    
    assert capable == False


def test_supported_compliance_rule_allowed(planner):
    """Test that supported compliance rules are allowed."""
    planned_input = {"rule_type": "fire_safety"}
    capable = planner._check_tool_capability("validate_rule", planned_input)
    
    assert capable == True


def test_unknown_tool_capability_assumed_capable(planner):
    """Test that unknown tools are assumed capable."""
    planned_input = {"some_param": "value"}
    capable = planner._check_tool_capability("unknown_tool", planned_input)
    
    assert capable == True


def test_tool_without_limitations_assumed_capable(planner):
    """Test that tools without known limitations are assumed capable."""
    planned_input = {"element_type": "doors"}
    capable = planner._check_tool_capability("get_all_elements", planned_input)
    
    assert capable == True


def test_empty_input_is_capable(planner):
    """Test that empty input is considered capable."""
    planned_input = {}
    capable = planner._check_tool_capability("find_related", planned_input)
    
    assert capable == True


def test_capability_safe_alternative_find_related(planner):
    """Test capability-safe alternative for find_related."""
    task = Task(
        id="test",
        name="Find spatial relationships",
        description="Find spatial relationships for doors"
    )
    
    alternative = planner._get_capability_safe_alternative("find_related", task)
    assert alternative == "query_elements"


def test_capability_safe_alternative_validate_rule(planner):
    """Test capability-safe alternative for validate_rule."""
    task = Task(
        id="test",
        name="Validate compliance",
        description="Validate stair compliance rules"
    )
    
    alternative = planner._get_capability_safe_alternative("validate_rule", task)
    assert alternative == "search_compliance_documents"


def test_capability_safe_alternative_unknown_tool(planner):
    """Test capability-safe alternative for unknown tools."""
    task = Task(
        id="test",
        name="Unknown task",
        description="Unknown task description"
    )
    
    alternative = planner._get_capability_safe_alternative("unknown_tool", task)
    assert alternative == "query_elements"


def test_select_primary_tool_load(planner):
    """Test primary tool selection for load operations."""
    task = Task(
        id="test",
        name="Load building data",
        description="Load and initialize building data from file"
    )
    
    primary_tool = planner._select_primary_tool(task)
    assert primary_tool == "load_building_data"


def test_select_primary_tool_search(planner):
    """Test primary tool selection for search operations."""
    task = Task(
        id="test",
        name="Search compliance documents",
        description="Search compliance documents for regulations"
    )
    
    primary_tool = planner._select_primary_tool(task)
    assert primary_tool == "search_compliance_documents"


def test_select_primary_tool_validate(planner):
    """Test primary tool selection for validation operations."""
    task = Task(
        id="test",
        name="Validate compliance",
        description="Validate building elements against compliance rules"
    )
    
    primary_tool = planner._select_primary_tool(task)
    assert primary_tool == "validate_rule"


def test_select_primary_tool_relationships(planner):
    """Test primary tool selection for relationship operations."""
    task = Task(
        id="test",
        name="Find related elements",
        description="Find spatial relationships between elements"
    )
    
    primary_tool = planner._select_primary_tool(task)
    assert primary_tool == "find_related"


def test_select_primary_tool_default(planner):
    """Test primary tool selection defaults to query_elements."""
    task = Task(
        id="test",
        name="Generic task",
        description="Generic task with no specific keywords"
    )
    
    primary_tool = planner._select_primary_tool(task)
    assert primary_tool == "query_elements"


def test_prepare_tool_input_spatial_relationship(planner):
    """Test input preparation for spatial relationships."""
    task = Task(
        id="test",
        name="Find relationships",
        description="Find spatial relationships between doors"
    )
    
    planned_input = planner._prepare_tool_input("find_related", task, {})
    assert planned_input["relationship_type"] == "spatial"


def test_prepare_tool_input_connected_relationship(planner):
    """Test input preparation for connected relationships."""
    task = Task(
        id="test",
        name="Find connections",
        description="Find connected elements in the building"
    )
    
    planned_input = planner._prepare_tool_input("find_related", task, {})
    assert planned_input["relationship_type"] == "connected"


def test_prepare_tool_input_stair_compliance(planner):
    """Test input preparation for stair compliance rules."""
    task = Task(
        id="test",
        name="Validate stairs",
        description="Validate stair_compliance rules for building"
    )
    
    planned_input = planner._prepare_tool_input("validate_rule", task, {})
    assert planned_input["rule_type"] == "stair_compliance"


def test_prepare_tool_input_door_compliance(planner):
    """Test input preparation for door compliance rules."""
    task = Task(
        id="test",
        name="Validate doors", 
        description="Check door compliance requirements"
    )
    
    planned_input = planner._prepare_tool_input("validate_rule", task, {})
    assert planned_input["rule_type"] == "door_compliance"


def test_prepare_tool_input_no_keywords(planner):
    """Test input preparation when no keywords are found."""
    task = Task(
        id="test",
        name="Generic task",
        description="Generic task description"
    )
    
    planned_input = planner._prepare_tool_input("find_related", task, {})
    assert planned_input == {}


def test_select_tool_with_capability_check_capable(planner):
    """Test tool selection when primary tool is capable."""
    task = Task(
        id="test",
        name="Find connections",
        description="Find connected elements in building"
    )
    
    selected_tool = planner.select_tool_with_capability_check(task, {})
    assert selected_tool == "find_related"


def test_select_tool_with_capability_check_incapable(planner):
    """Test tool selection when primary tool is not capable."""
    task = Task(
        id="test", 
        name="Find relationships",
        description="Find spatial relationships between elements"
    )
    
    selected_tool = planner.select_tool_with_capability_check(task, {})
    # Should fall back to safe alternative since spatial is unsupported
    assert selected_tool == "query_elements"


def test_select_tool_with_capability_check_stair_compliance(planner):
    """Test tool selection for unsupported stair compliance."""
    task = Task(
        id="test",
        name="Validate stairs",
        description="Validate stair_compliance rules for stairs"
    )
    
    selected_tool = planner.select_tool_with_capability_check(task, {})
    # Should fall back to search since stair_compliance is unsupported
    assert selected_tool == "search_compliance_documents"


def test_known_limitations_structure(planner):
    """Test that known_limitations has expected structure."""
    limitations = planner.known_limitations
    
    assert isinstance(limitations, dict)
    assert "find_related" in limitations
    assert "validate_rule" in limitations
    
    find_related_limits = limitations["find_related"]
    assert "unsupported_relationships" in find_related_limits
    assert "supported_relationships" in find_related_limits
    assert isinstance(find_related_limits["unsupported_relationships"], list)
    assert "spatial" in find_related_limits["unsupported_relationships"]
    
    validate_rule_limits = limitations["validate_rule"] 
    assert "unsupported_rule_types" in validate_rule_limits
    assert "supported_rule_types" in validate_rule_limits
    assert isinstance(validate_rule_limits["unsupported_rule_types"], list)
    assert "stair_compliance" in validate_rule_limits["unsupported_rule_types"]


def test_case_insensitive_keyword_detection(planner):
    """Test that keyword detection works with mixed case."""
    task = Task(
        id="test",
        name="FIND SPATIAL RELATIONSHIPS",
        description="FIND SPATIAL RELATIONSHIPS FOR DOORS"
    )
    
    planned_input = planner._prepare_tool_input("find_related", task, {})
    assert planned_input["relationship_type"] == "spatial"