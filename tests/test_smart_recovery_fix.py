"""
Tests for Fix 3: Smart Recovery Alternatives

This module tests that the recovery system suggests appropriate alternatives
for specific failures and uses contextual information for better recovery.
"""

import pytest
from unittest.mock import Mock

from aec_agent.core.simple_recovery import SimpleRecovery, SMART_TOOL_ALTERNATIVES
from aec_agent.core.reasoning_utils import Task, ExecutionResult, Priority


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.invoke.return_value = Mock(content="try_different_tool")
    return llm


@pytest.fixture
def recovery(mock_llm):
    """Create a SimpleRecovery instance with mocked LLM."""
    return SimpleRecovery(llm=mock_llm)


def test_spatial_relationship_recovery(recovery):
    """Test recovery suggests appropriate alternative for spatial failures."""
    alternative = recovery.get_contextual_alternative(
        failed_tool="find_related",
        task_description="Find spatial relationships for stairs", 
        error_message="spatial relationship not implemented"
    )
    
    assert alternative == "calculate_distances"


def test_compliance_rule_recovery(recovery):
    """Test recovery for unsupported compliance rule types."""
    alternative = recovery.get_contextual_alternative(
        failed_tool="validate_rule",
        task_description="Validate stairs against compliance rules",
        error_message="stair_compliance rule type not implemented"  
    )
    
    assert alternative == "search_compliance_documents"


def test_door_compliance_recovery(recovery):
    """Test recovery for door compliance failures."""
    alternative = recovery.get_contextual_alternative(
        failed_tool="validate_rule", 
        task_description="Validate doors against accessibility rules",
        error_message="door_compliance rule type not implemented"
    )
    
    assert alternative == "search_compliance_documents"


def test_relationship_type_recovery(recovery):
    """Test recovery for relationship type failures.""" 
    alternative = recovery.get_contextual_alternative(
        failed_tool="find_related",
        task_description="Find relationship between spaces",
        error_message="relationship not implemented"
    )
    
    assert alternative == "calculate_distances"


def test_tool_specific_alternatives_spatial(recovery):
    """Test tool-specific alternatives for spatial errors."""
    alternative = recovery.get_contextual_alternative(
        failed_tool="find_related",
        task_description="Analyze spatial connections",  
        error_message="Some other error with spatial keyword"
    )
    
    # Should match "spatial" keyword in error and return first alternative
    assert alternative == "query_elements"


def test_tool_specific_alternatives_stair_compliance(recovery):
    """Test tool-specific alternatives for stair compliance."""
    alternative = recovery.get_contextual_alternative(
        failed_tool="validate_rule",
        task_description="Check stair_compliance for building",
        error_message="Unknown error"
    )
    
    # Should match "stair_compliance" in task description
    assert alternative == "search_compliance_documents"


def test_fallback_to_default_tool(recovery):
    """Test fallback to default tool when no specific alternatives found."""
    alternative = recovery.get_contextual_alternative(
        failed_tool="unknown_tool",
        task_description="Do something with no specific keywords",
        error_message="Generic error message"
    )
    
    # Should return default safe alternative
    assert alternative == "query_elements"


def test_case_insensitive_matching(recovery):
    """Test that keyword matching is case insensitive."""
    alternative = recovery.get_contextual_alternative(
        failed_tool="validate_rule",
        task_description="Check STAIR_COMPLIANCE requirements",
        error_message="ERROR with SPATIAL elements"
    )
    
    # Should match "STAIR_COMPLIANCE" in uppercase
    assert alternative == "search_compliance_documents"


def test_smart_tool_alternatives_structure():
    """Test that SMART_TOOL_ALTERNATIVES has the expected structure."""
    assert isinstance(SMART_TOOL_ALTERNATIVES, dict)
    
    # Check find_related alternatives
    assert "find_related" in SMART_TOOL_ALTERNATIVES
    find_related_alts = SMART_TOOL_ALTERNATIVES["find_related"]
    assert "spatial" in find_related_alts
    assert "relationship" in find_related_alts
    assert isinstance(find_related_alts["spatial"], list)
    assert len(find_related_alts["spatial"]) > 0
    
    # Check validate_rule alternatives
    assert "validate_rule" in SMART_TOOL_ALTERNATIVES  
    validate_rule_alts = SMART_TOOL_ALTERNATIVES["validate_rule"]
    assert "stair_compliance" in validate_rule_alts
    assert "door_compliance" in validate_rule_alts
    assert isinstance(validate_rule_alts["stair_compliance"], list)
    assert len(validate_rule_alts["stair_compliance"]) > 0


def test_contextual_recovery_with_error_message(recovery, mock_llm):
    """Test that recovery passes error message to contextual alternatives."""
    # Create task and error
    task = Task(
        id="test-task",
        name="Find spatial relationships",
        description="Find spatial relationships for doors",
        tool_sequence=["find_related"],
        priority=Priority.MEDIUM
    )
    
    error_result = ExecutionResult(
        success=False,
        tool_name="find_related",
        output=None,
        error_message="spatial relationship not implemented"
    )
    
    context = {
        'available_tools': ['find_related', 'calculate_distances', 'query_elements'],
        'active_files': ['test.ifc']
    }
    
    # Execute recovery
    result = recovery.recover(task, error_result, context)
    
    # Should suggest appropriate alternative
    assert result.success == True
    assert result.strategy == "try_different_tool"
    assert result.modified_task is not None
    assert "calculate_distances" in result.modified_task.tool_sequence


def test_recovery_fallback_when_contextual_tool_unavailable(recovery, mock_llm):
    """Test recovery falls back to available tools when contextual choice isn't available."""
    task = Task(
        id="test-task", 
        name="Find spatial relationships",
        description="Find spatial relationships for doors",
        tool_sequence=["find_related"],
        priority=Priority.MEDIUM
    )
    
    error_result = ExecutionResult(
        success=False,
        tool_name="find_related",
        output=None,
        error_message="spatial relationship not implemented"
    )
    
    # Context without calculate_distances (contextual choice) 
    context = {
        'available_tools': ['find_related', 'query_elements', 'get_all_properties'],
        'active_files': ['test.ifc']
    }
    
    # Execute recovery
    result = recovery.recover(task, error_result, context)
    
    # Should fall back to available alternative
    assert result.success == True
    assert result.strategy == "try_different_tool" 
    assert result.modified_task is not None
    # Should use first available alternative (not the failed tool)
    assert result.modified_task.tool_sequence[0] in ['query_elements', 'get_all_properties']
    assert result.modified_task.tool_sequence[0] != 'find_related'


def test_multiple_keywords_in_error(recovery):
    """Test handling of multiple keywords in error message."""
    alternative = recovery.get_contextual_alternative(
        failed_tool="find_related",
        task_description="Analyze building elements",
        error_message="spatial relationship not implemented for stair_compliance"
    )
    
    # Should prioritize "spatial" match over "stair_compliance" since it's first
    assert alternative == "calculate_distances"


def test_keyword_priority_in_task_description(recovery):
    """Test keyword matching prioritizes task description."""
    alternative = recovery.get_contextual_alternative(
        failed_tool="validate_rule",
        task_description="Check door_compliance requirements for spatial analysis", 
        error_message="Generic error"
    )
    
    # Should match "door_compliance" in task description
    assert alternative == "search_compliance_documents"


def test_empty_error_message(recovery):
    """Test handling of empty error messages."""
    alternative = recovery.get_contextual_alternative(
        failed_tool="unknown_tool",
        task_description="Generic task description",
        error_message=""
    )
    
    # Should return default alternative
    assert alternative == "query_elements"


def test_none_error_handling(recovery):
    """Test graceful handling of None values."""
    alternative = recovery.get_contextual_alternative(
        failed_tool=None,
        task_description="Task description",
        error_message="Some error"
    )
    
    # Should handle None failed_tool gracefully
    assert alternative == "query_elements"