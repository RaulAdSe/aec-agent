"""
Tests for Fix 1: Compliance Task Priority Classification

This module tests that compliance-critical tasks are never marked as skippable,
even when they have low priority.
"""

import pytest
from unittest.mock import Mock, MagicMock

from aec_agent.core.reasoning_controller import ReasoningController
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus
from aec_agent.core.task_graph import TaskGraph
from aec_agent.config import AgentConfig


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for ReasoningController."""
    config = AgentConfig.from_env()
    
    goal_decomposer = Mock()
    tool_planner = Mock()
    executor = Mock()
    validator = Mock()
    
    return {
        'goal_decomposer': goal_decomposer,
        'tool_planner': tool_planner,
        'executor': executor,
        'validator': validator,
        'config': config
    }


@pytest.fixture
def controller(mock_dependencies):
    """Create a ReasoningController with mocked dependencies."""
    return ReasoningController(**mock_dependencies)


def test_compliance_search_never_skipped(controller):
    """Test that compliance search tasks are never marked as skippable."""
    # Create compliance search task
    task = Task(
        id="test_compliance",
        name="Search compliance documents for stair requirements",
        description="Search building compliance documents for stair requirements",
        priority=Priority.LOW
    )
    
    # Should be marked as critical despite low priority
    assert controller._is_compliance_critical(task) == True


def test_non_compliance_tasks_can_be_skipped(controller):
    """Test that non-compliance tasks can still be skipped."""
    task = Task(
        id="test_chart",
        name="Generate pretty charts",
        description="Generate visualization charts for data presentation",
        priority=Priority.LOW
    )
    assert controller._is_compliance_critical(task) == False


def test_compliance_validation_is_critical(controller):
    """Test that validation tasks are considered critical."""
    task = Task(
        id="test_validate",
        name="Validate building against fire safety rules",
        description="Validate building structure against fire safety compliance rules",
        priority=Priority.LOW
    )
    assert controller._is_compliance_critical(task) == True


def test_load_building_data_is_critical(controller):
    """Test that loading building data is considered critical."""
    task = Task(
        id="test_load",
        name="Load building data from IFC file",
        description="Load building data from IFC file for analysis",
        priority=Priority.LOW
    )
    assert controller._is_compliance_critical(task) == True


def test_building_analysis_is_critical(controller):
    """Test that building analysis tasks are considered critical."""
    task = Task(
        id="test_building",
        name="Analyze building structure for compliance",
        description="Analyze building structure for compliance requirements",
        priority=Priority.LOW
    )
    assert controller._is_compliance_critical(task) == True


def test_case_insensitive_keywords(controller):
    """Test that keyword matching is case-insensitive."""
    task = Task(
        id="test_case",
        name="SEARCH COMPLIANCE DOCUMENTS",
        description="SEARCH COMPLIANCE DOCUMENTS IN DATABASE",
        priority=Priority.LOW
    )
    assert controller._is_compliance_critical(task) == True


def test_partial_keyword_matching(controller):
    """Test that partial keyword matching works."""
    task = Task(
        id="test_partial",
        name="Task to validate structural compliance requirements",
        description="Task to validate structural compliance requirements for the building",
        priority=Priority.LOW
    )
    assert controller._is_compliance_critical(task) == True


def test_multiple_keywords_in_task(controller):
    """Test task with multiple critical keywords."""
    task = Task(
        id="test_multiple",
        name="Load and validate building compliance documents",
        description="Load and validate building compliance documents from database",
        priority=Priority.LOW
    )
    assert controller._is_compliance_critical(task) == True


def test_non_critical_task_examples(controller):
    """Test various non-critical task examples."""
    non_critical_tasks = [
        "Generate visualization charts",
        "Create summary report",
        "Export data to Excel",
        "Send notification email",
        "Update user interface"
    ]
    
    for task_name in non_critical_tasks:
        task = Task(
            id=f"test_{task_name.replace(' ', '_').lower()}",
            name=task_name,
            description=f"Task to {task_name.lower()}",
            priority=Priority.LOW
        )
        assert controller._is_compliance_critical(task) == False, f"Task '{task_name}' should not be critical"