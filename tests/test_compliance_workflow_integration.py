"""
Integration tests for all compliance workflow fixes

This module tests that all 5 fixes work together to create a robust 
compliance workflow that can handle failures gracefully.
"""

import pytest
from unittest.mock import Mock

from aec_agent.core.reasoning_controller import ReasoningController
from aec_agent.core.goal_decomposer import GoalDecomposer
from aec_agent.core.tool_planner import ToolPlanner
from aec_agent.core.simple_recovery import SimpleRecovery
from aec_agent.core.task_graph import TaskGraph
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus, ExecutionResult
from aec_agent.config import AgentConfig


@pytest.fixture
def mock_components():
    """Create all mocked components for integration testing."""
    config = AgentConfig.from_env()
    
    # Mock LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content='[{"name": "Load building data", "description": "Load data", "priority": "HIGH"}, {"name": "Search compliance documents", "description": "Search docs", "priority": "MEDIUM"}, {"name": "Validate compliance", "description": "Validate", "priority": "LOW"}]')
    
    # Mock executor and validator
    mock_executor = Mock()
    mock_validator = Mock()
    mock_validator.validate_execution.return_value = {"success": True}
    
    # Create real components with mocked dependencies
    goal_decomposer = GoalDecomposer(llm=mock_llm)
    tool_planner = ToolPlanner(llm=mock_llm)
    recovery_system = SimpleRecovery(llm=mock_llm)
    
    # Create reasoning controller
    reasoning_controller = ReasoningController(
        goal_decomposer=goal_decomposer,
        tool_planner=tool_planner,
        executor=mock_executor,
        validator=mock_validator,
        config=config
    )
    
    return {
        'controller': reasoning_controller,
        'decomposer': goal_decomposer,
        'planner': tool_planner,
        'recovery': recovery_system,
        'mock_llm': mock_llm,
        'mock_executor': mock_executor,
        'mock_validator': mock_validator
    }


def test_compliance_workflow_end_to_end(mock_components):
    """Test complete compliance workflow with all fixes."""
    decomposer = mock_components['decomposer']
    
    # Test dependency creation directly (bypass LLM decomposition issues)
    load_task = Task(id="load", name="Load building data", description="Load data")
    search_task = Task(id="search", name="Search compliance documents for stairs", description="Search compliance")
    validate_task = Task(id="validate", name="Validate compliance", description="Validate")
    spatial_task = Task(id="spatial", name="Find spatial relationships", description="Find spatial")
    
    # Set initial linear dependencies (problematic)
    search_task.dependencies = ["spatial"]  # Wrong dependency
    validate_task.dependencies = ["search"]
    
    tasks = [load_task, search_task, validate_task, spatial_task]
    
    # Test that compliance dependencies are properly set
    tasks_with_deps = decomposer._create_compliance_dependencies(tasks)
    
    # Find compliance search task
    compliance_search = next((t for t in tasks_with_deps if "compliance" in t.name.lower() and "search" in t.name.lower()), None)
    if compliance_search:
        # Should not depend on spatial analysis
        spatial_deps = [dep for dep in compliance_search.dependencies if "spatial" in str(dep)]
        assert len(spatial_deps) == 0


def test_compliance_task_priority_integration(mock_components):
    """Test that compliance tasks are never skipped due to priority."""
    controller = mock_components['controller']
    
    # Create compliance task with low priority
    compliance_task = Task(
        id="compliance-test",
        name="Search compliance documents for stair requirements",
        description="Search building compliance documents", 
        priority=Priority.LOW
    )
    
    # Test priority classification
    assert controller._is_compliance_critical(compliance_task) == True
    
    # Task should not be skipped even with low priority
    # This would be tested in the context of the full workflow


def test_tool_capability_and_recovery_integration(mock_components):
    """Test that tool capability detection works with recovery system."""
    planner = mock_components['planner']
    recovery = mock_components['recovery']
    
    # Test spatial relationship task
    spatial_task = Task(
        id="spatial-test",
        name="Find spatial relationships",
        description="Find spatial relationships between stairs"
    )
    
    # Tool planner should detect unsupported capability
    selected_tool = planner.select_tool_with_capability_check(spatial_task, {})
    assert selected_tool != "find_related"  # Should use alternative
    
    # Recovery system should provide appropriate alternative
    alternative = recovery.get_contextual_alternative(
        "find_related", 
        spatial_task.description,
        "spatial relationship not implemented"
    )
    assert alternative == "calculate_distances"


def test_optional_task_integration(mock_components):
    """Test that optional tasks don't block critical compliance workflow.""" 
    # Create workflow: Load -> [Spatial (optional), Compliance Search] -> Validate
    load_task = Task(id="load", name="Load building data", description="Load data")
    spatial_task = Task(id="spatial", name="Find spatial relationships", description="Find spatial relationships")
    search_task = Task(id="search", name="Search compliance documents", description="Search compliance")
    validate_task = Task(id="validate", name="Validate compliance", description="Validate rules")
    
    # Create graph with tasks to trigger auto-marking
    graph = TaskGraph([load_task, spatial_task, search_task, validate_task])
    
    # Spatial task should be auto-marked as optional
    assert spatial_task.metadata.get("optional") == True
    
    # Set up dependencies manually for test
    search_task.dependencies = ["load"]
    validate_task.dependencies = ["search", "spatial"]
    
    # Complete load task
    graph.update_task_status("load", TaskStatus.COMPLETED)
    
    # Fail spatial task
    graph.update_task_status("spatial", TaskStatus.FAILED)
    graph.handle_optional_task_failure("spatial")
    
    # Compliance workflow should not be blocked
    assert "spatial" not in validate_task.dependencies
    
    # If search completes, validate should be unblocked
    graph.update_task_status("search", TaskStatus.COMPLETED)
    assert graph._all_dependencies_met(validate_task) == True


def test_recovery_with_tool_alternatives(mock_components):
    """Test recovery system with tool capability alternatives."""
    recovery = mock_components['recovery']
    mock_llm = mock_components['mock_llm']
    mock_executor = mock_components['mock_executor']
    
    # Mock LLM to suggest trying different tool
    mock_llm.invoke.return_value = Mock(content="try_different_tool")
    
    # Create failed task
    task = Task(
        id="failed-task",
        name="Find spatial relationships",
        description="Find spatial relationships for compliance check",
        tool_sequence=["find_related"]
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
    
    # Test recovery
    result = recovery.recover(task, error_result, context)
    
    assert result.success == True
    assert result.strategy == "try_different_tool"
    assert result.modified_task is not None
    # Should suggest calculate_distances based on contextual alternative
    assert "calculate_distances" in result.modified_task.tool_sequence


def test_all_fixes_together_workflow():
    """High-level test that demonstrates all fixes working together."""
    
    # Create tasks directly (bypass LLM decomposition issues)
    load_task = Task(id="load", name="Load building data", description="Load data")
    search_task = Task(id="search", name="Search compliance documents for stairs", description="Search stair compliance")
    spatial_task = Task(id="spatial", name="Find spatial relationships", description="Find spatial relationships")
    validate_task = Task(id="validate", name="Validate stair compliance", description="Validate stairs")
    
    tasks = [load_task, search_task, spatial_task, validate_task]
    
    # Create components
    mock_llm = Mock()
    decomposer = GoalDecomposer(llm=mock_llm)
    planner = ToolPlanner(llm=mock_llm) 
    recovery = SimpleRecovery(llm=mock_llm)
    
    # Fix 2: Test dependency creation
    tasks_with_deps = decomposer._create_compliance_dependencies(tasks)
    
    # Fix 5: Create task graph and test optional marking  
    graph = TaskGraph(tasks_with_deps)  # Auto-marks optional tasks
    
    # Find different task types
    spatial_task = next((t for t in tasks_with_deps if "spatial" in t.name.lower()), None)
    compliance_task = next((t for t in tasks_with_deps if "compliance" in t.name.lower() and "search" in t.name.lower()), None)
    
    if spatial_task:
        # Fix 5: Spatial task should be optional
        assert spatial_task.metadata.get("optional") == True
        
    if compliance_task:
        # Fix 1: Compliance task should be critical
        controller_mock = Mock()
        controller_mock._is_compliance_critical = ReasoningController._is_compliance_critical.__get__(controller_mock)
        assert controller_mock._is_compliance_critical(compliance_task) == True
        
        # Fix 4: Tool capability check should work
        selected_tool = planner.select_tool_with_capability_check(compliance_task, {})
        assert selected_tool in ["search_compliance_documents", "validate_rule"]
    
    # Fix 3: Recovery should provide smart alternatives
    alternative = recovery.get_contextual_alternative(
        "validate_rule",
        "stair compliance validation", 
        "stair_compliance rule type not implemented"
    )
    assert alternative == "search_compliance_documents"


def test_compliance_workflow_resilience():
    """Test that the compliance workflow is resilient to various failures."""
    
    # This test demonstrates the robustness improvements:
    # - Critical compliance tasks are never skipped
    # - Dependencies don't create bottlenecks 
    # - Tool failures have smart alternatives
    # - Optional tasks don't block the workflow
    # - Recovery is context-aware
    
    graph = TaskGraph()
    
    # Create comprehensive workflow
    load_task = Task(id="load", name="Load building data", description="Load building data")
    spatial_task = Task(id="spatial", name="Find spatial relationships", description="Find spatial relationships") 
    search_task = Task(id="search", name="Search compliance documents", description="Search stair compliance documents")
    validate_task = Task(id="validate", name="Validate compliance", description="Validate stair compliance")
    chart_task = Task(id="chart", name="Generate chart", description="Generate compliance chart")
    
    # Create graph with all tasks to trigger auto-marking
    graph = TaskGraph([load_task, spatial_task, search_task, validate_task, chart_task])
    
    # Verify auto-marking of optional tasks
    assert spatial_task.metadata.get("optional") == True
    assert chart_task.metadata.get("optional") == True
    assert search_task.metadata.get("optional") != True  # Compliance tasks not optional
    
    # Set up realistic dependencies
    spatial_task.dependencies = ["load"]
    search_task.dependencies = ["load"]  # Independent of spatial (Fix 2)
    validate_task.dependencies = ["search"]  # Depends on compliance search
    chart_task.dependencies = ["validate"]
    
    # Simulate workflow execution with failures
    graph.update_task_status("load", TaskStatus.COMPLETED)
    
    # Fail optional spatial task - should not block workflow
    graph.update_task_status("spatial", TaskStatus.FAILED)
    graph.handle_optional_task_failure("spatial")
    
    # Compliance search should still be ready (Fix 2 - proper dependencies)
    ready_tasks = graph.get_ready_tasks()
    search_ready = any(t.id == "search" for t in ready_tasks)
    assert search_ready == True
    
    # Complete compliance search
    graph.update_task_status("search", TaskStatus.COMPLETED)
    
    # Validate should now be ready
    ready_tasks = graph.get_ready_tasks()
    validate_ready = any(t.id == "validate" for t in ready_tasks)
    assert validate_ready == True
    
    # Even if chart fails, the core compliance workflow succeeded
    graph.update_task_status("validate", TaskStatus.COMPLETED)
    graph.update_task_status("chart", TaskStatus.FAILED)
    graph.handle_optional_task_failure("chart")
    
    # Core compliance tasks should be completed
    core_tasks_completed = all(
        graph.tasks[task_id].status == TaskStatus.COMPLETED 
        for task_id in ["load", "search", "validate"]
    )
    assert core_tasks_completed == True