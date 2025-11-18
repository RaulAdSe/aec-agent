"""
Tests for Fix 5: Optional Task Support

This module tests that optional tasks don't block dependents when they fail
and that the system can gracefully handle optional task failures.
"""

import pytest

from aec_agent.core.task_graph import TaskGraph
from aec_agent.core.reasoning_utils import Task, TaskStatus, Priority


def test_mark_task_optional():
    """Test that tasks can be marked as optional."""
    graph = TaskGraph()
    
    task = Task(
        id="test-task",
        name="Test task",
        description="A test task"
    )
    
    graph.add_task(task)
    graph.mark_task_optional("test-task")
    
    assert task.metadata.get("optional") == True


def test_optional_task_failure_unblocks_dependents():
    """Test that optional task failure doesn't block dependents."""
    graph = TaskGraph()
    
    # Create tasks: A (optional) -> B -> C
    task_a = Task(id="a", name="Find spatial relationships", description="Find spatial relationships")  
    task_b = Task(id="b", name="Validate compliance", description="Validate compliance", dependencies=["a"])
    task_c = Task(id="c", name="Generate report", description="Generate report", dependencies=["b"])
    
    graph.add_task(task_a)
    graph.add_task(task_b) 
    graph.add_task(task_c)
    graph.mark_task_optional("a")
    
    # Set task B as blocked initially due to dependency on A
    task_b.status = TaskStatus.BLOCKED
    
    # Fail optional task A
    graph.update_task_status("a", TaskStatus.FAILED)
    graph.handle_optional_task_failure("a")
    
    # Task B should be unblocked
    assert "a" not in graph.tasks["b"].dependencies
    assert graph.tasks["b"].status == TaskStatus.PENDING


def test_spatial_tasks_auto_marked_optional():
    """Test that spatial relationship tasks are automatically optional."""
    task = Task(
        id="test-spatial",
        name="Find spatial relationships for stairs",
        description="Analyze spatial relationships between stairs"
    )
    
    graph = TaskGraph([task])
    
    assert task.metadata.get("optional") == True


def test_chart_tasks_auto_marked_optional():
    """Test that chart/visualization tasks are automatically optional."""
    task = Task(
        id="test-chart",
        name="Generate visualization chart",
        description="Create chart for data presentation"
    )
    
    graph = TaskGraph([task])
    
    assert task.metadata.get("optional") == True


def test_relationship_tasks_auto_marked_optional():
    """Test that relationship tasks are automatically optional."""
    task = Task(
        id="test-relationship",
        name="Analyze relationships",
        description="Analyze relationship patterns in data"
    )
    
    graph = TaskGraph([task])
    
    assert task.metadata.get("optional") == True


def test_compliance_tasks_not_marked_optional():
    """Test that compliance tasks are not automatically marked optional."""
    task = Task(
        id="test-compliance",
        name="Validate compliance rules",
        description="Validate building against compliance requirements"
    )
    
    graph = TaskGraph([task])
    
    assert task.metadata.get("optional") != True


def test_load_tasks_not_marked_optional():
    """Test that load tasks are not automatically marked optional."""
    task = Task(
        id="test-load",
        name="Load building data",
        description="Load building data from file"
    )
    
    graph = TaskGraph([task])
    
    assert task.metadata.get("optional") != True


def test_optional_task_with_multiple_dependents():
    """Test optional task failure unblocks multiple dependents."""
    graph = TaskGraph()
    
    # Create tasks: A (optional) -> B, A -> C
    task_a = Task(id="a", name="Generate spatial analysis", description="Generate spatial analysis")
    task_b = Task(id="b", name="Task B", description="Task B", dependencies=["a"])
    task_c = Task(id="c", name="Task C", description="Task C", dependencies=["a"])
    
    graph.add_task(task_a)
    graph.add_task(task_b)
    graph.add_task(task_c)
    graph.mark_task_optional("a")
    
    # Set dependents as blocked
    task_b.status = TaskStatus.BLOCKED
    task_c.status = TaskStatus.BLOCKED
    
    # Fail optional task A
    graph.update_task_status("a", TaskStatus.FAILED)
    graph.handle_optional_task_failure("a")
    
    # Both dependents should be unblocked
    assert "a" not in graph.tasks["b"].dependencies
    assert "a" not in graph.tasks["c"].dependencies
    assert graph.tasks["b"].status == TaskStatus.PENDING
    assert graph.tasks["c"].status == TaskStatus.PENDING


def test_optional_task_with_other_dependencies():
    """Test that tasks with other dependencies remain blocked."""
    graph = TaskGraph()
    
    # Create tasks: A (optional) -> C, B -> C
    task_a = Task(id="a", name="Optional task", description="Optional visualization task")
    task_b = Task(id="b", name="Required task", description="Required load task") 
    task_c = Task(id="c", name="Dependent task", description="Task depending on both", dependencies=["a", "b"])
    
    graph.add_task(task_a)
    graph.add_task(task_b)
    graph.add_task(task_c)
    graph.mark_task_optional("a")
    
    # Set task C as blocked
    task_c.status = TaskStatus.BLOCKED
    
    # Fail optional task A, but B is still pending
    graph.update_task_status("a", TaskStatus.FAILED)
    graph.handle_optional_task_failure("a")
    
    # Task C should still be blocked because B is not completed
    assert "a" not in graph.tasks["c"].dependencies  # A removed
    assert "b" in graph.tasks["c"].dependencies      # B still required
    assert graph.tasks["c"].status == TaskStatus.BLOCKED  # Still blocked


def test_non_optional_task_failure_no_effect():
    """Test that non-optional task failure doesn't trigger special handling."""
    graph = TaskGraph()
    
    task_a = Task(id="a", name="Required task", description="Required task")
    task_b = Task(id="b", name="Dependent task", description="Dependent task", dependencies=["a"])
    
    graph.add_task(task_a)
    graph.add_task(task_b)
    # A is not marked as optional
    
    original_deps = list(task_b.dependencies)
    
    # Fail non-optional task A
    graph.update_task_status("a", TaskStatus.FAILED)
    graph.handle_optional_task_failure("a")  # Should have no effect
    
    # Task B dependencies should remain unchanged
    assert graph.tasks["b"].dependencies == original_deps


def test_handle_nonexistent_task():
    """Test handling of optional failure for non-existent task."""
    graph = TaskGraph()
    
    # Should not crash when handling failure of non-existent task
    graph.handle_optional_task_failure("nonexistent")


def test_all_dependencies_met_check():
    """Test the _all_dependencies_met helper method."""
    graph = TaskGraph()
    
    task_a = Task(id="a", name="Task A", description="Task A")
    task_b = Task(id="b", name="Task B", description="Task B") 
    task_c = Task(id="c", name="Task C", description="Task C", dependencies=["a", "b"])
    
    graph.add_task(task_a)
    graph.add_task(task_b)
    graph.add_task(task_c)
    
    # No dependencies completed
    assert graph._all_dependencies_met(task_c) == False
    
    # One dependency completed
    graph.update_task_status("a", TaskStatus.COMPLETED)
    assert graph._all_dependencies_met(task_c) == False
    
    # All dependencies completed
    graph.update_task_status("b", TaskStatus.COMPLETED)
    assert graph._all_dependencies_met(task_c) == True


def test_auto_mark_multiple_keywords():
    """Test that tasks with multiple optional keywords are marked optional."""
    task = Task(
        id="test",
        name="Create spatial visualization chart", 
        description="Generate chart showing spatial relationships"
    )
    
    graph = TaskGraph([task])
    
    # Should be marked optional due to multiple keywords
    assert task.metadata.get("optional") == True


def test_case_insensitive_optional_keywords():
    """Test that optional keyword detection is case insensitive."""
    task = Task(
        id="test",
        name="SPATIAL RELATIONSHIP ANALYSIS",
        description="ANALYZE SPATIAL PATTERNS"
    )
    
    graph = TaskGraph([task])
    
    assert task.metadata.get("optional") == True


def test_partial_keyword_matching():
    """Test that partial keyword matching works for optional detection."""
    task = Task(
        id="test",
        name="Analyze relationships between elements",
        description="Study how elements relate spatially"
    )
    
    graph = TaskGraph([task])
    
    # Should match "relationship" and "spatially"
    assert task.metadata.get("optional") == True


def test_empty_task_graph_auto_mark():
    """Test auto-marking on empty graph doesn't crash."""
    graph = TaskGraph()
    graph.auto_mark_optional_tasks()  # Should not crash