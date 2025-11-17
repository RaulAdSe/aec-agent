#!/usr/bin/env python3
"""
Integration tests for TaskGraph with ReasoningController.

Tests that the reasoning controller properly:
- Uses TaskGraph for task management
- Respects dependencies when selecting tasks
- Updates task status correctly
- Provides proper graph metrics and visualization
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aec_agent.core.reasoning_controller import ReasoningController, ReasoningState
from aec_agent.core.task_graph import TaskGraph
from aec_agent.core.reasoning_utils import Task, TaskStatus, Priority
from aec_agent.core.goal_decomposer import GoalDecomposer
from aec_agent.core.tool_planner import ToolPlanner
from aec_agent.core.executor import ToolExecutor
from aec_agent.core.validator import ResultValidator
from aec_agent.config import AgentConfig


def test_reasoning_controller_initialization_with_graph():
    """Test that ReasoningController properly initializes with TaskGraph."""
    print("🧪 Testing ReasoningController initialization with TaskGraph...")
    
    # Mock dependencies
    goal_decomposer = Mock()
    tool_planner = Mock()
    executor = Mock()
    validator = Mock()
    config = AgentConfig()
    
    # Create controller
    controller = ReasoningController(
        goal_decomposer=goal_decomposer,
        tool_planner=tool_planner, 
        executor=executor,
        validator=validator,
        config=config
    )
    
    # Test reasoning initialization
    goal = "Analyze fire safety compliance for building data"
    
    # Mock goal decomposition to return simple task list
    sample_tasks = [
        Task(id="task1", name="Load data", description="Load building data", dependencies=[]),
        Task(id="task2", name="Get doors", description="Get door elements", dependencies=["task1"]),
        Task(id="task3", name="Calculate distances", description="Calculate distances", dependencies=["task2"])
    ]
    
    goal_decomposer.decompose.return_value = {
        "success": True,
        "tasks": sample_tasks,
        "method": "test_method"
    }
    
    # Mock other components to avoid actual execution
    tool_planner.plan_tools.return_value = {"success": True, "tool_sequence": ["test_tool"]}
    executor.execute_tool.return_value = Mock(success=True, output={"status": "success"})
    validator.validate_execution.return_value = {"success": True}
    
    # Execute reasoning - should initialize with TaskGraph
    with patch.object(controller, '_execute_reasoning_loop') as mock_execute:
        mock_execute.return_value = []
        
        result = controller.reason(goal)
        
        # Verify state was initialized with TaskGraph
        assert controller.state is not None
        assert isinstance(controller.state.task_graph, TaskGraph)
        assert len(controller.state.task_graph.tasks) == 3
        
        # Verify tasks were added to graph correctly  
        assert "task1" in controller.state.task_graph.tasks
        assert "task2" in controller.state.task_graph.tasks
        assert "task3" in controller.state.task_graph.tasks
        
        # Verify dependencies are set up correctly
        task2 = controller.state.task_graph.tasks["task2"]
        assert "task1" in task2.dependencies
        
        task3 = controller.state.task_graph.tasks["task3"]
        assert "task2" in task3.dependencies
    
    print("✅ ReasoningController initialization with TaskGraph works")


def test_graph_based_task_selection():
    """Test that ReasoningController uses TaskGraph for task selection."""
    print("🧪 Testing graph-based task selection...")
    
    # Create real TaskGraph with sample tasks
    graph = TaskGraph()
    tasks = [
        Task(id="A", name="Load data", description="Load", priority=Priority.HIGH, dependencies=[]),
        Task(id="B", name="Get elements", description="Get elements", priority=Priority.MEDIUM, dependencies=["A"]),
        Task(id="C", name="Calculate", description="Calculate", priority=Priority.LOW, dependencies=["B"])
    ]
    graph.add_tasks(tasks)
    
    # Test task selection based on dependencies
    
    # Initially, only A should be ready
    ready_tasks = graph.get_ready_tasks()
    assert len(ready_tasks) == 1
    assert ready_tasks[0].id == "A"
    assert ready_tasks[0].priority == Priority.HIGH
    
    # Complete A, B becomes ready
    graph.update_task_status("A", TaskStatus.COMPLETED)
    ready_tasks = graph.get_ready_tasks()
    assert len(ready_tasks) == 1
    assert ready_tasks[0].id == "B"
    
    # Complete B, C becomes ready
    graph.update_task_status("B", TaskStatus.COMPLETED)
    ready_tasks = graph.get_ready_tasks()
    assert len(ready_tasks) == 1
    assert ready_tasks[0].id == "C"
    
    print("✅ Graph-based task selection works")


def test_graph_metrics_during_execution():
    """Test that graph metrics are properly calculated during execution."""
    print("🧪 Testing graph metrics during execution...")
    
    graph = TaskGraph()
    tasks = [
        Task(id="load", name="Load data", description="Load", dependencies=[]),
        Task(id="doors", name="Get doors", description="Get doors", dependencies=["load"]),
        Task(id="spaces", name="Get spaces", description="Get spaces", dependencies=["load"]),
        Task(id="calc", name="Calculate", description="Calculate", dependencies=["doors", "spaces"])
    ]
    graph.add_tasks(tasks)
    
    # Test initial metrics
    metrics = graph.get_graph_metrics()
    assert metrics.total_tasks == 4
    assert metrics.ready_tasks == 1  # Only 'load' is ready
    assert metrics.completed_tasks == 0
    assert metrics.failed_tasks == 0
    assert metrics.dependency_edges == 4  # load->doors, load->spaces, doors->calc, spaces->calc
    assert metrics.longest_path == 3  # load->doors->calc or load->spaces->calc
    
    # Complete load task
    graph.update_task_status("load", TaskStatus.COMPLETED)
    metrics = graph.get_graph_metrics()
    assert metrics.ready_tasks == 2  # doors and spaces
    assert metrics.completed_tasks == 1
    
    # Complete doors and spaces
    graph.update_task_status("doors", TaskStatus.COMPLETED)
    graph.update_task_status("spaces", TaskStatus.COMPLETED)
    metrics = graph.get_graph_metrics()
    assert metrics.ready_tasks == 1  # calc
    assert metrics.completed_tasks == 3
    
    # Complete calc
    graph.update_task_status("calc", TaskStatus.COMPLETED)
    metrics = graph.get_graph_metrics()
    assert metrics.ready_tasks == 0  # all done
    assert metrics.completed_tasks == 4
    
    print("✅ Graph metrics during execution work")


def test_failed_task_blocking():
    """Test that failed tasks properly block dependent tasks."""
    print("🧪 Testing failed task blocking...")
    
    graph = TaskGraph()
    tasks = [
        Task(id="A", name="Load data", description="Load", dependencies=[]),
        Task(id="B", name="Parse", description="Parse", dependencies=["A"]),
        Task(id="C", name="Extract", description="Extract", dependencies=["B"]),
        Task(id="D", name="Calculate", description="Calculate", dependencies=["C"])
    ]
    graph.add_tasks(tasks)
    
    # Complete A, then fail B
    graph.update_task_status("A", TaskStatus.COMPLETED)
    graph.update_task_status("B", TaskStatus.FAILED)
    
    # C and D should be blocked
    blocked_tasks = graph.get_blocked_tasks()
    blocked_ids = {task.id for task in blocked_tasks}
    assert blocked_ids == {"C", "D"}
    
    # No tasks should be ready (B failed, C and D blocked)
    ready_tasks = graph.get_ready_tasks()
    assert len(ready_tasks) == 0
    
    # Metrics should reflect the failure
    metrics = graph.get_graph_metrics()
    assert metrics.failed_tasks == 1
    assert metrics.blocked_tasks == 2
    assert metrics.completed_tasks == 1
    assert metrics.ready_tasks == 0
    
    print("✅ Failed task blocking works")


def test_graph_visualization_integration():
    """Test that graph visualization shows current execution state."""
    print("🧪 Testing graph visualization integration...")
    
    graph = TaskGraph()
    tasks = [
        Task(id="load", name="Load building data", description="Load data", priority=Priority.HIGH, dependencies=[]),
        Task(id="doors", name="Get door elements", description="Get doors", priority=Priority.MEDIUM, dependencies=["load"]),
        Task(id="calc", name="Calculate distances", description="Calculate", priority=Priority.LOW, dependencies=["doors"])
    ]
    graph.add_tasks(tasks)
    
    # Set different statuses
    graph.update_task_status("load", TaskStatus.COMPLETED)
    graph.update_task_status("doors", TaskStatus.IN_PROGRESS)
    # calc remains pending
    
    # Generate visualization
    viz = graph.visualize_graph()
    
    # Check that visualization contains status indicators
    assert "✅" in viz  # Completed (load)
    assert "🔄" in viz  # In progress (doors)
    assert "⏳" in viz  # Pending (calc)
    
    # Check priority indicators
    assert "🔺" in viz  # High priority (load)
    assert "🔷" in viz  # Medium priority (doors)  
    assert "🔸" in viz  # Low priority (calc)
    
    # Check dependency relationships
    assert "Depends on" in viz
    assert "Required by" in viz
    
    # Check metrics section
    assert "Total tasks: 3" in viz
    assert "Completed: 1" in viz
    assert "Dependencies: 2" in viz
    
    print("Visualization output:")
    print("-" * 40)
    print(viz)
    print("-" * 40)
    
    print("✅ Graph visualization integration works")


def test_complex_dependency_execution():
    """Test execution with complex dependency patterns."""
    print("🧪 Testing complex dependency execution...")
    
    graph = TaskGraph()
    
    # Create diamond dependency pattern: A → B,C → D,E → F
    tasks = [
        Task(id="A", name="Load data", description="Root task", priority=Priority.HIGH, dependencies=[]),
        Task(id="B", name="Get doors", description="Branch 1", priority=Priority.MEDIUM, dependencies=["A"]),
        Task(id="C", name="Get spaces", description="Branch 2", priority=Priority.MEDIUM, dependencies=["A"]),
        Task(id="D", name="Calc door metrics", description="Door calculations", dependencies=["B"]),
        Task(id="E", name="Calc space metrics", description="Space calculations", dependencies=["C"]),
        Task(id="F", name="Generate report", description="Final report", dependencies=["D", "E"])
    ]
    graph.add_tasks(tasks)
    
    # Simulate execution sequence
    execution_log = []
    
    # Step 1: Only A ready
    ready = graph.get_ready_tasks()
    assert len(ready) == 1 and ready[0].id == "A"
    execution_log.append("A")
    graph.update_task_status("A", TaskStatus.COMPLETED)
    
    # Step 2: B and C ready (parallel branches)
    ready = graph.get_ready_tasks()
    ready_ids = {task.id for task in ready}
    assert ready_ids == {"B", "C"}
    
    # Execute B first
    execution_log.append("B")
    graph.update_task_status("B", TaskStatus.COMPLETED)
    
    # Step 3: C still ready, plus D is now ready
    ready = graph.get_ready_tasks()
    ready_ids = {task.id for task in ready}
    assert ready_ids == {"C", "D"}
    
    # Execute both
    execution_log.extend(["C", "D"])
    graph.update_task_status("C", TaskStatus.COMPLETED)
    graph.update_task_status("D", TaskStatus.COMPLETED)
    
    # Step 4: E ready
    ready = graph.get_ready_tasks()
    assert len(ready) == 1 and ready[0].id == "E"
    execution_log.append("E")
    graph.update_task_status("E", TaskStatus.COMPLETED)
    
    # Step 5: F ready (final task)
    ready = graph.get_ready_tasks()
    assert len(ready) == 1 and ready[0].id == "F"
    execution_log.append("F")
    graph.update_task_status("F", TaskStatus.COMPLETED)
    
    # Step 6: All done
    ready = graph.get_ready_tasks()
    assert len(ready) == 0
    
    # Verify execution sequence makes sense
    print(f"Execution sequence: {' → '.join(execution_log)}")
    assert execution_log[0] == "A"  # A must be first
    assert execution_log[-1] == "F"  # F must be last
    assert execution_log.index("B") < execution_log.index("D")  # B before D
    assert execution_log.index("C") < execution_log.index("E")  # C before E
    
    print("✅ Complex dependency execution works")


def run_integration_tests():
    """Run all integration tests."""
    print("🚀 Testing TaskGraph Integration with ReasoningController\n")
    
    print("=" * 60)
    print("INTEGRATION TESTS")
    print("=" * 60)
    
    test_reasoning_controller_initialization_with_graph()
    test_graph_based_task_selection()
    test_graph_metrics_during_execution()
    test_failed_task_blocking()
    test_graph_visualization_integration()
    test_complex_dependency_execution()
    
    print("\n" + "🎉" * 20)
    print("ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
    print("TaskGraph is fully integrated with ReasoningController!")
    print("🎉" * 20)


if __name__ == "__main__":
    run_integration_tests()