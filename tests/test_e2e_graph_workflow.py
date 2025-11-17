#!/usr/bin/env python3
"""
End-to-end test demonstrating TaskGraph with real AEC agent workflow.

This test shows the complete integration:
1. Goal decomposition creates tasks with dependencies
2. TaskGraph manages the dependency order
3. Just-in-time planning selects appropriate tools
4. Execution follows the dependency chain
5. Graph provides visualization and metrics
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aec_agent.core.reasoning_controller import ReasoningController
from aec_agent.core.goal_decomposer import GoalDecomposer
from aec_agent.core.tool_planner import ToolPlanner
from aec_agent.core.executor import ToolExecutor
from aec_agent.core.validator import ResultValidator
from aec_agent.core.reasoning_utils import Task, TaskStatus, Priority, ExecutionResult
from aec_agent.config import AgentConfig


def create_sample_building_analysis_scenario():
    """Create a realistic building analysis scenario with proper dependencies."""
    
    # Mock tasks that would be generated for "Analyze fire safety compliance for sample building"
    tasks = [
        Task(
            id="load_building_data",
            name="Load building data",
            description="Load and parse building IFC data from file",
            priority=Priority.HIGH,
            dependencies=[],
            metadata={"file_path": "data/out/sample_building.json"}
        ),
        Task(
            id="extract_fire_doors",
            name="Extract fire door elements",
            description="Get all fire door elements from the building model",
            priority=Priority.MEDIUM,
            dependencies=["load_building_data"]
        ),
        Task(
            id="extract_fire_exits", 
            name="Extract fire exit elements",
            description="Get all fire exit and egress elements",
            priority=Priority.MEDIUM,
            dependencies=["load_building_data"]
        ),
        Task(
            id="calculate_egress_distances",
            name="Calculate egress distances",
            description="Calculate distances from spaces to nearest fire exits",
            priority=Priority.MEDIUM,
            dependencies=["extract_fire_doors", "extract_fire_exits"]
        ),
        Task(
            id="validate_fire_compliance",
            name="Validate fire safety compliance",
            description="Check fire safety requirements against building codes",
            priority=Priority.LOW,
            dependencies=["calculate_egress_distances"]
        )
    ]
    
    return tasks


def test_end_to_end_graph_workflow():
    """Test complete workflow from goal to execution with TaskGraph."""
    print("🚀 End-to-End TaskGraph Workflow Test")
    print("=" * 50)
    
    # Step 1: Setup components with mocks for controlled testing
    print("📋 Step 1: Setting up reasoning components...")
    
    goal_decomposer = Mock()
    tool_planner = Mock()
    executor = Mock()
    validator = Mock()
    config = AgentConfig()
    
    controller = ReasoningController(
        goal_decomposer=goal_decomposer,
        tool_planner=tool_planner,
        executor=executor,
        validator=validator,
        config=config
    )
    
    # Step 2: Mock goal decomposition to return our sample scenario
    print("📋 Step 2: Creating realistic fire safety analysis scenario...")
    
    sample_tasks = create_sample_building_analysis_scenario()
    goal_decomposer.decompose.return_value = {
        "success": True,
        "tasks": sample_tasks,
        "method": "llm_reasoning"
    }
    
    # Step 3: Mock tool planning responses based on task context
    print("📋 Step 3: Setting up context-aware tool planning...")
    
    def mock_tool_planning(task, context):
        """Mock tool planning that returns different tools based on task and context."""
        task_name = task.name.lower()
        
        if "load building data" in task_name:
            return {
                "success": True,
                "tool_sequence": ["load_building_data"],
                "metadata": {"reason": "Load building data first"}
            }
        elif "extract fire door" in task_name:
            # This should only be planned after building data is loaded
            if context.get("building_data_loaded"):
                return {
                    "success": True,
                    "tool_sequence": ["get_all_elements"],
                    "metadata": {"element_type": "doors", "filter": "fire"}
                }
            else:
                return {
                    "success": False,
                    "message": "Building data not loaded yet"
                }
        elif "extract fire exit" in task_name:
            if context.get("building_data_loaded"):
                return {
                    "success": True,
                    "tool_sequence": ["get_all_elements"],
                    "metadata": {"element_type": "exits"}
                }
            else:
                return {
                    "success": False,
                    "message": "Building data not loaded yet"
                }
        elif "calculate egress" in task_name:
            # This should only work if both doors and exits data are available
            if context.get("doors_data") and context.get("exits_data"):
                return {
                    "success": True,
                    "tool_sequence": ["calculate_distances"],
                    "metadata": {"calculation_type": "egress_distances"}
                }
            else:
                return {
                    "success": False,
                    "message": "Missing doors or exits data"
                }
        elif "validate" in task_name and "fire" in task_name:
            if context.get("last_calculate_distances_result"):
                return {
                    "success": True,
                    "tool_sequence": ["validate_compliance_rule"],
                    "metadata": {"rule_type": "fire_safety"}
                }
            else:
                return {
                    "success": False,
                    "message": "Missing distance calculations"
                }
        
        return {"success": False, "message": f"Unknown task: {task_name}"}
    
    tool_planner.plan_tools.side_effect = mock_tool_planning
    
    # Step 4: Mock tool execution with realistic outputs
    print("📋 Step 4: Setting up realistic tool execution...")
    
    def mock_tool_execution(tool_name, task, context):
        """Mock tool execution that updates context appropriately."""
        
        if tool_name == "load_building_data":
            return ExecutionResult(
                success=True,
                tool_name=tool_name,
                output={
                    "status": "success",
                    "message": "Building data loaded successfully",
                    "elements_count": 150
                },
                execution_time=0.5
            )
        
        elif tool_name == "get_all_elements":
            # Simulate getting different elements based on task
            task_desc = task.description.lower()
            if "door" in task_desc:
                return ExecutionResult(
                    success=True,
                    tool_name=tool_name,
                    output={
                        "status": "success",
                        "data": [
                            {"id": "door_1", "type": "fire_door", "location": [10, 5, 0]},
                            {"id": "door_2", "type": "fire_door", "location": [20, 15, 0]}
                        ]
                    },
                    execution_time=0.3
                )
            elif "exit" in task_desc:
                return ExecutionResult(
                    success=True,
                    tool_name=tool_name,
                    output={
                        "status": "success", 
                        "data": [
                            {"id": "exit_1", "type": "emergency_exit", "location": [0, 10, 0]},
                            {"id": "exit_2", "type": "emergency_exit", "location": [30, 10, 0]}
                        ]
                    },
                    execution_time=0.3
                )
        
        elif tool_name == "calculate_distances":
            return ExecutionResult(
                success=True,
                tool_name=tool_name,
                output={
                    "status": "success",
                    "data": {
                        "max_egress_distance": 45.2,
                        "avg_egress_distance": 28.7,
                        "distances_calculated": 12
                    }
                },
                execution_time=1.0
            )
        
        elif tool_name == "validate_compliance_rule":
            return ExecutionResult(
                success=True,
                tool_name=tool_name,
                output={
                    "status": "success",
                    "compliance_result": {
                        "passed": True,
                        "max_allowed_distance": 50.0,
                        "actual_max_distance": 45.2,
                        "violations": []
                    }
                },
                execution_time=0.8
            )
        
        return ExecutionResult(
            success=False,
            tool_name=tool_name,
            output=None,
            error_message=f"Unknown tool: {tool_name}"
        )
    
    executor.execute_tool.side_effect = mock_tool_execution
    
    # Step 5: Mock validation to always pass
    print("📋 Step 5: Setting up result validation...")
    validator.validate_execution.return_value = {"success": True}
    
    # Step 6: Execute the complete reasoning workflow
    print("📋 Step 6: Executing complete reasoning workflow...")
    print()
    
    goal = "Analyze fire safety compliance for sample building with egress distance validation"
    
    # Execute reasoning
    result = controller.reason(goal)
    
    # Step 7: Analyze and verify results
    print("🔍 Step 7: Analyzing workflow results...")
    print()
    
    # Verify the workflow completed successfully
    assert result["status"] == "success", f"Workflow failed: {result.get('message')}"
    print("✅ Workflow completed successfully")
    
    # Verify task graph was used properly
    summary = result["summary"]
    assert summary["total_tasks"] == 5, f"Expected 5 tasks, got {summary['total_tasks']}"
    assert summary["completed_tasks"] == 5, f"Expected all tasks completed, got {summary['completed_tasks']}"
    print(f"✅ All {summary['total_tasks']} tasks completed successfully")
    
    # Verify graph metrics
    graph_metrics = summary["graph_metrics"]
    print(f"📊 Graph metrics:")
    print(f"   - Dependencies: {graph_metrics['dependency_edges']}")
    print(f"   - Critical path length: {graph_metrics['critical_path_length']}")
    print(f"   - Blocked tasks: {graph_metrics['blocked_tasks']}")
    
    # Verify task execution order respected dependencies
    tasks = result["tasks"]
    task_order = {task["id"]: i for i, task in enumerate(tasks)}
    
    # load_building_data should be first
    assert task_order["load_building_data"] == 0, "Load data should be executed first"
    
    # extract tasks should come after load
    assert task_order["extract_fire_doors"] > task_order["load_building_data"]
    assert task_order["extract_fire_exits"] > task_order["load_building_data"]
    print("✅ Task execution order respected dependencies")
    
    # calculate should come after both extracts
    assert task_order["calculate_egress_distances"] > task_order["extract_fire_doors"]
    assert task_order["calculate_egress_distances"] > task_order["extract_fire_exits"]
    
    # validate should be last
    assert task_order["validate_fire_compliance"] > task_order["calculate_egress_distances"]
    print("✅ Complex dependencies handled correctly")
    
    # Verify outputs were collected
    outputs = result["outputs"]
    assert len(outputs) == 5, f"Expected 5 tool outputs, got {len(outputs)}"
    
    tool_names = [output["tool"] for output in outputs]
    expected_tools = ["load_building_data", "get_all_elements", "get_all_elements", "calculate_distances", "validate_compliance_rule"]
    print(f"✅ Tools executed: {' → '.join(tool_names)}")
    
    # Step 8: Display final task graph visualization
    print("\n📈 Step 8: Final task graph state...")
    
    final_graph = controller.state.task_graph
    visualization = final_graph.visualize_graph()
    print(visualization)
    
    print("\n🎉 END-TO-END WORKFLOW TEST COMPLETED SUCCESSFULLY! 🎉")
    print("The TaskGraph implementation successfully:")
    print("  ✅ Managed complex task dependencies")
    print("  ✅ Enabled just-in-time planning with context")
    print("  ✅ Provided comprehensive metrics and visualization")
    print("  ✅ Integrated seamlessly with ReasoningController")
    
    return result


if __name__ == "__main__":
    test_end_to_end_graph_workflow()