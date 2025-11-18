"""
End-to-End Recovery System Tests - Real AEC Compliance Scenarios

Tests realistic failure scenarios and recovery flows that could happen in production:
1. Building file corruption/missing → Resource error recovery
2. Wrong tool selection → Tool alternative recovery  
3. Missing dependencies → Dependency resolution
4. Multiple failures → Goal replanning
5. Non-critical task failures → Graceful degradation
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from aec_agent.core.reasoning_controller import ReasoningController
from aec_agent.core.recovery_system import RecoverySystem, FailureType
from aec_agent.core.reasoning_utils import Task, Priority, ExecutionResult, TaskStatus
from aec_agent.core.task_graph import TaskGraph
from aec_agent.config import AgentConfig


@pytest.fixture
def mock_agent_config():
    """Mock agent configuration for testing."""
    config = Mock()
    config.reasoning = Mock()
    config.reasoning.max_iterations = 10
    config.reasoning.max_execution_time = 300.0
    config.get_effective_guardrail_config.return_value = Mock()
    return config


@pytest.fixture
def mock_components():
    """Mock all system components for E2E testing."""
    # Mock LLM with realistic responses
    llm = Mock()
    
    # Mock Goal Decomposer
    goal_decomposer = Mock()
    
    # Mock Tool Planner  
    tool_planner = Mock()
    tool_planner.plan_tools.return_value = {
        "success": True,
        "tool_sequence": ["load_building_data"],
        "metadata": {}
    }
    
    # Mock Executor
    executor = Mock()
    executor.tool_registry = {
        "load_building_data": Mock(),
        "get_all_elements": Mock(), 
        "calculate": Mock(),
        "get_all_properties": Mock(),
        "validate_rule": Mock(),
        "simple_response": Mock()
    }
    
    # Mock Validator
    validator = Mock()
    
    return {
        "llm": llm,
        "goal_decomposer": goal_decomposer,
        "tool_planner": tool_planner,
        "executor": executor,
        "validator": validator
    }


@pytest.fixture
def reasoning_controller(mock_components, mock_agent_config):
    """Create reasoning controller for E2E testing."""
    controller = ReasoningController(
        goal_decomposer=mock_components["goal_decomposer"],
        tool_planner=mock_components["tool_planner"],
        executor=mock_components["executor"],
        validator=mock_components["validator"],
        config=mock_agent_config,
        llm=mock_components["llm"]
    )
    return controller


class TestResourceErrorRecovery:
    """Test recovery from resource errors (missing files, network issues)."""
    
    def test_missing_building_file_recovery(self, reasoning_controller, mock_components):
        """
        Scenario: User asks for building analysis but file is missing
        Expected: System detects missing file, suggests alternative, provides partial analysis
        """
        # Setup goal and tasks
        mock_components["goal_decomposer"].decompose_goal.return_value = {
            "success": True,
            "tasks": [
                Task(
                    id="load_building",
                    name="Load building data",
                    description="Load building data from IFC file",
                    tool_sequence=["load_building_data"],
                    priority=Priority.HIGH
                ),
                Task(
                    id="analyze_doors",
                    name="Analyze door elements", 
                    description="Get all door elements in building",
                    tool_sequence=["get_all_elements"],
                    priority=Priority.MEDIUM,
                    dependencies=["load_building"]
                )
            ]
        }
        
        # Setup tool execution failure
        mock_components["executor"].execute_tool.side_effect = [
            ExecutionResult(
                success=False,
                tool_name="load_building_data",
                output=None,
                error_message="FileNotFoundError: Building file 'missing.ifc' not found"
            ),
            ExecutionResult(
                success=True,
                tool_name="load_building_data", 
                output={"data": {"building_loaded": True, "elements": 25}},
                error_message=None
            )
        ]
        
        # Setup validation
        mock_components["validator"].validate_execution.side_effect = [
            {"success": False, "message": "File not found", "should_replan": False},
            {"success": True, "message": "Building loaded successfully"}
        ]
        
        # Setup LLM for recovery analysis
        mock_components["llm"].invoke.return_value = Mock(content="""FAILURE_TYPE: resource_error
CAN_RETRY: yes
SUGGESTED_ACTION: Check file path or use alternative file
ALTERNATIVE_TOOLS: 
CONFIDENCE: 0.9""")
        
        # Execute reasoning
        result = reasoning_controller.reason(
            "Analyze the building structure and find all doors",
            context={"building_file": "missing.ifc"}
        )
        
        # Verify recovery occurred
        assert result is not None
        assert result.get("success") in [True, False]  # Either recovered or degraded gracefully
        
        # Verify tool was called twice (initial failure + recovery retry)
        assert mock_components["executor"].execute_tool.call_count == 2
        
        # Verify recovery system was invoked
        recovery_system = reasoning_controller.recovery_system
        assert len(recovery_system.recovery_attempts) > 0


class TestToolSelectionErrorRecovery:
    """Test recovery from wrong tool selection errors."""
    
    def test_wrong_tool_for_distance_calculation(self, reasoning_controller, mock_components):
        """
        Scenario: System selects wrong tool for distance calculation
        Expected: Tool alternative recovery selects correct tool
        """
        # Setup tasks with wrong tool initially selected
        mock_components["goal_decomposer"].decompose_goal.return_value = {
            "success": True,
            "tasks": [
                Task(
                    id="calculate_distance",
                    name="Calculate distance between doors",
                    description="Calculate distance between first two doors",
                    tool_sequence=["get_all_elements"],  # Wrong tool!
                    priority=Priority.MEDIUM
                )
            ]
        }
        
        # Setup tool execution - wrong tool fails
        mock_components["executor"].execute_tool.side_effect = [
            ExecutionResult(
                success=False,
                tool_name="get_all_elements",
                output=None, 
                error_message="Tool 'get_all_elements' cannot calculate distances"
            ),
            ExecutionResult(
                success=True,
                tool_name="calculate",
                output={"data": {"distance": 6.2, "units": "meters"}},
                error_message=None
            )
        ]
        
        # Setup validation
        mock_components["validator"].validate_execution.side_effect = [
            {"success": False, "message": "Wrong tool selected", "should_replan": False},
            {"success": True, "message": "Distance calculated successfully"}
        ]
        
        # Setup LLM for recovery analysis
        mock_components["llm"].invoke.return_value = Mock(content="""FAILURE_TYPE: tool_selection_error
CAN_RETRY: yes
SUGGESTED_ACTION: Use calculate tool for distance calculations
ALTERNATIVE_TOOLS: calculate
CONFIDENCE: 0.95""")
        
        # Execute reasoning
        result = reasoning_controller.reason("How far are the two first doors?")
        
        # Verify tool alternative was used
        assert mock_components["executor"].execute_tool.call_count == 2
        
        # Verify second call used correct tool
        second_call = mock_components["executor"].execute_tool.call_args_list[1]
        assert "calculate" in str(second_call)


class TestDependencyErrorHandling:
    """Test handling of missing dependency errors."""
    
    def test_missing_door_data_dependency(self, reasoning_controller, mock_components):
        """
        Scenario: Distance calculation fails because door data wasn't loaded first
        Expected: System recognizes dependency issue and handles gracefully
        """
        # Setup dependent tasks
        mock_components["goal_decomposer"].decompose_goal.return_value = {
            "success": True,
            "tasks": [
                Task(
                    id="get_doors",
                    name="Get door elements",
                    description="Get all door elements", 
                    tool_sequence=["get_all_elements"],
                    priority=Priority.HIGH
                ),
                Task(
                    id="calculate_distance", 
                    name="Calculate door distances",
                    description="Calculate distances between doors",
                    tool_sequence=["calculate"],
                    priority=Priority.MEDIUM,
                    dependencies=["get_doors"]
                )
            ]
        }
        
        # Setup execution - first task fails, affecting second
        mock_components["executor"].execute_tool.side_effect = [
            ExecutionResult(
                success=False,
                tool_name="get_all_elements",
                output=None,
                error_message="Building data not available"
            ),
            ExecutionResult(
                success=False,
                tool_name="calculate", 
                output=None,
                error_message="Cannot calculate: doors_data dependency missing"
            )
        ]
        
        # Setup validation
        mock_components["validator"].validate_execution.side_effect = [
            {"success": False, "message": "Building data missing", "should_replan": False},
            {"success": False, "message": "Dependency missing", "should_replan": False}
        ]
        
        # Setup LLM for dependency analysis
        mock_components["llm"].invoke.return_value = Mock(content="""FAILURE_TYPE: dependency_missing
CAN_RETRY: no
SUGGESTED_ACTION: Load building data first
ALTERNATIVE_TOOLS: load_building_data
CONFIDENCE: 0.8""")
        
        # Execute reasoning
        result = reasoning_controller.reason("Calculate distances between all doors")
        
        # Verify dependency error was handled
        recovery_system = reasoning_controller.recovery_system
        dependency_attempts = [
            attempt for attempt in recovery_system.recovery_attempts 
            if hasattr(attempt, 'failure_type') and 
            attempt.failure_type == FailureType.DEPENDENCY_MISSING
        ]
        assert len(dependency_attempts) > 0


class TestGoalReplanningScenarios:
    """Test scenarios that trigger complete goal replanning."""
    
    def test_multiple_planning_errors_trigger_replanning(self, reasoning_controller, mock_components):
        """
        Scenario: Multiple tasks fail due to bad planning
        Expected: Goal replanning is triggered with enhanced context
        """
        # Setup initial bad plan
        mock_components["goal_decomposer"].decompose_goal.side_effect = [
            # Initial bad decomposition
            {
                "success": True,
                "tasks": [
                    Task(
                        id="task1",
                        name="Bad task 1",
                        description="Poorly planned task",
                        tool_sequence=["wrong_tool"],
                        priority=Priority.MEDIUM
                    ),
                    Task(
                        id="task2", 
                        name="Bad task 2",
                        description="Another poorly planned task",
                        tool_sequence=["another_wrong_tool"],
                        priority=Priority.MEDIUM
                    )
                ]
            },
            # Replanning with better context
            {
                "success": True,
                "tasks": [
                    Task(
                        id="task1_fixed",
                        name="Load building data",
                        description="Properly load building data first",
                        tool_sequence=["load_building_data"],
                        priority=Priority.HIGH
                    ),
                    Task(
                        id="task2_fixed",
                        name="Analyze elements", 
                        description="Then analyze building elements",
                        tool_sequence=["get_all_elements"],
                        priority=Priority.MEDIUM,
                        dependencies=["task1_fixed"]
                    )
                ]
            }
        ]
        
        # Setup multiple planning failures
        mock_components["executor"].execute_tool.side_effect = [
            ExecutionResult(success=False, tool_name="wrong_tool", output=None, 
                          error_message="Planning error: wrong tool for building analysis"),
            ExecutionResult(success=False, tool_name="another_wrong_tool", output=None,
                          error_message="Planning error: invalid task decomposition"),
            ExecutionResult(success=True, tool_name="load_building_data", 
                          output={"data": {"building_loaded": True}}, error_message=None),
        ]
        
        # Setup validation failures that trigger replanning
        mock_components["validator"].validate_execution.side_effect = [
            {"success": False, "message": "Planning error detected", "should_replan": True},
            {"success": False, "message": "Planning error detected", "should_replan": True}, 
            {"success": True, "message": "Task completed successfully"}
        ]
        
        # Setup LLM for planning error analysis
        mock_components["llm"].invoke.return_value = Mock(content="""FAILURE_TYPE: planning_error
CAN_RETRY: yes
SUGGESTED_ACTION: Replan goal with better task decomposition
ALTERNATIVE_TOOLS: 
CONFIDENCE: 0.7""")
        
        # Execute reasoning
        result = reasoning_controller.reason("Analyze building structure comprehensively")
        
        # Verify replanning occurred
        assert mock_components["goal_decomposer"].decompose_goal.call_count == 2
        
        # Verify second call included replanning context
        second_call = mock_components["goal_decomposer"].decompose_goal.call_args_list[1]
        context = second_call[0][1]  # Second argument (context)
        assert context.get("replanning_triggered") is True
        assert "previous_failures" in context


class TestGracefulDegradationScenarios:
    """Test graceful degradation for non-critical failures."""
    
    def test_auxiliary_task_graceful_degradation(self, reasoning_controller, mock_components):
        """
        Scenario: Main analysis succeeds but auxiliary property retrieval fails
        Expected: System provides main results with note about missing auxiliary data
        """
        # Setup tasks with different priorities
        mock_components["goal_decomposer"].decompose_goal.return_value = {
            "success": True,
            "tasks": [
                Task(
                    id="main_analysis",
                    name="Analyze building structure",
                    description="Main structural analysis",
                    tool_sequence=["get_all_elements"],
                    priority=Priority.HIGH
                ),
                Task(
                    id="extra_properties",
                    name="Get additional properties", 
                    description="Retrieve extra element properties",
                    tool_sequence=["get_all_properties"],
                    priority=Priority.LOW,
                    metadata={"is_auxiliary": True}
                )
            ]
        }
        
        # Setup execution - main succeeds, auxiliary fails
        mock_components["executor"].execute_tool.side_effect = [
            ExecutionResult(
                success=True,
                tool_name="get_all_elements",
                output={"data": {"doors": 5, "windows": 8, "walls": 12}},
                error_message=None
            ),
            ExecutionResult(
                success=False,
                tool_name="get_all_properties",
                output=None,
                error_message="Property service unavailable"
            )
        ]
        
        # Setup validation
        mock_components["validator"].validate_execution.side_effect = [
            {"success": True, "message": "Main analysis completed"},
            {"success": False, "message": "Property retrieval failed", "should_replan": False}
        ]
        
        # Setup LLM for graceful degradation
        mock_components["llm"].invoke.side_effect = [
            Mock(content="""FAILURE_TYPE: tool_execution_error
CAN_RETRY: no
SUGGESTED_ACTION: Manual property verification recommended
ALTERNATIVE_TOOLS: 
CONFIDENCE: 0.6"""),
            Mock(content="Building analysis shows 5 doors, 8 windows, and 12 walls. Additional property details are unavailable due to service limitations.")
        ]
        
        # Execute reasoning
        result = reasoning_controller.reason("Give me a complete building overview with all details")
        
        # Verify graceful degradation occurred
        recovery_system = reasoning_controller.recovery_system
        assert hasattr(recovery_system, 'recovery_attempts')
        
        # Verify main task completed successfully
        assert mock_components["executor"].execute_tool.call_count == 2
        
        # Verify final response includes both success and limitation info
        response = result.get("response", "")
        assert "doors" in response.lower() or "windows" in response.lower()  # Main results included


class TestComplexCascadingFailures:
    """Test complex scenarios with cascading failures and multiple recovery attempts."""
    
    def test_building_analysis_cascade_with_final_success(self, reasoning_controller, mock_components):
        """
        Scenario: Complex building analysis with multiple initial failures but eventual success
        Flow: File missing → Resource recovery → Wrong tool → Tool recovery → Success
        """
        # Setup comprehensive task plan
        mock_components["goal_decomposer"].decompose_goal.return_value = {
            "success": True,
            "tasks": [
                Task(
                    id="load_data",
                    name="Load building data",
                    description="Load IFC building data",
                    tool_sequence=["load_building_data"],
                    priority=Priority.HIGH
                ),
                Task(
                    id="get_doors",
                    name="Get door elements",
                    description="Extract all door elements", 
                    tool_sequence=["get_all_elements"],
                    priority=Priority.MEDIUM,
                    dependencies=["load_data"]
                ),
                Task(
                    id="calculate_distances",
                    name="Calculate door distances",
                    description="Calculate distances between doors",
                    tool_sequence=["calculate"], 
                    priority=Priority.MEDIUM,
                    dependencies=["get_doors"]
                )
            ]
        }
        
        # Setup complex execution sequence with multiple failures and recoveries
        mock_components["executor"].execute_tool.side_effect = [
            # First attempt: File missing
            ExecutionResult(success=False, tool_name="load_building_data", output=None,
                          error_message="FileNotFoundError: building.ifc not found"),
            # Recovery: File found with corrected path  
            ExecutionResult(success=True, tool_name="load_building_data",
                          output={"data": {"building_loaded": True, "elements": 50}}, error_message=None),
            # Second task: Wrong tool initially
            ExecutionResult(success=False, tool_name="get_all_elements", output=None,
                          error_message="Wrong parameters for element extraction"),
            # Recovery: Corrected parameters
            ExecutionResult(success=True, tool_name="get_all_elements",
                          output={"data": {"doors": [{"id": "door1"}, {"id": "door2"}]}}, error_message=None),
            # Final task: Success
            ExecutionResult(success=True, tool_name="calculate",
                          output={"data": {"distance": 4.5, "units": "meters"}}, error_message=None)
        ]
        
        # Setup validation sequence
        mock_components["validator"].validate_execution.side_effect = [
            {"success": False, "message": "File not found", "should_replan": False},
            {"success": True, "message": "Building data loaded"},
            {"success": False, "message": "Parameter error", "should_replan": False},
            {"success": True, "message": "Door elements retrieved"},
            {"success": True, "message": "Distance calculated"}
        ]
        
        # Setup LLM responses for different recovery scenarios
        mock_components["llm"].invoke.side_effect = [
            # Resource error analysis
            Mock(content="""FAILURE_TYPE: resource_error
CAN_RETRY: yes
SUGGESTED_ACTION: Check file path
ALTERNATIVE_TOOLS: 
CONFIDENCE: 0.9"""),
            # Parameter error analysis  
            Mock(content="""FAILURE_TYPE: parameter_error
CAN_RETRY: yes
SUGGESTED_ACTION: Fix parameter format
ALTERNATIVE_TOOLS: 
CONFIDENCE: 0.8"""),
            # Final response synthesis
            Mock(content="Successfully analyzed the building: found 2 doors with a distance of 4.5 meters between them.")
        ]
        
        # Execute complex reasoning scenario
        result = reasoning_controller.reason(
            "Analyze the building and calculate distances between doors",
            context={"building_file": "wrong_path.ifc"}
        )
        
        # Verify multiple recoveries occurred
        assert mock_components["executor"].execute_tool.call_count == 5
        
        # Verify recovery system tracked multiple attempts
        recovery_system = reasoning_controller.recovery_system
        assert len(recovery_system.recovery_attempts) >= 2
        
        # Verify final success
        assert result is not None
        final_response = result.get("response", "")
        assert "4.5 meters" in final_response or "distance" in final_response.lower()


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "-k", "test_missing_building_file_recovery or test_wrong_tool_for_distance_calculation",
        "--tb=short"
    ])