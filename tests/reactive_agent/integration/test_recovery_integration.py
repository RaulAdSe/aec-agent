"""
Integration tests for Recovery System with Reasoning Controller.
"""

import pytest
from unittest.mock import Mock, patch

from aec_agent.core.reasoning_controller import ReasoningController
from aec_agent.core.recovery_system import RecoverySystem
from aec_agent.core.reasoning_utils import Task, Priority, ExecutionResult
from aec_agent.core.task_graph import TaskGraph


@pytest.fixture
def mock_components():
    """Mock all components needed for integration testing."""
    # Mock LLM
    llm = Mock()
    llm.invoke.return_value = Mock(content="Test response")
    
    # Mock Goal Decomposer
    goal_decomposer = Mock()
    goal_decomposer.decompose_goal.return_value = {
        "success": True,
        "tasks": [
            Task(
                id="test_task",
                name="Test task",
                description="A test task",
                tool_sequence=["calculate"],
                priority=Priority.MEDIUM
            )
        ],
        "message": "Goal decomposed"
    }
    
    # Mock Tool Planner
    tool_planner = Mock()
    tool_planner.plan_tools.return_value = {
        "success": True,
        "tool_sequence": ["calculate"],
        "metadata": {}
    }
    
    # Mock Executor
    executor = Mock()
    executor.tool_registry = {"calculate": Mock(), "get_all_elements": Mock()}
    
    # Mock Validator
    validator = Mock()
    validator.validate_execution.return_value = {
        "success": False,
        "message": "Task failed due to parameter error",
        "should_replan": False
    }
    
    return {
        "llm": llm,
        "goal_decomposer": goal_decomposer,
        "tool_planner": tool_planner,
        "executor": executor,
        "validator": validator
    }


@pytest.fixture
def reasoning_controller(mock_components):
    """Reasoning controller with recovery system for testing."""
    controller = ReasoningController(
        goal_decomposer=mock_components["goal_decomposer"],
        tool_planner=mock_components["tool_planner"],
        executor=mock_components["executor"],
        validator=mock_components["validator"],
        llm=mock_components["llm"]
    )
    return controller


class TestRecoveryIntegration:
    """Test recovery system integration with reasoning controller."""
    
    def test_tool_failure_recovery_flow(self, reasoning_controller, mock_components):
        """Test the complete flow when a tool fails and recovery is attempted."""
        # Setup: Mock tool execution failure
        mock_components["executor"].execute_tool.return_value = ExecutionResult(
            success=False,
            tool_name="calculate",
            output=None,
            error_message="missing required parameter 'building_data'"
        )
        
        # Mock recovery system to return successful recovery
        with patch.object(reasoning_controller.recovery_system, 'analyze_failure') as mock_analyze:
            with patch.object(reasoning_controller.recovery_system, 'attempt_recovery') as mock_recover:
                
                # Setup recovery mocks
                mock_analyze.return_value = Mock(
                    failure_type=Mock(value="parameter_error"),
                    can_retry=True,
                    suggested_action="Fix parameters",
                    alternative_tools=[],
                    confidence=0.8
                )
                
                mock_recover.return_value = type('RecoveryAttempt', (), {
                    'success': True,
                    'strategy': 'fix_parameters',
                    'modified_task': None,
                    'reason': 'Parameter error fixed',
                    'get': lambda self, key, default=None: default
                })()
                
                # Execute reasoning
                result = reasoning_controller.reason("Calculate distance between doors")
                
                # Verify recovery was attempted
                assert mock_analyze.called
                assert mock_recover.called
                assert result is not None
    
    def test_graceful_degradation_flow(self, reasoning_controller, mock_components):
        """Test graceful degradation when recovery fails."""
        # Setup: Mock tool execution failure
        mock_components["executor"].execute_tool.return_value = ExecutionResult(
            success=False,
            tool_name="get_all_properties",
            output=None,
            error_message="Tool execution failed"
        )
        
        # Create a low-priority task that can be degraded
        low_priority_task = Task(
            id="degradable_task",
            name="Get additional properties",
            description="Get extra element properties",
            tool_sequence=["get_all_properties"],
            priority=Priority.LOW,
            metadata={"is_auxiliary": True}
        )
        
        mock_components["goal_decomposer"].decompose_goal.return_value = {
            "success": True,
            "tasks": [low_priority_task],
            "message": "Goal decomposed"
        }
        
        with patch.object(reasoning_controller.recovery_system, 'analyze_failure') as mock_analyze:
            with patch.object(reasoning_controller.recovery_system, 'attempt_recovery') as mock_recover:
                with patch.object(reasoning_controller.recovery_system, 'can_gracefully_degrade') as mock_can_degrade:
                    with patch.object(reasoning_controller.recovery_system, 'create_degraded_response') as mock_create_degraded:
                        
                        # Setup mocks for failed recovery but successful degradation
                        mock_analyze.return_value = Mock(
                            failure_type=Mock(value="tool_execution_error"),
                            can_retry=False
                        )
                        
                        mock_recover.return_value = type('RecoveryAttempt', (), {
                            'success': False,
                            'reason': 'Recovery failed'
                        })()
                        
                        mock_can_degrade.return_value = True
                        mock_create_degraded.return_value = {
                            "success": False,
                            "degraded": True,
                            "message": "Partial results available",
                            "partial_data": {}
                        }
                        
                        # Execute reasoning
                        result = reasoning_controller.reason("Get building properties")
                        
                        # Verify degradation was attempted
                        assert mock_can_degrade.called
                        assert mock_create_degraded.called
                        assert result is not None
    
    def test_planning_error_replanning(self, reasoning_controller, mock_components):
        """Test goal replanning when planning errors occur."""
        # Setup: Mock planning error
        mock_components["executor"].execute_tool.return_value = ExecutionResult(
            success=False,
            tool_name="calculate",
            output=None,
            error_message="Planning error: invalid task decomposition"
        )
        
        with patch.object(reasoning_controller.recovery_system, 'analyze_failure') as mock_analyze:
            with patch.object(reasoning_controller.recovery_system, 'attempt_recovery') as mock_recover:
                with patch.object(reasoning_controller, '_trigger_goal_replanning') as mock_replan:
                    
                    # Setup mocks for planning error that requires replanning
                    mock_analyze.return_value = Mock(
                        failure_type=Mock(value="planning_error"),
                        can_retry=True
                    )
                    
                    mock_recover.return_value = type('RecoveryAttempt', (), {
                        'success': True,
                        'strategy': 'replan',
                        'get': lambda self, key, default=None: {
                            'requires_goal_replanning': True,
                            'failed_task_context': {}
                        }.get(key, default)
                    })()
                    
                    mock_replan.return_value = True
                    
                    # Execute reasoning
                    result = reasoning_controller.reason("Complex building analysis")
                    
                    # Verify replanning was triggered
                    assert mock_replan.called
                    assert result is not None


class TestRecoverySystemConfiguration:
    """Test recovery system configuration and limits."""
    
    def test_recovery_system_initialization(self, reasoning_controller):
        """Test that recovery system is properly initialized."""
        assert hasattr(reasoning_controller, 'recovery_system')
        assert isinstance(reasoning_controller.recovery_system, RecoverySystem)
        
        # Check default configuration
        recovery_system = reasoning_controller.recovery_system
        assert recovery_system.max_retries == 2
        assert recovery_system.max_replanning_attempts == 1
        assert recovery_system.recovery_attempts == []
        assert recovery_system.replanning_attempts == 0
    
    def test_recovery_stats_tracking(self, reasoning_controller):
        """Test that recovery statistics are properly tracked.""" 
        recovery_system = reasoning_controller.recovery_system
        
        # Initially empty
        stats = recovery_system.get_recovery_stats()
        assert stats["total_recovery_attempts"] == 0
        assert stats["successful_recoveries"] == 0
        assert stats["recovery_success_rate"] == 0


if __name__ == "__main__":
    pytest.main([__file__])