"""
Simplified tests for the Recovery System - focusing on core functionality.
"""

import pytest
from unittest.mock import Mock

from aec_agent.core.recovery_system import RecoverySystem, FailureType
from aec_agent.core.reasoning_utils import Task, Priority, ExecutionResult
from aec_agent.core.task_graph import TaskGraph


@pytest.fixture
def mock_components():
    """Mock all required components."""
    llm = Mock()
    llm.invoke.return_value = Mock(content="""FAILURE_TYPE: tool_execution_error
CAN_RETRY: yes
SUGGESTED_ACTION: Check tool parameters and retry
ALTERNATIVE_TOOLS: get_all_elements
CONFIDENCE: 0.8""")
    
    goal_decomposer = Mock()
    goal_decomposer.decompose_goal.return_value = {
        "success": True,
        "tasks": [],
        "message": "Goal decomposed successfully"
    }
    
    tool_planner = Mock()
    tool_planner.plan_tools.return_value = {
        "success": True,
        "tool_sequence": ["calculate"],
        "metadata": {}
    }
    
    return {
        "llm": llm,
        "goal_decomposer": goal_decomposer,
        "tool_planner": tool_planner
    }


@pytest.fixture
def recovery_system(mock_components):
    """Recovery system for testing."""
    return RecoverySystem(
        goal_decomposer=mock_components["goal_decomposer"],
        tool_planner=mock_components["tool_planner"],
        llm=mock_components["llm"]
    )


@pytest.fixture
def sample_task():
    """Sample task for testing."""
    return Task(
        id="test_task",
        name="Calculate distance",
        description="Calculate distance between elements",
        tool_sequence=["calculate"],
        priority=Priority.MEDIUM
    )


@pytest.fixture  
def low_priority_task():
    """Low priority task for degradation testing."""
    return Task(
        id="low_task",
        name="Get properties", 
        description="Get additional properties",
        tool_sequence=["get_all_properties"],
        priority=Priority.LOW,
        metadata={"is_auxiliary": True}
    )


class TestFailureAnalysis:
    """Test basic failure analysis."""
    
    def test_deterministic_parameter_error(self, recovery_system, sample_task):
        """Test deterministic recognition of parameter errors."""
        error_result = ExecutionResult(
            success=False,
            tool_name="calculate",
            output=None,
            error_message="missing required parameter 'building_data'"
        )
        
        analysis = recovery_system.analyze_failure(
            task=sample_task,
            error_result=error_result,
            context={}
        )
        
        assert analysis.failure_type == FailureType.PARAMETER_ERROR
        
    def test_deterministic_dependency_error(self, recovery_system, sample_task):
        """Test deterministic recognition of dependency errors.""" 
        error_result = ExecutionResult(
            success=False,
            tool_name="calculate",
            output=None,
            error_message="dependency not available: doors_data missing"
        )
        
        analysis = recovery_system.analyze_failure(
            task=sample_task,
            error_result=error_result,
            context={}
        )
        
        assert analysis.failure_type == FailureType.DEPENDENCY_MISSING


class TestGracefulDegradation:
    """Test graceful degradation logic."""
    
    def test_can_degrade_low_priority(self, recovery_system, low_priority_task):
        """Test that low priority tasks can be degraded."""
        analysis = Mock()
        analysis.failure_type = FailureType.TOOL_EXECUTION_ERROR
        
        can_degrade = recovery_system.can_gracefully_degrade(low_priority_task, analysis)
        assert can_degrade is True
        
    def test_cannot_degrade_high_priority(self, recovery_system):
        """Test that high priority tasks cannot be easily degraded."""
        high_priority_task = Task(
            id="critical_task",
            name="Load building data",
            description="Load critical data",
            tool_sequence=["load_building_data"],
            priority=Priority.HIGH
        )
        
        analysis = Mock()
        analysis.failure_type = FailureType.TOOL_EXECUTION_ERROR
        
        can_degrade = recovery_system.can_gracefully_degrade(high_priority_task, analysis)
        assert can_degrade is False
    
    def test_create_degraded_response(self, recovery_system, sample_task):
        """Test creation of degraded responses."""
        analysis = Mock()
        analysis.failure_type = FailureType.DEPENDENCY_MISSING
        analysis.error_message = "Missing data"
        analysis.suggested_action = "Check data"
        analysis.alternative_tools = ["get_all_elements"]
        
        response = recovery_system.create_degraded_response(sample_task, analysis)
        
        assert response["success"] is False
        assert response["degraded"] is True
        assert "distance" in response["message"].lower()


class TestRecoveryStats:
    """Test recovery statistics."""
    
    def test_get_empty_stats(self, recovery_system):
        """Test statistics with no attempts."""
        stats = recovery_system.get_recovery_stats()
        
        assert stats["total_recovery_attempts"] == 0
        assert stats["successful_recoveries"] == 0
        assert stats["recovery_success_rate"] == 0
        assert stats["failure_type_breakdown"] == {}
        assert stats["replanning_attempts"] == 0


class TestReplanningLogic:
    """Test replanning decision logic."""
    
    def test_no_replanning_single_failure(self, recovery_system):
        """Test that single failures don't trigger replanning."""
        failed_tasks = [Mock()]
        should_replan = recovery_system.should_trigger_replanning(failed_tasks)
        assert should_replan is False
        
    def test_replanning_with_multiple_planning_errors(self, recovery_system):
        """Test replanning with multiple planning errors."""
        # Add planning error attempts
        for _ in range(3):
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'failure_type': FailureType.PLANNING_ERROR,
                    'success': False
                })()
            )
        
        failed_tasks = [Mock(), Mock()]
        should_replan = recovery_system.should_trigger_replanning(failed_tasks)
        assert should_replan is True


if __name__ == "__main__":
    pytest.main([__file__])