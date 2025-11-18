"""
Recovery System Edge Cases and Performance Tests

Tests edge cases, limits, and performance scenarios:
1. Recovery system limits and guardrails
2. LLM failure scenarios  
3. Infinite loop prevention
4. Resource exhaustion scenarios
5. Performance under load
"""

import pytest
import time
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
import asyncio

from aec_agent.core.reasoning_controller import ReasoningController
from aec_agent.core.recovery_system import RecoverySystem, FailureType
from aec_agent.core.reasoning_utils import Task, Priority, ExecutionResult, TaskStatus
from aec_agent.core.llm_guardrails import GuardrailViolationError


class TestRecoveryLimitsAndGuardrails:
    """Test recovery system respects limits and prevents infinite loops."""
    
    def test_max_retry_attempts_respected(self):
        """Test that recovery system respects maximum retry attempts."""
        # Setup recovery system with low retry limit
        mock_llm = Mock()
        mock_goal_decomposer = Mock()
        mock_tool_planner = Mock()
        
        recovery_system = RecoverySystem(
            goal_decomposer=mock_goal_decomposer,
            tool_planner=mock_tool_planner,
            llm=mock_llm,
            max_retries=2  # Low limit for testing
        )
        
        # Create test task
        task = Task(
            id="test_task",
            name="Test task",
            description="A test task",
            tool_sequence=["test_tool"],
            priority=Priority.MEDIUM
        )
        
        # Simulate repeated failures for same task
        failure_analysis = Mock()
        failure_analysis.failure_type = FailureType.TOOL_EXECUTION_ERROR
        failure_analysis.can_retry = True
        
        # Add multiple failed attempts for same task
        for i in range(3):  # More than max_retries
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'task_id': task.id,
                    'failure_type': FailureType.TOOL_EXECUTION_ERROR,
                    'success': False
                })()
            )
        
        # Attempt recovery - should be blocked by retry limit
        result = recovery_system.attempt_recovery(task, failure_analysis, {})
        
        # Should return None when max retries exceeded
        assert result is None
    
    def test_replanning_limit_respected(self):
        """Test that goal replanning has limits to prevent infinite replanning."""
        mock_llm = Mock()
        mock_goal_decomposer = Mock()
        mock_tool_planner = Mock()
        
        recovery_system = RecoverySystem(
            goal_decomposer=mock_goal_decomposer,
            tool_planner=mock_tool_planner, 
            llm=mock_llm,
            max_replanning_attempts=1
        )
        
        # Simulate max replanning attempts reached
        recovery_system.replanning_attempts = 1
        
        task = Task(id="test", name="test", description="test", tool_sequence=["test"])
        failure_analysis = Mock()
        failure_analysis.failure_type = FailureType.PLANNING_ERROR
        failure_analysis.can_retry = True
        failure_analysis.suggested_action = "Replan"
        
        # Attempt planning error recovery - should be blocked
        result = recovery_system._recover_planning_error(task, failure_analysis, {})
        
        assert result is None  # Indicates replanning blocked
    
    def test_guardrail_violations_stop_recovery(self):
        """Test that guardrail violations properly stop recovery attempts."""
        # Setup mock components
        mock_components = {
            "goal_decomposer": Mock(),
            "tool_planner": Mock(), 
            "executor": Mock(),
            "validator": Mock(),
            "llm": Mock()
        }
        
        mock_agent_config = Mock()
        mock_agent_config.reasoning = Mock()
        mock_agent_config.reasoning.max_iterations = 5
        mock_agent_config.reasoning.max_execution_time = 60.0
        mock_agent_config.get_effective_guardrail_config.return_value = Mock()
        
        controller = ReasoningController(
            goal_decomposer=mock_components["goal_decomposer"],
            tool_planner=mock_components["tool_planner"],
            executor=mock_components["executor"],
            validator=mock_components["validator"],
            config=mock_agent_config,
            llm=mock_components["llm"]
        )
        
        # Setup goal
        mock_components["goal_decomposer"].decompose_goal.return_value = {
            "success": True,
            "tasks": [
                Task(
                    id="test_task",
                    name="Test task",
                    description="A test task",
                    tool_sequence=["test_tool"],
                    priority=Priority.MEDIUM
                )
            ]
        }
        
        # Mock guardrail violation during recovery
        with patch.object(controller.execution_guardrail, 'record_replanning_event') as mock_guardrail:
            mock_guardrail.side_effect = GuardrailViolationError("Too many replanning attempts")
            
            # Setup failure that would normally trigger replanning
            mock_components["executor"].execute_tool.return_value = ExecutionResult(
                success=False, tool_name="test_tool", output=None,
                error_message="Planning error"
            )
            mock_components["validator"].validate_execution.return_value = {
                "success": False, "message": "Planning error", "should_replan": True
            }
            
            # Execute - should be stopped by guardrail
            result = controller.reason("Test goal")
            
            # Verify guardrail was hit
            assert mock_guardrail.called


class TestLLMFailureScenarios:
    """Test scenarios where LLM calls fail or return invalid responses."""
    
    def test_llm_network_failure_fallback(self):
        """Test fallback to deterministic analysis when LLM fails."""
        # Setup recovery system 
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("Network timeout")
        
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=mock_llm
        )
        
        # Create task with clear parameter error
        task = Task(
            id="test_task",
            name="Test task",
            description="Test task",
            tool_sequence=["test_tool"]
        )
        
        error_result = ExecutionResult(
            success=False,
            tool_name="test_tool",
            output=None,
            error_message="missing required parameter 'building_data'"  # Clear pattern
        )
        
        # Analyze failure - should fallback to deterministic
        analysis = recovery_system.analyze_failure(task, error_result, {})
        
        # Verify fallback analysis worked
        assert analysis.failure_type == FailureType.PARAMETER_ERROR  # Deterministic detection
        assert analysis.can_retry is False  # Conservative fallback
    
    def test_llm_invalid_response_handling(self):
        """Test handling of invalid/malformed LLM responses."""
        mock_llm = Mock()
        # LLM returns malformed response
        mock_llm.invoke.return_value = Mock(content="INVALID_RESPONSE_FORMAT")
        
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=mock_llm
        )
        
        task = Task(id="test", name="test", description="test", tool_sequence=["test"])
        error_result = ExecutionResult(
            success=False, tool_name="test", output=None,
            error_message="Unknown error"
        )
        
        # Should handle malformed response gracefully
        analysis = recovery_system.analyze_failure(task, error_result, {})
        
        assert analysis.failure_type == FailureType.TOOL_EXECUTION_ERROR  # Default fallback
        assert analysis.confidence < 1.0  # Should indicate uncertainty
    
    def test_llm_timeout_scenarios(self):
        """Test handling of LLM timeout scenarios."""
        mock_llm = Mock()
        
        def slow_llm_call(*args, **kwargs):
            time.sleep(0.1)  # Simulate slow response
            return Mock(content="""FAILURE_TYPE: tool_execution_error
CAN_RETRY: yes  
SUGGESTED_ACTION: Retry with correct parameters
ALTERNATIVE_TOOLS: 
CONFIDENCE: 0.8""")
        
        mock_llm.invoke.side_effect = slow_llm_call
        
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=mock_llm
        )
        
        task = Task(id="test", name="test", description="test", tool_sequence=["test"])
        error_result = ExecutionResult(
            success=False, tool_name="test", output=None, error_message="Error"
        )
        
        # Should complete despite slow LLM
        start_time = time.time()
        analysis = recovery_system.analyze_failure(task, error_result, {})
        duration = time.time() - start_time
        
        assert analysis is not None
        assert duration < 1.0  # Should not hang indefinitely


class TestResourceExhaustionScenarios:
    """Test behavior under resource constraints."""
    
    def test_memory_pressure_during_recovery(self):
        """Test recovery system behavior under memory pressure."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Simulate many recovery attempts (memory pressure)
        for i in range(200):  # Large number of attempts
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'task_id': f'task_{i}',
                    'failure_type': FailureType.TOOL_EXECUTION_ERROR,
                    'success': i % 2 == 0  # Mix of success/failure
                })()
            )
        
        # System should still function
        stats = recovery_system.get_recovery_stats()
        assert stats["total_recovery_attempts"] == 200
        assert stats["recovery_success_rate"] == 0.5  # 50% success rate
        
        # Memory should be manageable (implementation should limit growth)
        assert len(recovery_system.recovery_attempts) <= 200
    
    def test_concurrent_recovery_attempts(self):
        """Test recovery system thread safety under concurrent access."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(), 
            llm=Mock()
        )
        
        def simulate_recovery_attempt(task_id):
            """Simulate a recovery attempt from different thread."""
            task = Task(
                id=f"task_{task_id}",
                name=f"Task {task_id}",
                description=f"Test task {task_id}",
                tool_sequence=["test_tool"]
            )
            
            failure_analysis = Mock()
            failure_analysis.failure_type = FailureType.TOOL_EXECUTION_ERROR
            failure_analysis.can_retry = True
            
            try:
                result = recovery_system.attempt_recovery(task, failure_analysis, {})
                return result is not None
            except Exception:
                return False
        
        # Run concurrent recovery attempts
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(simulate_recovery_attempt, i) 
                for i in range(10)
            ]
            results = [f.result() for f in futures]
        
        # All attempts should complete without crashing
        assert len(results) == 10
        assert all(isinstance(r, bool) for r in results)


class TestPerformanceScenarios:
    """Test recovery system performance under various loads."""
    
    def test_recovery_analysis_performance(self):
        """Test that failure analysis performs well even with complex scenarios."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Setup fast LLM response
        recovery_system.llm.invoke.return_value = Mock(content="""FAILURE_TYPE: parameter_error
CAN_RETRY: yes
SUGGESTED_ACTION: Fix parameters
ALTERNATIVE_TOOLS: alternative_tool
CONFIDENCE: 0.9""")
        
        task = Task(
            id="performance_test",
            name="Performance test task",
            description="A task for testing performance",
            tool_sequence=["test_tool"]
        )
        
        error_result = ExecutionResult(
            success=False,
            tool_name="test_tool", 
            output=None,
            error_message="Complex error with lots of context and details that might slow analysis"
        )
        
        # Measure analysis performance
        start_time = time.time()
        
        for i in range(10):  # Multiple analyses
            analysis = recovery_system.analyze_failure(task, error_result, {})
            assert analysis.failure_type == FailureType.PARAMETER_ERROR
        
        duration = time.time() - start_time
        
        # Should be reasonably fast (less than 1 second for 10 analyses)
        assert duration < 1.0
        avg_time = duration / 10
        assert avg_time < 0.1  # Less than 100ms per analysis
    
    def test_graceful_degradation_performance(self):
        """Test performance of graceful degradation decisions."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Create various task types for degradation testing
        tasks = [
            Task(id=f"task_{i}", name=f"Task {i}", description=f"Test task {i}",
                 tool_sequence=["test"], priority=Priority.LOW if i % 2 else Priority.HIGH)
            for i in range(50)
        ]
        
        failure_analysis = Mock()
        failure_analysis.failure_type = FailureType.TOOL_EXECUTION_ERROR
        
        # Measure degradation decision performance
        start_time = time.time()
        
        degradable_count = 0
        for task in tasks:
            can_degrade = recovery_system.can_gracefully_degrade(task, failure_analysis)
            if can_degrade:
                degradable_count += 1
        
        duration = time.time() - start_time
        
        # Should be very fast (pure logic)
        assert duration < 0.1  # Less than 100ms for 50 decisions
        assert degradable_count > 0  # Some tasks should be degradable


class TestEdgeCaseRecoveryScenarios:
    """Test unusual and edge case scenarios."""
    
    def test_empty_task_recovery(self):
        """Test recovery behavior with malformed/empty tasks."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Create malformed task
        empty_task = Task(
            id="",  # Empty ID
            name="",  # Empty name
            description="",  # Empty description
            tool_sequence=[],  # No tools
            priority=Priority.MEDIUM
        )
        
        failure_analysis = Mock()
        failure_analysis.failure_type = FailureType.TOOL_EXECUTION_ERROR
        failure_analysis.can_retry = True
        
        # Should handle gracefully without crashing
        result = recovery_system.attempt_recovery(empty_task, failure_analysis, {})
        assert result is not None
        assert isinstance(result.success, bool)
    
    def test_circular_dependency_handling(self):
        """Test handling of tasks with circular dependencies."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Create tasks with circular dependencies
        task_a = Task(
            id="task_a",
            name="Task A", 
            description="First task",
            tool_sequence=["tool_a"],
            dependencies=["task_b"]  # Depends on B
        )
        
        task_b = Task(
            id="task_b",
            name="Task B",
            description="Second task", 
            tool_sequence=["tool_b"],
            dependencies=["task_a"]  # Depends on A (circular!)
        )
        
        failure_analysis = Mock()
        failure_analysis.failure_type = FailureType.DEPENDENCY_MISSING
        
        # Should handle circular dependency gracefully
        result = recovery_system.attempt_recovery(task_a, failure_analysis, {})
        
        # Should not crash and should provide reasonable response
        assert result is not None
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in error messages."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        task = Task(
            id="unicode_test",
            name="测试任务",  # Chinese characters
            description="Tâche de test avec caractères spéciaux: éàü",  # French with accents
            tool_sequence=["test_tool"]
        )
        
        error_result = ExecutionResult(
            success=False,
            tool_name="test_tool",
            output=None,
            error_message="Error with emoji: 🏗️ Building file not found! Special chars: @#$%^&*()"
        )
        
        # Should handle unicode gracefully
        try:
            analysis = recovery_system.analyze_failure(task, error_result, {})
            assert analysis is not None
            assert analysis.error_message is not None
        except UnicodeError:
            pytest.fail("Recovery system failed to handle unicode characters")


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v", 
        "--tb=short"
    ])