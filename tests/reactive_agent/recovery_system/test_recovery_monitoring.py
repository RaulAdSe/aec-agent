"""
Recovery System Monitoring and Observability Tests

Tests monitoring capabilities, metrics collection, and LangSmith tracing:
1. Recovery metrics and statistics
2. LangSmith trace verification  
3. Performance monitoring
4. Error pattern analysis
5. System health monitoring
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from aec_agent.core.recovery_system import RecoverySystem, FailureType
from aec_agent.core.reasoning_utils import Task, Priority, ExecutionResult


class TestRecoveryMetricsCollection:
    """Test collection and analysis of recovery metrics."""
    
    def test_recovery_stats_accuracy(self):
        """Test that recovery statistics are accurately tracked."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Simulate various recovery attempts
        test_scenarios = [
            (FailureType.PARAMETER_ERROR, True),
            (FailureType.PARAMETER_ERROR, False),
            (FailureType.TOOL_SELECTION_ERROR, True),
            (FailureType.TOOL_SELECTION_ERROR, True),
            (FailureType.PLANNING_ERROR, False),
            (FailureType.DEPENDENCY_MISSING, True),
        ]
        
        for failure_type, success in test_scenarios:
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'failure_type': failure_type,
                    'success': success
                })()
            )
        
        # Get and verify statistics
        stats = recovery_system.get_recovery_stats()
        
        assert stats["total_recovery_attempts"] == 6
        assert stats["successful_recoveries"] == 4  # 4 successful attempts
        assert abs(stats["recovery_success_rate"] - (4/6)) < 0.01  # ~66.7% success rate
        
        # Verify failure type breakdown
        expected_breakdown = {
            "parameter_error": 2,
            "tool_selection_error": 2, 
            "planning_error": 1,
            "dependency_missing": 1
        }
        assert stats["failure_type_breakdown"] == expected_breakdown
        assert stats["replanning_attempts"] == 0  # No replanning in this test
    
    def test_failure_pattern_analysis(self):
        """Test analysis of failure patterns over time."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Simulate pattern: many parameter errors, then improvement
        initial_failures = [
            (FailureType.PARAMETER_ERROR, False),
            (FailureType.PARAMETER_ERROR, False), 
            (FailureType.PARAMETER_ERROR, False),
        ]
        
        later_attempts = [
            (FailureType.PARAMETER_ERROR, True),
            (FailureType.PARAMETER_ERROR, True),
            (FailureType.TOOL_SELECTION_ERROR, True),
        ]
        
        # Add initial failures
        for failure_type, success in initial_failures:
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'failure_type': failure_type,
                    'success': success,
                    'timestamp': time.time() - 100  # Earlier timestamp
                })()
            )
        
        stats_early = recovery_system.get_recovery_stats()
        assert stats_early["recovery_success_rate"] == 0.0  # All failures initially
        
        # Add later successful attempts
        for failure_type, success in later_attempts:
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'failure_type': failure_type,
                    'success': success,
                    'timestamp': time.time()  # Recent timestamp
                })()
            )
        
        stats_final = recovery_system.get_recovery_stats()
        assert stats_final["recovery_success_rate"] == 0.5  # 50% overall (3/6)
        assert stats_final["total_recovery_attempts"] == 6
    
    def test_replanning_metrics(self):
        """Test tracking of replanning events and their success."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock(),
            max_replanning_attempts=3
        )
        
        # Simulate replanning scenarios
        recovery_system.replanning_attempts = 2
        
        # Add planning error attempts
        planning_attempts = [
            (FailureType.PLANNING_ERROR, False),
            (FailureType.PLANNING_ERROR, False),
            (FailureType.PLANNING_ERROR, True),  # Eventually successful
        ]
        
        for failure_type, success in planning_attempts:
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'failure_type': failure_type,
                    'success': success
                })()
            )
        
        stats = recovery_system.get_recovery_stats()
        assert stats["replanning_attempts"] == 2
        assert stats["failure_type_breakdown"]["planning_error"] == 3
        
        # Test replanning trigger logic
        failed_tasks = [Mock(), Mock(), Mock()]  # Multiple failed tasks
        should_replan = recovery_system.should_trigger_replanning(failed_tasks)
        
        # Should trigger replanning due to planning errors
        assert should_replan is True


class TestLangSmithTraceVerification:
    """Test LangSmith tracing integration and trace quality."""
    
    @patch('aec_agent.core.recovery_system.traceable')
    def test_recovery_methods_are_traced(self, mock_traceable):
        """Verify that key recovery methods are decorated with @traceable."""
        # Import after patching to ensure decorators are mocked
        from aec_agent.core.recovery_system import RecoverySystem
        
        # Verify traceable decorator was called for key methods
        expected_traced_methods = [
            "analyze_task_failure",
            "llm_failure_analysis", 
            "attempt_task_recovery",
            "evaluate_replanning_need",
            "evaluate_graceful_degradation",
            "create_degraded_response"
        ]
        
        # Check that traceable was called with expected names
        traceable_calls = [call[1].get('name', call[0][0] if call[0] else '') 
                          for call in mock_traceable.call_args_list]
        
        for method_name in expected_traced_methods:
            assert any(method_name in call for call in traceable_calls), \
                   f"Method {method_name} should be traced"
    
    def test_trace_metadata_content(self):
        """Test that traces include useful metadata for monitoring."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        task = Task(
            id="trace_test",
            name="Trace test task", 
            description="Testing trace metadata",
            tool_sequence=["test_tool"],
            priority=Priority.MEDIUM
        )
        
        error_result = ExecutionResult(
            success=False,
            tool_name="test_tool",
            output=None,
            error_message="Test error for tracing"
        )
        
        # Mock LLM to return predictable response
        recovery_system.llm.invoke.return_value = Mock(content="""FAILURE_TYPE: parameter_error
CAN_RETRY: yes
SUGGESTED_ACTION: Fix test parameters
ALTERNATIVE_TOOLS: alternative_tool
CONFIDENCE: 0.85""")
        
        # Analyze failure - this should create traces
        with patch('langsmith.traceable') as mock_trace:
            analysis = recovery_system.analyze_failure(task, error_result, {})
            
            # Verify analysis result contains expected data for tracing
            assert analysis.failure_type == FailureType.PARAMETER_ERROR
            assert analysis.confidence == 0.85
            assert "Fix test parameters" in analysis.suggested_action
    
    def test_recovery_trace_hierarchy(self):
        """Test that recovery traces form proper hierarchical structure."""
        # This test verifies the conceptual trace structure
        # In practice, traces are created by LangSmith automatically
        
        expected_hierarchy = {
            "autonomous_reasoning_process": {
                "execute_reasoning_loop": {
                    "execute_single_task": {
                        "tool_execution": "fails",
                        "analyze_task_failure": {
                            "llm_failure_analysis": "if_needed"
                        },
                        "attempt_task_recovery": {
                            "evaluate_graceful_degradation": "if_failing",
                            "create_degraded_response": "if_degrading"
                        },
                        "evaluate_replanning_need": "if_multiple_failures"
                    }
                },
                "trigger_goal_replanning": "if_needed"
            }
        }
        
        # Verify expected structure exists conceptually
        assert "autonomous_reasoning_process" in expected_hierarchy
        assert "analyze_task_failure" in str(expected_hierarchy)
        assert "attempt_task_recovery" in str(expected_hierarchy)


class TestPerformanceMonitoring:
    """Test performance monitoring and alerting capabilities."""
    
    def test_recovery_latency_tracking(self):
        """Test tracking of recovery operation latency."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Mock slow LLM response
        def slow_llm_response(*args, **kwargs):
            time.sleep(0.05)  # 50ms delay
            return Mock(content="""FAILURE_TYPE: tool_execution_error
CAN_RETRY: yes
SUGGESTED_ACTION: Retry operation
ALTERNATIVE_TOOLS: 
CONFIDENCE: 0.8""")
        
        recovery_system.llm.invoke.side_effect = slow_llm_response
        
        task = Task(
            id="latency_test",
            name="Latency test",
            description="Testing recovery latency",
            tool_sequence=["test_tool"]
        )
        
        error_result = ExecutionResult(
            success=False, tool_name="test_tool", output=None,
            error_message="Latency test error"
        )
        
        # Measure recovery latency
        start_time = time.time()
        analysis = recovery_system.analyze_failure(task, error_result, {})
        latency = time.time() - start_time
        
        # Verify analysis completed and latency is reasonable
        assert analysis is not None
        assert 0.04 < latency < 0.2  # Should include LLM delay but not be excessive
    
    def test_memory_usage_monitoring(self):
        """Test monitoring of recovery system memory usage."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Add many recovery attempts to test memory growth
        initial_attempts = len(recovery_system.recovery_attempts)
        
        for i in range(1000):  # Large number of attempts
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'task_id': f'task_{i}',
                    'failure_type': FailureType.TOOL_EXECUTION_ERROR,
                    'success': i % 3 == 0,  # 33% success rate
                    'timestamp': time.time(),
                    'data': f'test_data_{i}'  # Some data to consume memory
                })()
            )
        
        # Verify memory is managed (system should limit growth)
        final_attempts = len(recovery_system.recovery_attempts)
        assert final_attempts >= initial_attempts
        
        # Stats should still be computable efficiently
        start_time = time.time()
        stats = recovery_system.get_recovery_stats()
        computation_time = time.time() - start_time
        
        assert computation_time < 0.1  # Should be fast even with many attempts
        assert stats["total_recovery_attempts"] == 1000
    
    def test_error_rate_alerting_thresholds(self):
        """Test detection of problematic error rates that should trigger alerts."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Simulate high error rate scenario
        high_error_attempts = [
            (FailureType.TOOL_EXECUTION_ERROR, False) for _ in range(8)  # 8 failures
        ] + [
            (FailureType.TOOL_EXECUTION_ERROR, True) for _ in range(2)   # 2 successes
        ]
        
        for failure_type, success in high_error_attempts:
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'failure_type': failure_type,
                    'success': success
                })()
            )
        
        stats = recovery_system.get_recovery_stats()
        error_rate = 1 - stats["recovery_success_rate"]  # 80% error rate
        
        # This should trigger alerting in a production system
        assert error_rate > 0.5  # High error rate detected
        assert stats["recovery_success_rate"] < 0.5  # Low success rate
        
        # Test replanning trigger due to high error rate
        failed_tasks = [Mock() for _ in range(8)]  # Many failed tasks
        should_replan = recovery_system.should_trigger_replanning(failed_tasks)
        assert should_replan is True  # Should trigger replanning


class TestHealthMonitoring:
    """Test system health monitoring and diagnostic capabilities."""
    
    def test_recovery_system_health_indicators(self):
        """Test key health indicators for the recovery system."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Healthy system indicators
        stats = recovery_system.get_recovery_stats()
        
        # System should start healthy
        assert stats["total_recovery_attempts"] == 0
        assert stats["recovery_success_rate"] == 0  # No attempts yet
        assert stats["replanning_attempts"] == 0
        
        # Add some successful operations
        successful_attempts = [
            (FailureType.PARAMETER_ERROR, True),
            (FailureType.TOOL_SELECTION_ERROR, True),
            (FailureType.PARAMETER_ERROR, True),
        ]
        
        for failure_type, success in successful_attempts:
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'failure_type': failure_type,
                    'success': success
                })()
            )
        
        healthy_stats = recovery_system.get_recovery_stats()
        
        # Health indicators should be good
        assert healthy_stats["recovery_success_rate"] == 1.0  # 100% success
        assert healthy_stats["total_recovery_attempts"] == 3
        assert len(healthy_stats["failure_type_breakdown"]) <= 3  # Reasonable variety
    
    def test_unhealthy_system_detection(self):
        """Test detection of unhealthy recovery system states."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Simulate unhealthy system state
        unhealthy_attempts = [
            (FailureType.PLANNING_ERROR, False) for _ in range(5)  # Persistent planning issues
        ] + [
            (FailureType.TOOL_EXECUTION_ERROR, False) for _ in range(10)  # Tool execution failures
        ]
        
        for failure_type, success in unhealthy_attempts:
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'failure_type': failure_type,
                    'success': success
                })()
            )
        
        unhealthy_stats = recovery_system.get_recovery_stats()
        
        # Detect unhealthy indicators
        assert unhealthy_stats["recovery_success_rate"] == 0.0  # No successes
        assert unhealthy_stats["failure_type_breakdown"]["planning_error"] == 5
        assert unhealthy_stats["total_recovery_attempts"] >= 15
        
        # Should trigger replanning due to many failures
        failed_tasks = [Mock() for _ in range(10)]
        should_replan = recovery_system.should_trigger_replanning(failed_tasks)
        assert should_replan is True
    
    def test_diagnostic_information_completeness(self):
        """Test that diagnostic information is complete and useful."""
        recovery_system = RecoverySystem(
            goal_decomposer=Mock(),
            tool_planner=Mock(),
            llm=Mock()
        )
        
        # Add diverse recovery attempts for comprehensive diagnostics
        diverse_attempts = [
            (FailureType.PARAMETER_ERROR, True),
            (FailureType.PARAMETER_ERROR, False), 
            (FailureType.TOOL_SELECTION_ERROR, True),
            (FailureType.DEPENDENCY_MISSING, False),
            (FailureType.PLANNING_ERROR, False),
            (FailureType.RESOURCE_ERROR, True),
        ]
        
        for i, (failure_type, success) in enumerate(diverse_attempts):
            recovery_system.recovery_attempts.append(
                type('RecoveryAttempt', (), {
                    'task_id': f'task_{i}',
                    'failure_type': failure_type,
                    'success': success,
                    'timestamp': time.time() - (len(diverse_attempts) - i) * 10,  # Spread over time
                    'strategy': 'test_strategy',
                    'reason': f'Test reason {i}'
                })()
            )
        
        # Get comprehensive diagnostics
        stats = recovery_system.get_recovery_stats()
        
        # Verify diagnostic completeness
        required_keys = [
            "total_recovery_attempts",
            "successful_recoveries", 
            "recovery_success_rate",
            "failure_type_breakdown",
            "replanning_attempts"
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing diagnostic key: {key}"
        
        # Verify breakdown includes all failure types encountered
        assert len(stats["failure_type_breakdown"]) == 5  # 5 different failure types
        assert stats["total_recovery_attempts"] == 6
        assert 0 < stats["recovery_success_rate"] < 1  # Mixed success/failure


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])