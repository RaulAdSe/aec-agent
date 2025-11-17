"""
Tests for Phase 2 Guardrails - External loop caps and counter-based guardrails.

These tests verify that the external loop caps prevent infinite execution
and that the counter-based guardrails work correctly with the reasoning controller.
"""

import pytest
from unittest.mock import Mock, patch
from aec_agent.core.llm_guardrails import GuardrailConfig, ExecutionGuardrail, GuardrailViolationError
from aec_agent.core.reasoning_controller import ReasoningController
from aec_agent.memory.execution_memory import ExecutionMemory
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus


class TestReasoningControllerGuardrails:
    """Test guardrail integration in ReasoningController."""
    
    def test_controller_with_custom_guardrail_config(self):
        """Test that ReasoningController accepts custom guardrail config."""
        config = GuardrailConfig(
            max_replanning_events=3,
            max_same_task_attempts=2,
            max_total_execution_steps=10
        )
        
        # Mock the dependencies
        goal_decomposer = Mock()
        tool_planner = Mock()
        executor = Mock()
        validator = Mock()
        
        controller = ReasoningController(
            goal_decomposer=goal_decomposer,
            tool_planner=tool_planner,
            executor=executor,
            validator=validator,
            guardrail_config=config
        )
        
        assert controller.execution_guardrail.config.max_replanning_events == 3
        assert controller.execution_guardrail.config.max_same_task_attempts == 2
        assert controller.execution_guardrail.config.max_total_execution_steps == 10
    
    def test_controller_resets_guardrails_on_new_reasoning(self):
        """Test that guardrails are reset when starting new reasoning."""
        config = GuardrailConfig(max_total_execution_steps=5)
        
        # Mock dependencies
        goal_decomposer = Mock()
        tool_planner = Mock() 
        executor = Mock()
        validator = Mock()
        
        controller = ReasoningController(
            goal_decomposer=goal_decomposer,
            tool_planner=tool_planner,
            executor=executor,
            validator=validator,
            guardrail_config=config
        )
        
        # Simulate some usage
        controller.execution_guardrail.record_execution_step()
        controller.execution_guardrail.record_execution_step()
        assert controller.execution_guardrail.total_steps == 2
        
        # Mock the methods to prevent actual reasoning
        with patch.object(controller, '_analyze_and_plan'):
            with patch.object(controller, '_execute_reasoning_loop', return_value=[]):
                with patch.object(controller, '_finalize_results', return_value={}):
                    controller.reason("Test goal")
        
        # Guardrails should be reset
        assert controller.execution_guardrail.total_steps == 0
        assert controller.execution_guardrail.replanning_count == 0
    
    def test_controller_includes_guardrail_status_in_results(self):
        """Test that controller includes guardrail status in final results."""
        config = GuardrailConfig()
        
        # Mock dependencies
        goal_decomposer = Mock()
        tool_planner = Mock()
        executor = Mock()
        validator = Mock()
        
        controller = ReasoningController(
            goal_decomposer=goal_decomposer,
            tool_planner=tool_planner,
            executor=executor,
            validator=validator,
            guardrail_config=config
        )
        
        # Mock successful execution
        with patch.object(controller, '_analyze_and_plan'):
            with patch.object(controller, '_execute_reasoning_loop', return_value=[]):
                with patch.object(controller, '_is_goal_achieved', return_value=True):
                    controller.state = Mock()
                    controller.state.tasks = []
                    controller.state.completed_tasks = 1
                    controller.state.failed_tasks = 0
                    controller.state.iteration = 1
                    controller.state.total_execution_time = 10.0
                    
                    with patch('aec_agent.core.reasoning_controller.ReasoningUtils.calculate_task_progress', return_value=100.0):
                        result = controller._finalize_results([])
        
        # Should include guardrail status
        assert "guardrails" in result["summary"]
        guardrail_status = result["summary"]["guardrails"]
        assert "replanning_events" in guardrail_status
        assert "total_steps" in guardrail_status
        assert "task_attempts" in guardrail_status


class TestExecutionMemoryGuardrails:
    """Test guardrail integration in ExecutionMemory."""
    
    def test_execution_memory_with_custom_guardrail_config(self):
        """Test ExecutionMemory with custom guardrail configuration."""
        config = GuardrailConfig(
            max_total_execution_steps=5,
            max_same_task_attempts=2
        )
        
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        assert memory.execution_guardrail.config.max_total_execution_steps == 5
        assert memory.execution_guardrail.config.max_same_task_attempts == 2
    
    def test_record_execution_step_checks_guardrails(self):
        """Test that recording execution step checks guardrails."""
        config = GuardrailConfig(max_total_execution_steps=2)
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        # First two steps should be fine
        memory.record_execution_step("task1", "task-id-1", "tool1", {})
        memory.record_execution_step("task2", "task-id-2", "tool2", {})
        
        # Third step should raise exception
        with pytest.raises(GuardrailViolationError) as exc_info:
            memory.record_execution_step("task3", "task-id-3", "tool3", {})
        
        assert "Maximum execution steps exceeded" in str(exc_info.value)
    
    def test_record_execution_step_checks_task_attempts(self):
        """Test that recording execution step checks task attempt limits."""
        config = GuardrailConfig(max_same_task_attempts=2)
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        # First two attempts for same task should be fine
        memory.record_execution_step("task1", "task-id-1", "tool1", {})
        memory.record_execution_step("task1", "task-id-1", "tool2", {})
        
        # Third attempt should raise exception
        with pytest.raises(GuardrailViolationError) as exc_info:
            memory.record_execution_step("task1", "task-id-1", "tool3", {})
        
        assert "Maximum attempts for task 'task-id-1' exceeded" in str(exc_info.value)
    
    def test_record_plan_modification_checks_replanning_limit(self):
        """Test that plan modification checks replanning guardrails."""
        config = GuardrailConfig(max_replanning_events=2)
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        # First two replanning events should be fine
        memory.record_plan_modification("failure", {}, "old", "new", "reason1")
        memory.record_plan_modification("error", {}, "old", "new", "reason2")
        
        # Third should raise exception
        with pytest.raises(GuardrailViolationError) as exc_info:
            memory.record_plan_modification("another", {}, "old", "new", "reason3")
        
        assert "Maximum replanning events exceeded" in str(exc_info.value)
    
    def test_execution_context_includes_guardrail_status(self):
        """Test that execution context includes guardrail status."""
        config = GuardrailConfig()
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        # Record some activity
        memory.record_execution_step("task1", "task-id-1", "tool1", {})
        memory.record_plan_modification("test", {}, "old", "new", "reason")
        
        context = memory.get_context_for_replanning()
        
        assert "guardrails_status" in context
        status = context["guardrails_status"]
        assert "replanning_events" in status
        assert "total_steps" in status
        assert status["replanning_events"] == "1/5"
        assert status["total_steps"] == "1/50"


class TestGuardrailScenarios:
    """Test realistic guardrail scenarios."""
    
    def test_multiple_task_failures_trigger_limit(self):
        """Test that repeated task failures eventually trigger guardrail."""
        config = GuardrailConfig(max_same_task_attempts=3)
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        # Simulate same task failing multiple times
        for i in range(3):
            memory.record_execution_step(
                f"retry_task_attempt_{i+1}",
                "failing-task-id",
                f"tool_{i+1}",
                {},
                tool_success=False,
                error_message="Tool execution failed"
            )
        
        # Fourth attempt should be blocked
        with pytest.raises(GuardrailViolationError):
            memory.record_execution_step(
                "retry_task_attempt_4",
                "failing-task-id", 
                "tool_4",
                {}
            )
    
    def test_excessive_replanning_triggers_limit(self):
        """Test that excessive replanning triggers guardrail."""
        config = GuardrailConfig(max_replanning_events=3)
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        # Simulate multiple replanning events
        triggers = ["validation_failure", "execution_error", "blocked_tasks"]
        for i, trigger in enumerate(triggers):
            memory.record_plan_modification(
                trigger,
                {"failure_count": i+1},
                f"original_plan_{i}",
                f"revised_plan_{i}",
                f"Revising plan due to {trigger}"
            )
        
        # Fourth replanning should be blocked
        with pytest.raises(GuardrailViolationError):
            memory.record_plan_modification(
                "another_failure",
                {"failure_count": 4},
                "original_plan_3", 
                "revised_plan_3",
                "Another revision attempt"
            )
    
    def test_mixed_guardrail_scenarios(self):
        """Test guardrail behavior with mixed scenarios."""
        config = GuardrailConfig(
            max_total_execution_steps=5,
            max_same_task_attempts=2,
            max_replanning_events=2
        )
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        # Record some execution steps
        memory.record_execution_step("task1", "task-id-1", "tool1", {})
        memory.record_execution_step("task2", "task-id-2", "tool2", {})
        memory.record_execution_step("task3", "task-id-3", "tool3", {})
        
        # Record a replanning event
        memory.record_plan_modification("error", {}, "old", "new", "reason")
        
        # Try to record more steps - should still work
        memory.record_execution_step("task4", "task-id-4", "tool4", {})
        memory.record_execution_step("task5", "task-id-5", "tool5", {})
        
        # Now we should hit the total steps limit
        with pytest.raises(GuardrailViolationError) as exc_info:
            memory.record_execution_step("task6", "task-id-6", "tool6", {})
        
        assert "Maximum execution steps exceeded: 6/5" in str(exc_info.value)
        
        # Also verify we can't exceed replanning limit
        memory.record_plan_modification("another", {}, "old", "new", "reason")
        
        with pytest.raises(GuardrailViolationError) as exc_info:
            memory.record_plan_modification("third", {}, "old", "new", "reason")
        
        assert "Maximum replanning events exceeded: 3/2" in str(exc_info.value)


class TestGuardrailConfigEnvironment:
    """Test guardrail configuration from environment variables."""
    
    @patch.dict('os.environ', {
        'AEC_MAX_REPLANNING': '7',
        'AEC_MAX_TASK_ATTEMPTS': '4',
        'AEC_MAX_EXECUTION_STEPS': '25'
    })
    def test_environment_configuration_applied(self):
        """Test that environment variables are properly applied."""
        memory = ExecutionMemory("test-session", "Test goal")
        
        assert memory.execution_guardrail.config.max_replanning_events == 7
        assert memory.execution_guardrail.config.max_same_task_attempts == 4
        assert memory.execution_guardrail.config.max_total_execution_steps == 25
    
    @patch.dict('os.environ', {'AEC_MAX_EXECUTION_STEPS': '3'})
    def test_environment_override_works_functionally(self):
        """Test that environment overrides work functionally."""
        memory = ExecutionMemory("test-session", "Test goal")
        
        # Should be able to record 3 steps
        for i in range(3):
            memory.record_execution_step(f"task{i+1}", f"task-id-{i+1}", f"tool{i+1}", {})
        
        # Fourth step should fail due to environment limit
        with pytest.raises(GuardrailViolationError) as exc_info:
            memory.record_execution_step("task4", "task-id-4", "tool4", {})
        
        assert "Maximum execution steps exceeded: 4/3" in str(exc_info.value)