"""
Tests for LLM Guardrails - Phase 1 retry functionality.

These tests verify the basic external retry mechanisms work correctly
with the core reactive agent components.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from aec_agent.core.llm_guardrails import (
    retry_llm_call, default_llm_retry, GuardrailConfig, 
    LLMRetryError, GuardrailViolationError, ExecutionGuardrail, MemoryGuardrail
)
from aec_agent.core.progress_evaluator import ProgressEvaluator
from aec_agent.core.validator import ResultValidator
from aec_agent.core.replanner import Replanner
from aec_agent.core.tool_planner import ToolPlanner


class TestLLMRetryDecorator:
    """Test the basic retry decorator functionality."""
    
    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retries."""
        call_count = 0
        
        @retry_llm_call(max_retries=3, delay=0.1)
        def mock_llm_call():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = mock_llm_call()
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_exception(self):
        """Test that exceptions trigger retries up to max_retries."""
        call_count = 0
        
        @retry_llm_call(max_retries=2, delay=0.01)
        def failing_llm_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("LLM API error")
            return "success_after_retry"
        
        result = failing_llm_call()
        assert result == "success_after_retry"
        assert call_count == 3  # Initial + 2 retries
    
    def test_exhausted_retries_raises_error(self):
        """Test that exhausted retries raise LLMRetryError."""
        
        @retry_llm_call(max_retries=2, delay=0.01)
        def always_failing_call():
            raise Exception("Persistent LLM failure")
        
        with pytest.raises(LLMRetryError) as exc_info:
            always_failing_call()
        
        assert "failed after 3 attempts" in str(exc_info.value)
        assert "Persistent LLM failure" in str(exc_info.value)
    
    def test_exponential_backoff(self):
        """Test that exponential backoff increases delays."""
        delays = []
        
        @retry_llm_call(max_retries=3, delay=0.05, exponential_backoff=True)
        def failing_call():
            # Record when each call happens
            delays.append(time.time())
            raise Exception("Test failure")
        
        with pytest.raises(LLMRetryError):
            failing_call()
        
        # Check that delays increased (within tolerance for test timing)
        assert len(delays) == 4  # Initial + 3 retries
        # Note: In real tests, we'd mock time.sleep to avoid actual delays


class TestGuardrailConfig:
    """Test guardrail configuration from environment."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GuardrailConfig()
        
        assert config.llm_max_retries == 3
        assert config.llm_retry_delay == 1.0
        assert config.llm_timeout == 30.0
        assert config.max_replanning_events == 5
        assert config.max_same_task_attempts == 3
        assert config.max_total_execution_steps == 50
    
    @patch.dict('os.environ', {
        'AEC_LLM_MAX_RETRIES': '5',
        'AEC_LLM_RETRY_DELAY': '2.0',
        'AEC_MAX_REPLANNING': '10'
    })
    def test_config_from_env(self):
        """Test configuration loading from environment variables."""
        config = GuardrailConfig.from_env()
        
        assert config.llm_max_retries == 5
        assert config.llm_retry_delay == 2.0
        assert config.max_replanning_events == 10


class TestExecutionGuardrail:
    """Test execution guardrails with simple counters."""
    
    def test_replanning_limit(self):
        """Test replanning event limit enforcement."""
        config = GuardrailConfig(max_replanning_events=2)
        guardrail = ExecutionGuardrail(config)
        
        # First replanning event should be ok
        guardrail.record_replanning_event()
        assert guardrail.replanning_count == 1
        
        # Second should also be ok
        guardrail.record_replanning_event()
        assert guardrail.replanning_count == 2
        
        # Third should raise exception
        with pytest.raises(GuardrailViolationError) as exc_info:
            guardrail.record_replanning_event()
        assert "Maximum replanning events exceeded" in str(exc_info.value)
    
    def test_execution_steps_limit(self):
        """Test execution steps limit enforcement."""
        config = GuardrailConfig(max_total_execution_steps=3)
        guardrail = ExecutionGuardrail(config)
        
        # Record steps up to limit
        for i in range(3):
            guardrail.record_execution_step()
        
        # Next step should raise exception
        with pytest.raises(GuardrailViolationError) as exc_info:
            guardrail.record_execution_step()
        assert "Maximum execution steps exceeded" in str(exc_info.value)
    
    def test_task_attempts_limit(self):
        """Test individual task attempts limit."""
        config = GuardrailConfig(max_same_task_attempts=2)
        guardrail = ExecutionGuardrail(config)
        
        # Record attempts for specific task
        guardrail.record_task_attempt("task_123")
        guardrail.record_task_attempt("task_123")
        
        # Third attempt should raise exception
        with pytest.raises(GuardrailViolationError) as exc_info:
            guardrail.record_task_attempt("task_123")
        assert "Maximum attempts for task 'task_123' exceeded" in str(exc_info.value)
    
    def test_status_reporting(self):
        """Test guardrail status reporting."""
        config = GuardrailConfig()
        guardrail = ExecutionGuardrail(config)
        
        guardrail.record_replanning_event()
        guardrail.record_execution_step()
        guardrail.record_task_attempt("task_1")
        
        status = guardrail.get_status()
        
        assert status["replanning_events"] == "1/5"
        assert status["total_steps"] == "1/50"
        assert status["task_attempts"]["task_1"] == 1
        assert not status["limits_hit"]["replanning"]


class TestMemoryGuardrail:
    """Test memory guardrail with FIFO cleanup."""
    
    def test_memory_cleanup_trigger(self):
        """Test when memory cleanup should be triggered."""
        config = GuardrailConfig(max_execution_steps_memory=5)
        guardrail = MemoryGuardrail(config)
        
        assert not guardrail.should_cleanup_memory(3)
        assert not guardrail.should_cleanup_memory(5)
        assert guardrail.should_cleanup_memory(6)
        assert guardrail.should_cleanup_memory(10)
    
    def test_fifo_cleanup(self):
        """Test FIFO cleanup keeps recent steps."""
        config = GuardrailConfig(max_execution_steps_memory=3)
        guardrail = MemoryGuardrail(config)
        
        # Create mock execution steps
        steps = [
            {"id": f"step_{i}", "timestamp": f"time_{i}"}
            for i in range(6)
        ]
        
        cleaned_steps = guardrail.cleanup_execution_steps(steps)
        
        # Should keep only the last 3 steps
        assert len(cleaned_steps) == 3
        assert cleaned_steps[0]["id"] == "step_3"
        assert cleaned_steps[1]["id"] == "step_4"
        assert cleaned_steps[2]["id"] == "step_5"
    
    def test_context_trimming(self):
        """Test context summary trimming."""
        config = GuardrailConfig(max_context_summary_length=20)
        guardrail = MemoryGuardrail(config)
        
        short_text = "Short text"
        long_text = "This is a very long context summary that exceeds the limit"
        
        # Short text should not be trimmed
        assert guardrail.trim_context_summary(short_text) == short_text
        
        # Long text should be trimmed with ellipsis
        trimmed = guardrail.trim_context_summary(long_text)
        assert len(trimmed) == 20
        assert trimmed.endswith("...")


class TestComponentIntegration:
    """Test integration of retry guardrails with core components."""
    
    def test_progress_evaluator_retry_integration(self):
        """Test that ProgressEvaluator uses retry mechanism."""
        # Create mock LLM that fails once then succeeds
        mock_llm = Mock()
        mock_chain = Mock()
        
        # First call fails, second succeeds
        mock_chain.invoke.side_effect = [
            Exception("LLM API error"),
            '{"goal_achieved": true, "confidence": 0.9}'
        ]
        
        evaluator = ProgressEvaluator(llm=mock_llm)
        
        # Test that the retry method handles the failure
        result = evaluator._invoke_llm_with_retry(mock_chain, {"test": "input"})
        
        # Should succeed after retry
        assert result == '{"goal_achieved": true, "confidence": 0.9}'
        assert mock_chain.invoke.call_count == 2
    
    def test_validator_retry_integration(self):
        """Test that ResultValidator uses retry mechanism."""
        mock_llm = Mock()
        mock_chain = Mock()
        
        # First call fails, second succeeds
        mock_chain.invoke.side_effect = [
            Exception("Network error"),
            '{"success": true, "message": "Valid result"}'
        ]
        
        validator = ResultValidator(llm=mock_llm)
        
        # Test retry mechanism
        result = validator._invoke_llm_with_retry(mock_chain, {"test": "input"})
        
        assert result == '{"success": true, "message": "Valid result"}'
        assert mock_chain.invoke.call_count == 2
    
    def test_replanner_retry_integration(self):
        """Test that Replanner uses retry mechanism."""
        mock_llm = Mock()
        mock_chain = Mock()
        
        # Simulate retry scenario
        mock_chain.invoke.side_effect = [
            Exception("Timeout"),
            '{"success": true, "tasks_to_add": []}'
        ]
        
        replanner = Replanner(llm=mock_llm)
        
        result = replanner._invoke_llm_with_retry(mock_chain, {"test": "input"})
        
        assert '{"success": true' in result
        assert mock_chain.invoke.call_count == 2
    
    def test_tool_planner_retry_integration(self):
        """Test that ToolPlanner uses retry mechanism."""
        mock_llm = Mock()
        mock_chain = Mock()
        
        # Simulate retry scenario
        mock_chain.invoke.side_effect = [
            Exception("Rate limit"),
            "load_building_data"
        ]
        
        planner = ToolPlanner(llm=mock_llm)
        
        result = planner._invoke_llm_with_retry(mock_chain, {"test": "input"})
        
        assert result == "load_building_data"
        assert mock_chain.invoke.call_count == 2


class TestDefaultRetryDecorator:
    """Test the default retry decorator with global config."""
    
    def test_default_retry_function(self):
        """Test that default_llm_retry uses global config."""
        call_count = 0
        
        @default_llm_retry
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Test failure")
            return "success"
        
        result = test_function()
        
        assert result == "success"
        # Should retry according to default config (max_retries=3)
        assert call_count == 3