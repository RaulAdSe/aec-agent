"""
Tests for Phase 4 Monitoring - Comprehensive guardrail monitoring and configuration.

These tests verify that the monitoring system provides comprehensive visibility
into guardrail status, performance metrics, and configuration management.
"""

import pytest
import time
from unittest.mock import Mock, patch
from aec_agent.core.llm_guardrails import (
    GuardrailConfig, ExecutionGuardrail, MemoryGuardrail, GuardrailMonitor,
    monitored_llm_retry, create_guardrail_dashboard
)


class TestGuardrailMonitor:
    """Test comprehensive guardrail monitoring functionality."""
    
    def test_monitor_initialization(self):
        """Test that monitor initializes correctly with guardrails."""
        config = GuardrailConfig()
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        assert monitor.execution_guardrail is execution_guardrail
        assert monitor.memory_guardrail is memory_guardrail
        assert isinstance(monitor.start_time, float)
        assert monitor.llm_retry_stats["total_calls"] == 0
    
    def test_llm_call_recording(self):
        """Test LLM call statistics recording."""
        config = GuardrailConfig()
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        # Record successful calls
        monitor.record_llm_call(success=True, retry_attempts=0, retry_time=0.5)
        monitor.record_llm_call(success=True, retry_attempts=1, retry_time=1.2)
        
        # Record failed call
        monitor.record_llm_call(success=False, retry_attempts=3, retry_time=2.1)
        
        stats = monitor.llm_retry_stats
        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 2
        assert stats["failed_calls"] == 1
        assert stats["retry_attempts"] == 4  # 0 + 1 + 3
        assert stats["total_retry_time"] == 3.8  # 0.5 + 1.2 + 2.1
    
    def test_comprehensive_status(self):
        """Test comprehensive status reporting."""
        config = GuardrailConfig()
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        # Add some activity
        execution_guardrail.record_execution_step()
        execution_guardrail.record_replanning_event()
        monitor.record_llm_call(True, 1, 0.8)
        
        status = monitor.get_comprehensive_status()
        
        # Check structure
        assert "monitoring" in status
        assert "llm_performance" in status
        assert "execution_guardrails" in status
        assert "memory_status" in status
        assert "configuration" in status
        
        # Check monitoring info
        assert "session_uptime" in status["monitoring"]
        assert "start_time" in status["monitoring"]
        
        # Check LLM performance
        llm_perf = status["llm_performance"]
        assert llm_perf["total_calls"] == 1
        assert llm_perf["success_rate"] == 1.0
        assert llm_perf["retry_attempts"] == 1
        assert llm_perf["average_retry_time"] == 0.8
        
        # Check execution guardrails
        exec_status = status["execution_guardrails"]
        assert "replanning_events" in exec_status
        assert "total_steps" in exec_status
        
        # Check memory status
        mem_status = status["memory_status"]
        assert "max_memory_steps" in mem_status
        assert "max_context_length" in mem_status
        
        # Check configuration
        config_status = status["configuration"]
        assert "max_replanning_events" in config_status
        assert "llm_max_retries" in config_status
    
    def test_alert_generation(self):
        """Test alert generation for approaching limits."""
        config = GuardrailConfig(max_replanning_events=5, max_total_execution_steps=10)
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        # No alerts initially
        alerts = monitor.get_alerts()
        assert len(alerts) == 0
        
        # Approach replanning limit (80%+)
        for _ in range(4):  # 4/5 = 80%
            execution_guardrail.record_replanning_event()
        
        alerts = monitor.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["type"] == "replanning_limit_approaching"
        assert alerts[0]["level"] == "warning"
        assert "80.0%" in alerts[0]["message"]
        
        # Approach execution steps limit
        for _ in range(9):  # 9/10 = 90%
            execution_guardrail.record_execution_step()
        
        alerts = monitor.get_alerts()
        assert len(alerts) == 2
        step_alert = next(a for a in alerts if a["type"] == "execution_steps_approaching")
        assert step_alert["level"] == "warning"
        assert "90.0%" in step_alert["message"]
    
    def test_llm_failure_rate_alerts(self):
        """Test LLM failure rate alert generation."""
        config = GuardrailConfig()
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        # Record calls with high failure rate (3 failures out of 10 = 30%)
        for _ in range(7):
            monitor.record_llm_call(success=True)
        for _ in range(3):
            monitor.record_llm_call(success=False)
        
        alerts = monitor.get_alerts()
        failure_alerts = [a for a in alerts if a["type"] == "high_llm_failure_rate"]
        assert len(failure_alerts) == 1
        assert "30.0%" in failure_alerts[0]["message"]


class TestMonitoredLLMRetry:
    """Test enhanced LLM retry with monitoring integration."""
    
    def test_monitored_retry_success(self):
        """Test monitored retry with successful call."""
        config = GuardrailConfig()
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        @monitored_llm_retry(monitor=monitor, max_retries=2)
        def mock_llm_call():
            return "success"
        
        result = mock_llm_call()
        
        assert result == "success"
        assert monitor.llm_retry_stats["total_calls"] == 1
        assert monitor.llm_retry_stats["successful_calls"] == 1
        assert monitor.llm_retry_stats["failed_calls"] == 0
    
    def test_monitored_retry_with_retries(self):
        """Test monitored retry with retries before success."""
        config = GuardrailConfig()
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        call_count = 0
        
        @monitored_llm_retry(monitor=monitor, max_retries=2, delay=0.01)
        def mock_failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success_after_retries"
        
        result = mock_failing_then_success()
        
        assert result == "success_after_retries"
        assert monitor.llm_retry_stats["total_calls"] == 1
        assert monitor.llm_retry_stats["successful_calls"] == 1
        assert monitor.llm_retry_stats["retry_attempts"] == 2  # Failed twice, succeeded on third
    
    def test_monitored_retry_complete_failure(self):
        """Test monitored retry with complete failure."""
        config = GuardrailConfig()
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        @monitored_llm_retry(monitor=monitor, max_retries=1, delay=0.01)
        def always_failing_call():
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception):
            always_failing_call()
        
        assert monitor.llm_retry_stats["total_calls"] == 1
        assert monitor.llm_retry_stats["successful_calls"] == 0
        assert monitor.llm_retry_stats["failed_calls"] == 1
        assert monitor.llm_retry_stats["retry_attempts"] == 2  # Initial + 1 retry


class TestConfigurationDashboard:
    """Test configuration dashboard functionality."""
    
    def test_dashboard_creation(self):
        """Test that dashboard creates proper display."""
        dashboard = create_guardrail_dashboard()
        
        assert isinstance(dashboard, str)
        assert "GUARDRAIL CONFIGURATION" in dashboard
        assert "LLM Retry Settings" in dashboard
        assert "Execution Limits" in dashboard
        assert "Memory Limits" in dashboard
        assert "Environment Variables" in dashboard
        
        # Check for configuration values
        assert "Max Retries:" in dashboard
        assert "Max Replanning Events:" in dashboard
        assert "Max Memory Steps:" in dashboard
    
    @patch.dict('os.environ', {
        'AEC_LLM_MAX_RETRIES': '5',
        'AEC_MAX_REPLANNING': '8',
        'AEC_MAX_MEMORY_STEPS': '150'
    })
    def test_dashboard_with_env_overrides(self):
        """Test dashboard reflects environment variable overrides."""
        dashboard = create_guardrail_dashboard()
        
        # Should show the overridden values
        assert "Max Retries: 5" in dashboard
        assert "Max Replanning Events: 8" in dashboard 
        assert "Max Memory Steps: 150" in dashboard


class TestIntegratedMonitoring:
    """Test integrated monitoring across all components."""
    
    def test_full_monitoring_integration(self):
        """Test monitoring integration across all guardrail components."""
        config = GuardrailConfig(
            max_replanning_events=3,
            max_total_execution_steps=5,
            max_execution_steps_memory=4
        )
        
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        # Simulate realistic execution pattern
        
        # Step 1: Normal execution
        execution_guardrail.record_execution_step()
        execution_guardrail.record_task_attempt("task-1")
        monitor.record_llm_call(True, 0, 0.2)
        
        # Step 2: Task failure, retry
        execution_guardrail.record_execution_step()
        execution_guardrail.record_task_attempt("task-1")  # Second attempt
        monitor.record_llm_call(True, 1, 0.8)
        
        # Step 3: Replanning triggered
        execution_guardrail.record_replanning_event()
        monitor.record_llm_call(True, 0, 0.3)
        
        # Step 4: Continue execution
        execution_guardrail.record_execution_step()
        execution_guardrail.record_task_attempt("task-2")
        monitor.record_llm_call(True, 2, 1.1)
        
        # Check comprehensive status
        status = monitor.get_comprehensive_status()
        
        # Verify execution tracking
        exec_status = status["execution_guardrails"]
        assert exec_status["replanning_events"] == "1/3"
        assert exec_status["total_steps"] == "3/5"
        assert "task-1" in exec_status["task_attempts"]
        assert exec_status["task_attempts"]["task-1"] == 2
        
        # Verify LLM performance tracking
        llm_perf = status["llm_performance"]
        assert llm_perf["total_calls"] == 4
        assert llm_perf["success_rate"] == 1.0
        assert llm_perf["retry_attempts"] == 3  # 0 + 1 + 0 + 2
        
        # Check for alerts
        alerts = monitor.get_alerts()
        # Should have warning since we're at 3/5 = 60% execution steps (above 80% threshold)
        # Actually no, since 60% < 80%, no alerts yet
        assert len(alerts) == 0
        
        # Push to trigger alert
        execution_guardrail.record_execution_step()  # 4/5 = 80%
        execution_guardrail.record_execution_step()  # 5/5 = 100%
        
        alerts = monitor.get_alerts()
        step_alerts = [a for a in alerts if a["type"] == "execution_steps_approaching"]
        assert len(step_alerts) == 1
    
    def test_performance_metrics_accuracy(self):
        """Test that performance metrics are accurately calculated."""
        config = GuardrailConfig()
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        # Record precise LLM call patterns
        monitor.record_llm_call(True, 0, 0.1)   # Successful, no retries
        monitor.record_llm_call(True, 1, 0.3)   # Successful after 1 retry
        monitor.record_llm_call(True, 2, 0.5)   # Successful after 2 retries
        monitor.record_llm_call(False, 3, 1.0)  # Failed after 3 retries
        monitor.record_llm_call(True, 0, 0.2)   # Successful, no retries
        
        status = monitor.get_comprehensive_status()
        llm_perf = status["llm_performance"]
        
        # Verify calculations
        assert llm_perf["total_calls"] == 5
        assert llm_perf["success_rate"] == 0.8  # 4/5 = 80%
        assert llm_perf["retry_attempts"] == 6   # 0+1+2+3+0 = 6
        assert abs(llm_perf["average_retry_time"] - 0.35) < 0.01  # (0.1+0.3+0.5+1.0+0.2)/6 ≈ 0.35
    
    def test_monitoring_with_memory_operations(self):
        """Test monitoring integration with memory operations."""
        config = GuardrailConfig(max_execution_steps_memory=3)
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        # Create execution steps that would trigger memory cleanup
        steps = [{"id": f"step_{i}"} for i in range(5)]
        
        # Test cleanup
        cleaned_steps = memory_guardrail.cleanup_execution_steps(steps)
        assert len(cleaned_steps) == 3
        
        # Test context trimming
        long_context = "This is a very long context" * 100
        trimmed = memory_guardrail.trim_context_summary(long_context)
        assert len(trimmed) <= config.max_context_summary_length
        
        # Monitor should still provide accurate status
        status = monitor.get_comprehensive_status()
        
        mem_status = status["memory_status"]
        assert mem_status["max_memory_steps"] == 3
        assert mem_status["max_context_length"] == config.max_context_summary_length
        assert mem_status["context_trimming"] == "Active"


class TestEnvironmentIntegration:
    """Test complete environment variable integration."""
    
    @patch.dict('os.environ', {
        'AEC_LLM_MAX_RETRIES': '7',
        'AEC_LLM_RETRY_DELAY': '1.5',
        'AEC_LLM_TIMEOUT': '45',
        'AEC_MAX_REPLANNING': '10',
        'AEC_MAX_TASK_ATTEMPTS': '5',
        'AEC_MAX_EXECUTION_STEPS': '75',
        'AEC_MAX_MEMORY_STEPS': '150',
        'AEC_MAX_CONTEXT_LENGTH': '6000'
    })
    def test_complete_environment_configuration(self):
        """Test that all environment variables are properly applied."""
        config = GuardrailConfig.from_env()
        execution_guardrail = ExecutionGuardrail(config)
        memory_guardrail = MemoryGuardrail(config)
        monitor = GuardrailMonitor(execution_guardrail, memory_guardrail)
        
        status = monitor.get_comprehensive_status()
        config_status = status["configuration"]
        
        # Verify all environment overrides are applied
        assert config_status["llm_max_retries"] == 7
        assert config_status["llm_retry_delay"] == "1.5s"
        assert config_status["llm_timeout"] == "45.0s"
        assert config_status["max_replanning_events"] == 10
        assert config_status["max_task_attempts"] == 5
        assert config_status["max_execution_steps"] == 75
        
        mem_status = status["memory_status"]
        assert mem_status["max_memory_steps"] == 150
        assert mem_status["max_context_length"] == 6000
        
        # Dashboard should also reflect these values
        dashboard = create_guardrail_dashboard()
        assert "Max Retries: 7" in dashboard
        assert "Max Replanning Events: 10" in dashboard
        assert "Max Memory Steps: 150" in dashboard