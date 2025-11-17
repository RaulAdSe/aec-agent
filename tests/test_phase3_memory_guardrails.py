"""
Tests for Phase 3 Memory Guardrails - Simple memory caps with FIFO cleanup.

These tests verify that memory guardrails prevent memory bloat in long-running
reactive sessions through simple FIFO cleanup and context trimming.
"""

import pytest
import json
from aec_agent.core.llm_guardrails import GuardrailConfig, MemoryGuardrail
from aec_agent.memory.execution_memory import ExecutionMemory


class TestMemoryGuardrailBasics:
    """Test basic memory guardrail functionality."""
    
    def test_memory_cleanup_trigger_threshold(self):
        """Test when memory cleanup should be triggered."""
        config = GuardrailConfig(max_execution_steps_memory=5)
        guardrail = MemoryGuardrail(config)
        
        # Below threshold should not trigger cleanup
        assert not guardrail.should_cleanup_memory(3)
        assert not guardrail.should_cleanup_memory(5)
        
        # Above threshold should trigger cleanup
        assert guardrail.should_cleanup_memory(6)
        assert guardrail.should_cleanup_memory(10)
    
    def test_fifo_execution_steps_cleanup(self):
        """Test FIFO cleanup of execution steps."""
        config = GuardrailConfig(max_execution_steps_memory=3)
        guardrail = MemoryGuardrail(config)
        
        # Create mock execution steps
        steps = [
            {"id": f"step_{i}", "task": f"task_{i}", "timestamp": f"time_{i}"}
            for i in range(6)
        ]
        
        # Cleanup should keep only the last 3 steps
        cleaned = guardrail.cleanup_execution_steps(steps)
        
        assert len(cleaned) == 3
        assert cleaned[0]["id"] == "step_3"
        assert cleaned[1]["id"] == "step_4" 
        assert cleaned[2]["id"] == "step_5"
    
    def test_context_trimming_with_ellipsis(self):
        """Test context trimming with ellipsis for long content."""
        config = GuardrailConfig(max_context_summary_length=20)
        guardrail = MemoryGuardrail(config)
        
        short_context = "Short context"
        long_context = "This is a very long context summary that definitely exceeds the character limit"
        
        # Short context should not be trimmed
        assert guardrail.trim_context_summary(short_context) == short_context
        
        # Long context should be trimmed with ellipsis
        trimmed = guardrail.trim_context_summary(long_context)
        assert len(trimmed) == 20
        assert trimmed.endswith("...")
        assert trimmed.startswith("This is a very")
    
    def test_no_cleanup_when_under_limit(self):
        """Test that cleanup doesn't happen when under limits."""
        config = GuardrailConfig(max_execution_steps_memory=10)
        guardrail = MemoryGuardrail(config)
        
        # Small list should not be modified
        steps = [{"id": f"step_{i}"} for i in range(5)]
        cleaned = guardrail.cleanup_execution_steps(steps)
        
        assert len(cleaned) == 5
        assert cleaned == steps  # Should be identical


class TestExecutionMemoryIntegration:
    """Test memory guardrail integration with ExecutionMemory."""
    
    def test_execution_memory_auto_cleanup(self):
        """Test that ExecutionMemory automatically cleans up when limit reached."""
        config = GuardrailConfig(max_execution_steps_memory=3)
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        # Add execution steps up to limit
        for i in range(5):
            memory.record_execution_step(
                f"task_{i}",
                f"task-id-{i}",
                f"tool_{i}",
                {"arg": f"value_{i}"},
                tool_output=f"output_{i}"
            )
        
        # Should have triggered cleanup and kept only last 3 steps
        assert len(memory.execution_steps) == 3
        assert memory.execution_steps[0].task_name == "task_2"
        assert memory.execution_steps[1].task_name == "task_3"
        assert memory.execution_steps[2].task_name == "task_4"
    
    def test_memory_with_custom_limits(self):
        """Test ExecutionMemory with custom memory limits."""
        config = GuardrailConfig(max_execution_steps_memory=2)
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        # Add more steps than the limit
        for i in range(4):
            memory.record_execution_step(f"task_{i}", f"id_{i}", f"tool_{i}", {})
        
        # Should only keep the last 2 steps
        assert len(memory.execution_steps) == 2
        assert memory.execution_steps[0].task_name == "task_2"
        assert memory.execution_steps[1].task_name == "task_3"
    
    def test_trimmed_context_for_llm(self):
        """Test trimmed context generation for LLM consumption."""
        config = GuardrailConfig(max_context_summary_length=200)
        memory = ExecutionMemory("test-session", "Count doors in building", config)
        
        # Add some execution history
        memory.record_execution_step("Load data", "task-1", "load_building_data", {"file": "test.ifc"})
        memory.record_execution_step("Count doors", "task-2", "get_all_elements", {"type": "door"})
        
        # Get trimmed context
        trimmed_context = memory.get_trimmed_context_for_llm("replanning")
        
        # Should be valid JSON that's been trimmed
        assert isinstance(trimmed_context, str)
        assert len(trimmed_context) <= 200
        assert trimmed_context.endswith("...")
        
        # Should still be valid up to the ellipsis
        trimmed_without_ellipsis = trimmed_context[:-3]
        # Basic structure should be preserved
        assert "goal" in trimmed_without_ellipsis
    
    def test_different_context_types(self):
        """Test trimmed context for different context types."""
        config = GuardrailConfig(max_context_summary_length=100)
        memory = ExecutionMemory("test-session", "Test goal", config)
        
        memory.record_execution_step("task1", "id1", "tool1", {})
        
        # Test different context types
        replanning_context = memory.get_trimmed_context_for_llm("replanning")
        progress_context = memory.get_trimmed_context_for_llm("progress_evaluation")
        other_context = memory.get_trimmed_context_for_llm("other")
        
        # All should be strings and trimmed
        assert isinstance(replanning_context, str)
        assert isinstance(progress_context, str)
        assert isinstance(other_context, str)
        
        assert len(replanning_context) <= 100
        assert len(progress_context) <= 100
        assert len(other_context) <= 100


class TestLongRunningMemoryScenarios:
    """Test memory behavior in long-running scenarios."""
    
    def test_very_long_execution_session(self):
        """Test memory management in very long execution session."""
        config = GuardrailConfig(max_execution_steps_memory=5)
        memory = ExecutionMemory("long-session", "Complex building analysis", config)
        
        # Simulate 20 execution steps
        for i in range(20):
            memory.record_execution_step(
                f"analysis_task_{i}",
                f"task-{i}",
                f"analysis_tool_{i % 3}",  # Rotate through 3 tools
                {"iteration": i, "data": f"complex_data_{i}"},
                tool_output={"result": f"analysis_result_{i}", "count": i * 10}
            )
        
        # Should never exceed memory limit
        assert len(memory.execution_steps) == 5
        
        # Should have the last 5 steps
        assert memory.execution_steps[0].task_name == "analysis_task_15"
        assert memory.execution_steps[-1].task_name == "analysis_task_19"
        
        # Verify all steps have proper structure
        for step in memory.execution_steps:
            assert step.tool_name.startswith("analysis_tool_")
            assert isinstance(step.tool_output, dict)
            assert "result" in step.tool_output
    
    def test_memory_with_plan_modifications(self):
        """Test memory behavior with plan modifications."""
        config = GuardrailConfig(
            max_execution_steps_memory=4,
            max_replanning_events=10  # Allow many replanning events for this test
        )
        memory = ExecutionMemory("planning-session", "Adaptive planning goal", config)
        
        # Mix execution steps and plan modifications
        for i in range(8):
            # Add execution step
            memory.record_execution_step(f"step_{i}", f"id_{i}", f"tool_{i}", {})
            
            # Add plan modification every 2 steps
            if i % 2 == 1:
                memory.record_plan_modification(
                    f"trigger_{i//2}",
                    {"reason": f"adjustment_{i}"},
                    f"old_plan_{i//2}",
                    f"new_plan_{i//2}",
                    f"Modified plan due to step {i}"
                )
        
        # Execution steps should be limited by memory guardrail
        assert len(memory.execution_steps) == 4
        assert memory.execution_steps[0].task_name == "step_4"
        assert memory.execution_steps[-1].task_name == "step_7"
        
        # Plan modifications should be preserved (not subject to memory cleanup)
        assert len(memory.plan_modifications) == 4
    
    def test_context_size_with_large_data(self):
        """Test context trimming with large data structures."""
        config = GuardrailConfig(max_context_summary_length=500)
        memory = ExecutionMemory("data-session", "Process large dataset", config)
        
        # Add steps with large output data
        large_data = {"elements": [{"id": f"elem_{i}", "props": {"data": "x" * 100}} for i in range(10)]}
        
        memory.record_execution_step(
            "process_large_data",
            "big-task-1",
            "data_processor",
            {"config": "full_analysis"},
            tool_output=large_data
        )
        
        memory.record_execution_step(
            "analyze_results",
            "big-task-2", 
            "analyzer",
            {"mode": "detailed"},
            tool_output={"analysis": {"detailed_stats": {"data": "y" * 200}}}
        )
        
        # Get trimmed context
        trimmed = memory.get_trimmed_context_for_llm("replanning")
        
        # Should be trimmed to safe size
        assert len(trimmed) == 500
        assert trimmed.endswith("...")
        
        # Original context should still be available
        full_context = memory.get_context_for_replanning()
        assert len(json.dumps(full_context, default=str)) > 500


class TestMemoryEnvironmentConfiguration:
    """Test memory guardrail configuration from environment."""
    
    def test_custom_memory_limits_from_config(self):
        """Test custom memory limits through configuration."""
        config = GuardrailConfig(
            max_execution_steps_memory=2,
            max_context_summary_length=50
        )
        memory = ExecutionMemory("custom-session", "Test goal", config)
        
        # Test execution steps limit
        for i in range(4):
            memory.record_execution_step(f"task_{i}", f"id_{i}", f"tool_{i}", {})
        
        assert len(memory.execution_steps) == 2
        
        # Test context trimming
        trimmed = memory.get_trimmed_context_for_llm()
        assert len(trimmed) <= 50
        assert trimmed.endswith("...") or len(trimmed) == 50


class TestMemoryGuardrailEdgeCases:
    """Test edge cases for memory guardrails."""
    
    def test_empty_execution_steps_cleanup(self):
        """Test cleanup with empty execution steps."""
        config = GuardrailConfig(max_execution_steps_memory=5)
        guardrail = MemoryGuardrail(config)
        
        empty_steps = []
        cleaned = guardrail.cleanup_execution_steps(empty_steps)
        
        assert cleaned == []
        assert len(cleaned) == 0
    
    def test_exactly_at_limit_cleanup(self):
        """Test cleanup when exactly at the limit."""
        config = GuardrailConfig(max_execution_steps_memory=3)
        guardrail = MemoryGuardrail(config)
        
        steps = [{"id": f"step_{i}"} for i in range(3)]
        cleaned = guardrail.cleanup_execution_steps(steps)
        
        # Should not modify when exactly at limit
        assert len(cleaned) == 3
        assert cleaned == steps
    
    def test_context_trimming_edge_cases(self):
        """Test context trimming with edge cases."""
        config = GuardrailConfig(max_context_summary_length=10)
        guardrail = MemoryGuardrail(config)
        
        # Empty string
        assert guardrail.trim_context_summary("") == ""
        
        # String shorter than limit
        short = "short"
        assert guardrail.trim_context_summary(short) == "short"
        
        # String exactly at limit
        exact = "1234567890"  # 10 chars
        assert guardrail.trim_context_summary(exact) == "1234567890"
        
        # String just over limit
        over = "12345678901"  # 11 chars
        trimmed = guardrail.trim_context_summary(over)
        assert trimmed == "1234567..."  # 7 chars + "..." = 10 chars
        assert len(trimmed) == 10
    
    def test_memory_consistency_after_cleanup(self):
        """Test that memory remains consistent after cleanup operations."""
        config = GuardrailConfig(max_execution_steps_memory=3)
        memory = ExecutionMemory("consistency-test", "Test goal", config)
        
        # Add many steps
        for i in range(10):
            memory.record_execution_step(
                f"task_{i}",
                f"id_{i}",
                f"tool_{i}",
                {"step": i},
                tool_output={"result": f"output_{i}"}
            )
        
        # Verify consistency
        assert len(memory.execution_steps) == 3
        
        # All remaining steps should have valid structure
        for step in memory.execution_steps:
            assert step.task_name is not None
            assert step.task_id is not None
            assert step.tool_name is not None
            assert step.tool_arguments is not None
            
        # Context methods should still work
        context = memory.get_context_for_replanning()
        assert "recent_execution_steps" in context
        assert len(context["recent_execution_steps"]) <= 3
        
        trimmed_context = memory.get_trimmed_context_for_llm()
        assert isinstance(trimmed_context, str)
        
        # Memory should still be functional for new operations
        memory.record_execution_step("new_task", "new_id", "new_tool", {})
        assert len(memory.execution_steps) == 3  # Still at limit after cleanup