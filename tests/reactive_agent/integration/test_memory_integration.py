"""
Integration tests for ExecutionMemory with basic memory components - Phase 1 reactive implementation.
"""

import pytest
import os
from typing import Dict, Any

from aec_agent.memory.memory_manager import MemoryManager, MemoryManagerConfig
from aec_agent.memory.execution_memory import ExecutionMemory, ValidationLevel
from aec_agent.memory.short_term_memory import ShortTermMemoryConfig
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus, ExecutionResult
from tests.reactive_agent.fixtures.reactive_fixtures import (
    test_config, memory_manager, sample_tasks, successful_execution_result, 
    failed_execution_result, create_test_task, create_test_execution_result
)


class TestExecutionMemoryIntegration:
    """Test ExecutionMemory integration with basic memory components."""
    
    def test_execution_memory_standalone_usage(self):
        """Test ExecutionMemory can be used standalone."""
        execution_memory = ExecutionMemory("session-123", "Test goal")
        
        # Test basic workflow
        execution_memory.start_iteration(1)
        
        step_id = execution_memory.record_execution_step(
            task_name="Load data",
            task_id="task-1",
            tool_name="load_tool",
            tool_arguments={"file": "test.json"},
            tool_success=True,
            tool_output={"result": "success"}
        )
        
        execution_memory.add_validation_result(step_id, {
            "success": True,
            "confidence": 0.9,
            "validation_level": "execution",
            "method": "basic_check",
            "issues": []
        })
        
        context_id = execution_memory.discover_context(
            "building_info", "loaded", True, step_id
        )
        
        # Verify all components work together
        assert len(execution_memory.execution_steps) == 1
        assert len(execution_memory.discovered_context) == 1
        
        current_context = execution_memory.get_current_context()
        assert current_context["goal"] == "Test goal"
        assert current_context["total_execution_steps"] == 1
        
        summary = execution_memory.get_execution_summary()
        assert summary["successful_steps"] == 1
        assert summary["failed_steps"] == 0
    
    def test_execution_memory_with_memory_manager(self, memory_manager):
        """Test ExecutionMemory works alongside MemoryManager."""
        # Create execution memory separately
        execution_memory = ExecutionMemory("session-456", "Integration test")
        
        # Test that it works independently of memory_manager
        execution_memory.start_iteration(1)
        
        step_id = execution_memory.record_execution_step(
            task_name="Integration task",
            task_id="task-1", 
            tool_name="integration_tool",
            tool_arguments={"param": "value"},
            tool_success=True
        )
        
        # Verify memory_manager is not affected
        assert memory_manager.session is not None
        assert memory_manager.short_term is not None
        
        # Verify execution_memory works correctly
        assert len(execution_memory.execution_steps) == 1
        step = execution_memory.execution_steps[0]
        assert step.task_name == "Integration task"
        assert step.tool_name == "integration_tool"
    
    def test_multiple_execution_memories(self):
        """Test multiple ExecutionMemory instances can coexist."""
        memory1 = ExecutionMemory("session-1", "Goal 1")
        memory2 = ExecutionMemory("session-2", "Goal 2") 
        
        # Add data to both
        memory1.start_iteration(1)
        memory2.start_iteration(1)
        
        step1_id = memory1.record_execution_step(
            task_name="Task 1",
            task_id="task-1",
            tool_name="tool1",
            tool_arguments={},
            tool_success=True
        )
        
        step2_id = memory2.record_execution_step(
            task_name="Task 2", 
            task_id="task-2",
            tool_name="tool2",
            tool_arguments={},
            tool_success=False,
            error_message="Failed"
        )
        
        # Verify they are independent
        assert memory1.session_id != memory2.session_id
        assert memory1.goal != memory2.goal
        assert len(memory1.execution_steps) == 1
        assert len(memory2.execution_steps) == 1
        
        # Verify different states
        assert memory1.execution_steps[0].tool_success is True
        assert memory2.execution_steps[0].tool_success is False
    
    def test_execution_memory_lifecycle(self):
        """Test ExecutionMemory lifecycle operations."""
        memory = ExecutionMemory("lifecycle-test", "Lifecycle goal")
        
        # Phase 1: Initial execution
        memory.start_iteration(1)
        step1_id = memory.record_execution_step(
            "Task 1", "task-1", "tool1", {}, True
        )
        memory.add_validation_result(step1_id, {
            "success": True,
            "confidence": 0.8,
            "validation_level": "execution",
            "method": "basic_check",
            "issues": []
        })
        
        # Phase 2: Plan modification
        mod_id = memory.record_plan_modification(
            "validation_failure",
            {"reason": "low confidence"},
            "Original plan",
            "Modified plan", 
            "Improving confidence"
        )
        
        # Phase 3: New iteration
        memory.start_iteration(2)
        step2_id = memory.record_execution_step(
            "Task 2", "task-2", "tool2", {}, True
        )
        
        # Phase 4: Context discovery
        context_id = memory.discover_context(
            "test_info", "result", "final", step2_id
        )
        
        # Verify complete lifecycle
        assert memory.current_iteration == 2
        assert len(memory.execution_steps) == 2
        assert len(memory.plan_modifications) == 1
        assert len(memory.discovered_context) == 1
        
        # Test replanning context includes all phases
        context = memory.get_context_for_replanning()
        assert len(context["recent_execution_steps"]) == 2
        assert len(context["recent_context_discoveries"]) == 1
        
        # Test clear functionality 
        memory.clear()
        assert len(memory.execution_steps) == 0
        assert len(memory.plan_modifications) == 0
        assert len(memory.discovered_context) == 0
        assert memory.current_iteration == 0


class TestExecutionMemoryErrorHandling:
    """Test error handling in execution memory integration."""
    
    def test_malformed_validation_result(self):
        """Test handling of malformed validation results."""
        memory = ExecutionMemory("error-test", "Error goal")
        memory.start_iteration(1)
        
        step_id = memory.record_execution_step(
            "Test task", "task-1", "test_tool", {}, True
        )
        
        # Test various malformed validation results
        malformed_validations = [
            {},  # Empty
            {"success": "invalid"},  # Wrong type
            {"confidence": "high"},  # Wrong type
            {"validation_level": "invalid_level"},  # Invalid enum
            {"success": True, "confidence": 2.0},  # Out of range
            {"success": True, "confidence": -0.5}  # Negative
        ]
        
        for validation in malformed_validations:
            # Should not crash, handle gracefully
            try:
                memory.add_validation_result(step_id, validation)
                # If it succeeds, verify step still exists
                assert len(memory.execution_steps) == 1
            except (ValueError, TypeError):
                # Acceptable to raise validation errors
                pass
    
    def test_concurrent_access_patterns(self):
        """Test patterns that might occur with concurrent access."""
        memory = ExecutionMemory("concurrent-test", "Concurrent goal")
        
        # Simulate rapid execution step recording
        step_ids = []
        for i in range(10):
            memory.start_iteration(i + 1)
            step_id = memory.record_execution_step(
                task_name=f"Rapid task {i}",
                task_id=f"task-{i}",
                tool_name=f"tool-{i}",
                tool_arguments={"index": i},
                tool_success=True,
                tool_output={"result": i}
            )
            step_ids.append(step_id)
        
        # Add validation results in different order
        for i in reversed(range(10)):
            memory.add_validation_result(step_ids[i], {
                "success": True,
                "confidence": 0.5 + (i * 0.05),
                "validation_level": "execution",
                "method": "basic_check",
                "issues": []
            })
        
        # Verify all data is consistent
        assert len(memory.execution_steps) == 10
        assert memory.current_iteration == 10
        
        # Verify all steps have validation
        for step in memory.execution_steps:
            assert step.validation_confidence > 0
            assert step.validation_success is True
    
    def test_memory_pressure_simulation(self):
        """Test behavior under memory pressure simulation."""
        memory = ExecutionMemory("pressure-test", "Memory pressure goal")
        
        # Add large amounts of data
        for iteration in range(5):
            memory.start_iteration(iteration + 1)
            
            # Multiple steps per iteration
            for step_num in range(20):
                step_id = memory.record_execution_step(
                    task_name=f"Iteration {iteration} Step {step_num}",
                    task_id=f"task-{iteration}-{step_num}",
                    tool_name="memory_tool",
                    tool_arguments={"large_data": list(range(100))},  # Simulate large data
                    tool_success=True,
                    tool_output={"large_result": list(range(200))}
                )
                
                memory.add_validation_result(step_id, {
                    "success": True,
                    "confidence": 0.7,
                    "validation_level": "execution",
                    "method": "basic_check",
                    "issues": []
                })
                
                # Add context discoveries
                memory.discover_context(
                    "performance_test",
                    f"key_{iteration}_{step_num}",
                    {"data": list(range(50))},
                    "memory_tool"
                )
        
        # Verify system still performs well
        assert len(memory.execution_steps) == 100
        assert len(memory.discovered_context) == 100
        
        # Test that context generation still works efficiently
        context = memory.get_context_for_replanning()
        assert "recent_execution_steps" in context
        assert "recent_context_discoveries" in context
        
        summary = memory.get_execution_summary()
        assert summary["total_execution_steps"] == 100
        assert summary["success_rate"] == 1.0


class TestExecutionMemoryComplexScenarios:
    """Test complex real-world scenarios."""
    
    def test_building_analysis_scenario(self):
        """Test realistic building analysis execution scenario."""
        memory = ExecutionMemory("building-analysis", "Analyze building compliance")
        
        # Phase 1: Data Loading
        memory.start_iteration(1)
        
        load_step_id = memory.record_execution_step(
            task_name="Load IFC building data",
            task_id="task-load",
            tool_name="load_building_data",
            tool_arguments={"file_path": "building.ifc"},
            tool_success=True,
            tool_output={"elements": 150, "building_name": "Test Building"}
        )
        memory.add_validation_result(load_step_id, {
            "success": True,
            "confidence": 0.95,
            "validation_level": "llm_intelligent",
            "method": "llm_reasoning",
            "issues": []
        })
        memory.discover_context("building_info", "total_elements", 150, "load_building_data")
        memory.discover_context("building_info", "name", "Test Building", "load_building_data")
        
        # Phase 2: Element Query (Failure)
        memory.start_iteration(2)
        
        query_step_id = memory.record_execution_step(
            task_name="Query door elements", 
            task_id="task-query",
            tool_name="query_elements",
            tool_arguments={"element_type": "IfcDoor"},
            tool_success=False,
            error_message="Query syntax error"
        )
        memory.add_validation_result(query_step_id, {
            "success": False,
            "confidence": 0.1,
            "validation_level": "execution",
            "method": "basic_check",
            "issues": ["Tool execution failed", "Invalid query syntax"]
        })
        
        # Phase 3: Plan Modification
        mod_id = memory.record_plan_modification(
            "execution_failure",
            {"failed_step": query_step_id, "tool": "query_elements"},
            "Use query_elements to get doors",
            "Use get_all_elements and filter for doors",
            "query_elements failed, switching to alternative approach",
            tasks_added=["filter_elements_task"],
            tasks_removed=["query_elements_task"]
        )
        
        # Phase 4: Alternative Approach (Success)
        memory.start_iteration(3)
        
        filter_step_id = memory.record_execution_step(
            task_name="Get all elements and filter doors",
            task_id="task-filter", 
            tool_name="get_all_elements",
            tool_arguments={"filter_type": "IfcDoor"},
            tool_success=True,
            tool_output={"doors": [{"id": "door1"}, {"id": "door2"}], "count": 2}
        )
        memory.add_validation_result(filter_step_id, {
            "success": True,
            "confidence": 0.9,
            "validation_level": "llm_intelligent", 
            "method": "llm_reasoning",
            "issues": []
        })
        memory.discover_context("building_elements", "door_count", 2, "get_all_elements")
        memory.discover_context("building_elements", "doors_found", True, "get_all_elements")
        
        # Phase 5: Goal Assessment
        memory.assess_goal_achievement({
            "goal_achieved": True,
            "confidence": 0.85,
            "completion_percentage": 100,
            "evidence": ["Building loaded", "Doors found and counted"],
            "remaining_tasks": []
        })
        
        # Verify complete scenario
        assert memory.current_iteration == 3
        assert len(memory.execution_steps) == 3
        assert len(memory.plan_modifications) == 1
        assert len(memory.discovered_context) == 4
        assert memory.is_goal_achieved is True
        
        # Verify replanning would not be triggered now
        assert memory.should_trigger_replanning() is False
        
        # Verify rich context for analysis
        context = memory.get_context_for_replanning()
        assert context["goal"] == "Analyze building compliance"
        assert len(context["recent_failures"]) == 1
        assert len(context["recent_context_discoveries"]) == 4
        
        summary = memory.get_execution_summary()
        assert summary["success_rate"] == 2/3  # 2 successful out of 3
        assert summary["plan_modifications"] == 1
        assert summary["goal_achieved"] is True