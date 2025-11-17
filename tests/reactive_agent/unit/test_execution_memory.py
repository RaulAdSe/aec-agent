"""
Unit tests for ExecutionMemory class - Phase 1 reactive implementation.
"""

import pytest
import uuid
from datetime import datetime
from typing import Dict, Any

from aec_agent.memory.execution_memory import (
    ExecutionMemory, ExecutionStep, PlanModification, DiscoveredContext,
    ValidationLevel
)
from tests.reactive_agent.fixtures.reactive_fixtures import (
    sample_task, successful_execution_result, failed_execution_result,
    sample_execution_context, create_test_task, create_test_execution_result,
    assert_task_valid, assert_execution_result_valid
)


class TestExecutionMemory:
    """Test ExecutionMemory core functionality."""
    
    def test_initialization(self):
        """Test proper initialization of ExecutionMemory."""
        session_id = "test-session-123"
        goal = "Count doors in building"
        
        memory = ExecutionMemory(session_id, goal)
        
        assert memory.session_id == session_id
        assert memory.goal == goal
        assert memory.current_iteration == 0
        assert len(memory.execution_steps) == 0
        assert len(memory.plan_modifications) == 0
        assert len(memory.discovered_context) == 0
        assert memory.current_plan_confidence == 1.0
        assert memory.is_goal_achieved is False
    
    def test_start_iteration(self):
        """Test iteration management."""
        memory = ExecutionMemory("test", "test goal")
        
        # Start first iteration
        memory.start_iteration(1)
        assert memory.current_iteration == 1
        
        # Start second iteration
        memory.start_iteration(2) 
        assert memory.current_iteration == 2
    
    def test_record_execution_step_success(self):
        """Test recording successful execution steps."""
        memory = ExecutionMemory("test", "test goal")
        memory.start_iteration(1)
        
        step_id = memory.record_execution_step(
            task_name="Load building data",
            task_id="task-1",
            tool_name="load_building_data",
            tool_arguments={"file_path": "test.json"},
            tool_success=True,
            tool_output={"doors": 15, "total_elements": 150}
        )
        
        assert step_id is not None
        assert len(memory.execution_steps) == 1
        
        step = memory.execution_steps[0]
        assert step.id == step_id
        assert step.iteration == 1
        assert step.task_name == "Load building data"
        assert step.task_id == "task-1"
        assert step.tool_name == "load_building_data"
        assert step.tool_success is True
        assert step.tool_output == {"doors": 15, "total_elements": 150}
        assert step.error_message is None
    
    def test_record_execution_step_failure(self):
        """Test recording failed execution steps."""
        memory = ExecutionMemory("test", "test goal")
        memory.start_iteration(1)
        
        step_id = memory.record_execution_step(
            task_name="Query elements",
            task_id="task-2", 
            tool_name="query_elements",
            tool_arguments={"element_type": "doors"},
            tool_success=False,
            error_message="Query failed: invalid parameters"
        )
        
        assert step_id is not None
        assert len(memory.execution_steps) == 1
        
        step = memory.execution_steps[0]
        assert step.id == step_id
        assert step.tool_success is False
        assert step.error_message == "Query failed: invalid parameters"
        assert step.tool_output is None
    
    def test_add_validation_result(self):
        """Test adding validation results to execution steps."""
        memory = ExecutionMemory("test", "test goal")
        memory.start_iteration(1)
        
        step_id = memory.record_execution_step(
            task_name="Test task",
            task_id="task-1",
            tool_name="test_tool",
            tool_arguments={},
            tool_success=True
        )
        
        validation_result = {
            "success": True,
            "confidence": 0.9,
            "validation_level": "llm_intelligent",
            "method": "llm_reasoning",
            "issues": []
        }
        
        memory.add_validation_result(step_id, validation_result)
        
        step = memory.execution_steps[0]
        assert step.validation_success is True
        assert step.validation_confidence == 0.9
        assert step.validation_level == ValidationLevel.LLM_INTELLIGENT
        assert step.validation_method == "llm_reasoning"
        assert step.validation_issues == []
    
    def test_discover_context(self):
        """Test context discovery recording."""
        memory = ExecutionMemory("test", "test goal")
        
        memory.discover_context(
            discovery_type="building_info",
            key="total_doors", 
            value=15,
            source_tool="test_tool",
            confidence=0.95
        )
        
        assert len(memory.discovered_context) == 1
        
        context = memory.discovered_context[0]
        assert context.discovery_type == "building_info"
        assert context.key == "total_doors"
        assert context.value == 15
        assert context.source_tool == "test_tool"
        assert context.confidence == 0.95
    
    def test_record_plan_modification(self):
        """Test recording plan modifications."""
        memory = ExecutionMemory("test", "test goal")
        
        modification_id = memory.record_plan_modification(
            trigger_type="validation_failure",
            trigger_details={"failed_step": "step-123"},
            original_plan_summary="Original plan",
            new_plan_summary="Updated plan", 
            modification_reasoning="Tool failed, trying alternative",
            tasks_added=["new-task-1"],
            tasks_removed=["failed-task-1"],
            tasks_modified=["modified-task-1"]
        )
        
        assert modification_id is not None
        assert len(memory.plan_modifications) == 1
        
        modification = memory.plan_modifications[0]
        assert modification.id == modification_id
        assert modification.trigger_type == "validation_failure"
        assert modification.modification_reasoning == "Tool failed, trying alternative"
        assert modification.tasks_added == ["new-task-1"]
        assert modification.tasks_removed == ["failed-task-1"]
        assert modification.tasks_modified == ["modified-task-1"]
    
    def test_get_current_context(self):
        """Test current context retrieval."""
        memory = ExecutionMemory("test", "Count doors in building")
        memory.start_iteration(1)
        
        # Add execution step
        step_id = memory.record_execution_step(
            task_name="Load data",
            task_id="task-1",
            tool_name="load_building_data",
            tool_arguments={"file_path": "test.json"},
            tool_success=True
        )
        
        # Add context discovery
        memory.discover_context("building_info", "total_doors", 15, "load_building_data")
        
        context = memory.get_current_context()
        
        assert context["goal"] == "Count doors in building"
        assert context["current_iteration"] == 1
        assert context["plan_confidence"] == 1.0
        assert context["total_execution_steps"] == 1
        assert context["is_goal_achieved"] is False
    
    def test_get_execution_summary(self):
        """Test execution summary generation."""
        memory = ExecutionMemory("test", "test goal")
        memory.start_iteration(1)
        
        # Add successful step
        step1_id = memory.record_execution_step(
            task_name="Success task",
            task_id="task-1",
            tool_name="working_tool",
            tool_arguments={},
            tool_success=True
        )
        memory.add_validation_result(step1_id, {
            "success": True,
            "confidence": 0.9,
            "validation_level": "execution",
            "method": "basic_check",
            "issues": []
        })
        
        # Add failed step
        step2_id = memory.record_execution_step(
            task_name="Failed task",
            task_id="task-2",
            tool_name="broken_tool",
            tool_arguments={},
            tool_success=False,
            error_message="Tool error"
        )
        memory.add_validation_result(step2_id, {
            "success": False,
            "confidence": 0.1,
            "validation_level": "execution", 
            "method": "basic_check",
            "issues": ["Tool execution failed"]
        })
        
        summary = memory.get_execution_summary()
        
        assert summary["total_iterations"] == 1
        assert summary["total_execution_steps"] == 2
        assert summary["successful_steps"] == 1
        assert summary["failed_steps"] == 1
        assert summary["success_rate"] == 0.5
        assert summary["goal_achieved"] is False
        assert summary["plan_confidence"] == 1.0
    
    def test_should_trigger_replanning_failure_pattern(self):
        """Test replanning trigger on repeated failures."""
        memory = ExecutionMemory("test", "test goal")
        
        # Add multiple failures with same tool
        for i in range(3):
            memory.start_iteration(i + 1)
            step_id = memory.record_execution_step(
                task_name=f"Query attempt {i+1}",
                task_id=f"task-{i+1}",
                tool_name="query_elements",
                tool_arguments={"element_type": "doors"},
                tool_success=False,
                error_message="Query failed"
            )
            memory.add_validation_result(step_id, {
                "success": False,
                "confidence": 0.1,
                "validation_level": "execution",
                "method": "basic_check",
                "issues": ["Tool execution failed"]
            })
        
        assert memory.should_trigger_replanning() is True
    
    def test_should_trigger_replanning_low_confidence(self):
        """Test replanning trigger on low validation confidence."""
        memory = ExecutionMemory("test", "test goal")
        
        # Add steps with consistently low confidence
        for i in range(2):
            memory.start_iteration(i + 1)
            step_id = memory.record_execution_step(
                task_name=f"Task {i+1}",
                task_id=f"task-{i+1}",
                tool_name="some_tool",
                tool_arguments={},
                tool_success=True
            )
            memory.add_validation_result(step_id, {
                "success": True,
                "confidence": 0.2,  # Low confidence
                "validation_level": "llm_intelligent",
                "method": "llm_reasoning",
                "issues": ["Low confidence result"]
            })
        
        assert memory.should_trigger_replanning() is True
    
    def test_should_not_trigger_replanning_success(self):
        """Test no replanning trigger on successful execution."""
        memory = ExecutionMemory("test", "test goal")
        memory.start_iteration(1)
        
        step_id = memory.record_execution_step(
            task_name="Successful task",
            task_id="task-1",
            tool_name="working_tool",
            tool_arguments={},
            tool_success=True
        )
        memory.add_validation_result(step_id, {
            "success": True,
            "confidence": 0.9,
            "validation_level": "execution",
            "method": "basic_check",
            "issues": []
        })
        
        assert memory.should_trigger_replanning() is False
    
    def test_get_context_for_replanning(self):
        """Test replanning context generation."""
        memory = ExecutionMemory("test", "Count doors")
        memory.start_iteration(1)
        
        # Add execution step
        step_id = memory.record_execution_step(
            task_name="Failed task",
            task_id="task-1", 
            tool_name="broken_tool",
            tool_arguments={},
            tool_success=False,
            error_message="Tool error"
        )
        memory.add_validation_result(step_id, {
            "success": False,
            "confidence": 0.1,
            "validation_level": "execution",
            "method": "basic_check",
            "issues": ["Tool execution failed"]
        })
        
        # Add context discovery
        memory.discover_context("building_info", "doors", 15, "broken_tool")
        
        context = memory.get_context_for_replanning()
        
        assert context["goal"] == "Count doors"
        assert context["current_iteration"] == 1
        assert "recent_execution_steps" in context
        assert "recent_failures" in context  
        assert "recent_context_discoveries" in context
        assert "plan_confidence" in context
    
    def test_clear(self):
        """Test clearing execution memory."""
        memory = ExecutionMemory("test", "test goal")
        memory.start_iteration(1)
        
        # Add some data
        step_id = memory.record_execution_step(
            task_name="Test task",
            task_id="task-1",
            tool_name="test_tool", 
            tool_arguments={},
            tool_success=True
        )
        memory.discover_context("test", "key", "value", "test_tool")
        memory.record_plan_modification(
            "test", {}, "old", "new", "reason"
        )
        
        # Verify data exists
        assert len(memory.execution_steps) > 0
        assert len(memory.discovered_context) > 0
        assert len(memory.plan_modifications) > 0
        
        # Clear and verify
        memory.clear()
        
        assert len(memory.execution_steps) == 0
        assert len(memory.discovered_context) == 0
        assert len(memory.plan_modifications) == 0
        assert memory.current_iteration == 0
        assert memory.current_plan_confidence == 1.0
        assert memory.is_goal_achieved is False


class TestExecutionMemoryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_execution_context(self):
        """Test execution context with no steps."""
        memory = ExecutionMemory("test", "test goal")
        
        context = memory.get_current_context()
        
        assert context["goal"] == "test goal"
        assert context["current_iteration"] == 0
        assert context["total_execution_steps"] == 0
        assert context["plan_confidence"] == 1.0
        assert context["is_goal_achieved"] is False
    
    def test_validation_result_on_nonexistent_step(self):
        """Test adding validation result to non-existent step."""
        memory = ExecutionMemory("test", "test goal")
        
        # Should not raise exception, just ignore silently
        memory.add_validation_result("invalid-id", {"success": True})
        
        # No steps should exist
        assert len(memory.execution_steps) == 0
    
    def test_large_execution_history(self):
        """Test performance with large execution history."""
        memory = ExecutionMemory("test", "test goal")
        
        # Add many execution steps
        for i in range(20):
            memory.start_iteration(i + 1)
            step_id = memory.record_execution_step(
                task_name=f"Task {i}",
                task_id=f"task-{i}",
                tool_name="test_tool",
                tool_arguments={},
                tool_success=True
            )
            memory.add_validation_result(step_id, {
                "success": True,
                "confidence": 0.8,
                "validation_level": "execution", 
                "method": "basic_check",
                "issues": []
            })
        
        # Test context generation still works efficiently
        context = memory.get_context_for_replanning()
        assert "recent_execution_steps" in context
        assert context["current_iteration"] == 20
        
        summary = memory.get_execution_summary()
        assert summary["total_execution_steps"] == 20
    
    def test_mixed_success_failure_confidence(self):
        """Test confidence calculation with mixed results.""" 
        memory = ExecutionMemory("test", "test goal")
        
        # Add mix of successful and failed steps
        confidences = [0.9, 0.1, 0.8, 0.2, 0.7]
        for i, conf in enumerate(confidences):
            memory.start_iteration(i + 1)
            step_id = memory.record_execution_step(
                task_name=f"Task {i}",
                task_id=f"task-{i}",
                tool_name="test_tool",
                tool_arguments={},
                tool_success=conf > 0.5
            )
            memory.add_validation_result(step_id, {
                "success": conf > 0.5,
                "confidence": conf,
                "validation_level": "execution",
                "method": "basic_check",
                "issues": [] if conf > 0.5 else ["Low confidence"]
            })
        
        summary = memory.get_execution_summary()
        
        # Should have reasonable metrics based on mixed results
        assert 0.0 <= summary["plan_confidence"] <= 1.0
        assert summary["success_rate"] == 0.6  # 3 successes out of 5
        assert summary["failed_steps"] == 2


class TestExecutionMemoryDataStructures:
    """Test data structure integrity and relationships."""
    
    def test_execution_step_data_integrity(self):
        """Test ExecutionStep data structure integrity."""
        memory = ExecutionMemory("test", "test goal")
        memory.start_iteration(1)
        
        step_id = memory.record_execution_step(
            task_name="Test task",
            task_id="task-1",
            tool_name="test_tool",
            tool_arguments={"param1": "value1", "param2": 123},
            tool_output={"result": "success", "data": [1, 2, 3]},
            tool_success=True
        )
        
        step = memory.execution_steps[0]
        
        # Verify all fields are properly set
        assert isinstance(step.id, str)
        assert step.iteration == 1
        assert step.task_name == "Test task"
        assert step.task_id == "task-1"
        assert step.tool_name == "test_tool"
        assert step.tool_arguments == {"param1": "value1", "param2": 123}
        assert step.tool_success is True
        assert step.tool_output == {"result": "success", "data": [1, 2, 3]}
        assert step.execution_time is not None  # Calculated automatically
        assert step.error_message is None  # Default empty string
    
    def test_plan_modification_data_integrity(self):
        """Test PlanModification data structure integrity."""
        memory = ExecutionMemory("test", "test goal")
        
        mod_id = memory.record_plan_modification(
            trigger_type="validation_failure",
            trigger_details={
                "failed_step": "step-123",
                "error": "Tool failed"
            },
            original_plan_summary="Old plan",
            new_plan_summary="Updated plan",
            modification_reasoning="Better approach found",
            tasks_added=["new-task"],
            tasks_removed=["old-task"],
            tasks_modified=["updated-task"]
        )
        
        modification = memory.plan_modifications[0]
        
        # Verify all fields
        assert isinstance(modification.id, str)
        assert modification.trigger_type == "validation_failure"
        assert modification.trigger_details["failed_step"] == "step-123"
        assert modification.modification_reasoning == "Better approach found"
        assert modification.tasks_added == ["new-task"]
        assert modification.tasks_removed == ["old-task"]
        assert modification.tasks_modified == ["updated-task"]
    
    def test_discovered_context_data_integrity(self):
        """Test DiscoveredContext data structure integrity."""
        memory = ExecutionMemory("test", "test goal")
        memory.start_iteration(1)
        
        step_id = memory.record_execution_step(
            task_name="Test task",
            task_id="task-1", 
            tool_name="test_tool",
            tool_arguments={},
            tool_success=True
        )
        
        memory.discover_context(
            discovery_type="building_analysis",
            key="room_count",
            value={"total": 25, "by_type": {"bedroom": 10, "bathroom": 5}},
            source_tool="test_tool",
            confidence=0.85
        )
        
        context = memory.discovered_context[0]
        
        # Verify all fields
        assert context.discovery_type == "building_analysis"
        assert context.key == "room_count"
        assert context.value == {"total": 25, "by_type": {"bedroom": 10, "bathroom": 5}}
        assert context.source_tool == "test_tool"
        assert context.confidence == 0.85