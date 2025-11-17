"""
Unit tests for Replanner class - Phase 2 reactive implementation.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from aec_agent.core.replanner import Replanner, ReplanningTrigger, ReplanningResult
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus
from tests.reactive_agent.fixtures.reactive_fixtures import (
    sample_tasks, sample_replanning_trigger, mock_llm, 
    create_test_task, assert_task_valid
)


class TestReplanner:
    """Test Replanner core functionality."""
    
    def test_initialization(self):
        """Test proper initialization of Replanner."""
        replanner = Replanner()
        
        assert replanner.llm is not None
        assert hasattr(replanner, 'logger')
    
    def test_replan_with_llm_success(self, mock_llm):
        """Test successful LLM-based replanning."""
        replanner = Replanner(llm=mock_llm)
        
        current_tasks = [
            create_test_task("Load data", "Load building data", "task-1"),
            create_test_task("Query doors", "Query door elements", "task-2")
        ]
        
        execution_context = {
            "goal": "Count doors in building",
            "current_iteration": 2,
            "recent_execution_steps": [
                {
                    "task": "Query doors",
                    "tool": "query_elements", 
                    "success": False,
                    "confidence": 0.1
                }
            ],
            "recent_failures": [
                {
                    "task": "Query doors",
                    "tool": "query_elements",
                    "error": "Query failed: invalid parameters"
                }
            ]
        }
        
        trigger = ReplanningTrigger(
            trigger_type="validation_failure",
            details={"failed_task": "Query doors", "tool": "query_elements"},
            confidence=0.8,
            reasoning="Tool execution failed consistently"
        )
        
        result = replanner.replan(
            current_goal="Count doors in building",
            current_tasks=current_tasks,
            execution_context=execution_context,
            trigger=trigger
        )
        
        assert isinstance(result, ReplanningResult)
        assert result.success is True
        assert len(result.new_tasks) > 0
        assert result.confidence >= 0.0
        assert result.reasoning is not None
    
    def test_replan_with_pattern_fallback(self, mock_llm):
        """Test fallback to pattern-based replanning when LLM fails."""
        # Mock LLM to return invalid response
        mock_llm.responses["replanning"] = "invalid json response"
        
        replanner = Replanner(llm=mock_llm)
        
        current_tasks = [
            create_test_task("Query doors", "Query door elements", "task-1")
        ]
        
        execution_context = {
            "goal": "Count doors in building",
            "recent_failures": [
                {
                    "task": "Query doors",
                    "tool": "query_elements",
                    "error": "Query failed"
                }
            ]
        }
        
        trigger = ReplanningTrigger(
            trigger_type="validation_failure",
            details={"failed_tool": "query_elements"},
            confidence=0.8,
            reasoning="Tool failed"
        )
        
        result = replanner.replan(
            current_goal="Count doors in building",
            current_tasks=current_tasks,
            execution_context=execution_context,
            trigger=trigger
        )
        
        assert isinstance(result, ReplanningResult)
        assert result.success is True
        assert result.method == "pattern_based"
        assert len(result.new_tasks) > 0  # Should suggest alternatives
    
    def test_pattern_based_replanning_tool_failure(self):
        """Test pattern-based replanning for tool failures."""
        replanner = Replanner()
        
        current_tasks = [
            create_test_task("Query doors", "Query door elements", "task-1")
        ]
        
        execution_context = {
            "goal": "Count doors in building",
            "recent_failures": [
                {
                    "task": "Query doors",
                    "tool": "query_elements",
                    "error": "Query failed"
                }
            ]
        }
        
        trigger = ReplanningTrigger(
            trigger_type="execution_failure",
            details={"failed_tool": "query_elements"},
            confidence=0.9,
            reasoning="Tool execution failed"
        )
        
        result = replanner._pattern_based_replan(
            current_goal="Count doors in building",
            current_tasks=current_tasks,
            execution_context=execution_context,
            trigger=trigger
        )
        
        assert result.success is True
        assert result.method == "pattern_based"
        assert "alternative" in result.reasoning.lower()
        
        # Should suggest alternative tools for door counting
        new_task_names = [task.name for task in result.new_tasks]
        assert any("get_all_elements" in name.lower() or "filter" in name.lower() 
                  for name in new_task_names)
    
    def test_pattern_based_replanning_low_confidence(self):
        """Test pattern-based replanning for low confidence results."""
        replanner = Replanner()
        
        current_tasks = [
            create_test_task("Analyze results", "Analyze door count results", "task-1")
        ]
        
        execution_context = {
            "goal": "Count doors in building",
            "recent_execution_steps": [
                {
                    "task": "Analyze results",
                    "tool": "analyze_data",
                    "success": True,
                    "confidence": 0.2  # Low confidence
                }
            ]
        }
        
        trigger = ReplanningTrigger(
            trigger_type="low_confidence",
            details={"confidence": 0.2},
            confidence=0.7,
            reasoning="Low confidence in results"
        )
        
        result = replanner._pattern_based_replan(
            current_goal="Count doors in building", 
            current_tasks=current_tasks,
            execution_context=execution_context,
            trigger=trigger
        )
        
        assert result.success is True
        assert "verification" in result.reasoning.lower() or "validation" in result.reasoning.lower()
        
        # Should suggest verification steps
        new_task_names = [task.name for task in result.new_tasks]
        assert any("verify" in name.lower() or "validate" in name.lower() 
                  for name in new_task_names)
    
    def test_pattern_based_replanning_context_discovery(self):
        """Test pattern-based replanning for context discoveries."""
        replanner = Replanner()
        
        current_tasks = [
            create_test_task("Count doors", "Count door elements", "task-1")
        ]
        
        execution_context = {
            "goal": "Count doors in building",
            "recent_context_discoveries": [
                {
                    "type": "building_info",
                    "key": "has_fire_doors",
                    "value": True,
                    "source": "analyze_building"
                }
            ]
        }
        
        trigger = ReplanningTrigger(
            trigger_type="context_discovery",
            details={"discovered": "fire_doors_present"},
            confidence=0.8,
            reasoning="Discovered additional door types"
        )
        
        result = replanner._pattern_based_replan(
            current_goal="Count doors in building",
            current_tasks=current_tasks,
            execution_context=execution_context, 
            trigger=trigger
        )
        
        assert result.success is True
        assert "context" in result.reasoning.lower() or "discovery" in result.reasoning.lower()
        
        # Should suggest enhanced analysis
        assert len(result.new_tasks) > 0
    
    def test_llm_replan_prompt_construction(self, mock_llm):
        """Test LLM replanning prompt construction."""
        replanner = Replanner(llm=mock_llm)
        
        current_tasks = [create_test_task("Test task", "Test description", "task-1")]
        
        execution_context = {
            "goal": "Test goal",
            "current_iteration": 1,
            "recent_execution_steps": [],
            "recent_failures": [],
            "plan_confidence": 0.8
        }
        
        trigger = ReplanningTrigger(
            trigger_type="test_trigger",
            details={},
            confidence=0.5,
            reasoning="Test reasoning"
        )
        
        # Call LLM replanning to trigger prompt construction
        result = replanner._llm_replan(
            current_goal="Test goal",
            current_tasks=current_tasks,
            execution_context=execution_context,
            trigger=trigger
        )
        
        # Verify LLM was called with proper prompt
        assert mock_llm.call_count > 0
        assert mock_llm.last_prompt is not None
        
        # Check prompt contains key information
        prompt_str = str(mock_llm.last_prompt)
        assert "Test goal" in prompt_str
        assert "test_trigger" in prompt_str
        assert "Test reasoning" in prompt_str
    
    def test_replan_invalid_trigger(self):
        """Test replanning with invalid trigger."""
        replanner = Replanner()
        
        current_tasks = [create_test_task("Test task", "Test description", "task-1")]
        execution_context = {"goal": "Test goal"}
        
        # Invalid trigger
        trigger = ReplanningTrigger(
            trigger_type="unknown_trigger",
            details={},
            confidence=0.5,
            reasoning="Unknown trigger type"
        )
        
        result = replanner.replan(
            current_goal="Test goal",
            current_tasks=current_tasks,
            execution_context=execution_context,
            trigger=trigger
        )
        
        # Should still succeed with generic pattern-based approach
        assert result.success is True
        assert result.method == "pattern_based"
    
    def test_replan_no_context(self):
        """Test replanning with minimal context."""
        replanner = Replanner()
        
        current_tasks = [create_test_task("Test task", "Test description", "task-1")]
        
        trigger = ReplanningTrigger(
            trigger_type="validation_failure",
            details={},
            confidence=0.5,
            reasoning="Minimal context test"
        )
        
        result = replanner.replan(
            current_goal="Test goal",
            current_tasks=current_tasks,
            execution_context={},
            trigger=trigger
        )
        
        assert result.success is True
        assert isinstance(result.new_tasks, list)
        assert result.confidence >= 0.0


class TestReplanningTrigger:
    """Test ReplanningTrigger data structure."""
    
    def test_trigger_creation(self):
        """Test creating replanning triggers."""
        trigger = ReplanningTrigger(
            trigger_type="execution_failure",
            details={"failed_tool": "query_elements", "error": "Invalid query"},
            confidence=0.9,
            reasoning="Tool execution failed after 3 attempts"
        )
        
        assert trigger.trigger_type == "execution_failure"
        assert trigger.details["failed_tool"] == "query_elements"
        assert trigger.confidence == 0.9
        assert "3 attempts" in trigger.reasoning
    
    def test_trigger_types(self):
        """Test different trigger types."""
        trigger_types = [
            "execution_failure",
            "validation_failure", 
            "low_confidence",
            "context_discovery",
            "goal_clarification"
        ]
        
        for trigger_type in trigger_types:
            trigger = ReplanningTrigger(
                trigger_type=trigger_type,
                details={},
                confidence=0.5,
                reasoning=f"Test {trigger_type}"
            )
            
            assert trigger.trigger_type == trigger_type
            assert trigger.confidence == 0.5


class TestReplanningResult:
    """Test ReplanningResult data structure."""
    
    def test_result_creation(self):
        """Test creating replanning results."""
        new_tasks = [
            create_test_task("New task 1", "Description 1", "new-task-1"),
            create_test_task("New task 2", "Description 2", "new-task-2")
        ]
        
        result = ReplanningResult(
            success=True,
            new_tasks=new_tasks,
            removed_task_ids=["old-task-1"],
            modified_task_ids=["modified-task-1"],
            reasoning="Added alternative approaches",
            confidence=0.8,
            method="llm_based"
        )
        
        assert result.success is True
        assert len(result.new_tasks) == 2
        assert result.removed_task_ids == ["old-task-1"]
        assert result.modified_task_ids == ["modified-task-1"]
        assert result.confidence == 0.8
        assert result.method == "llm_based"
        
        # Verify tasks are valid
        for task in result.new_tasks:
            assert_task_valid(task)
    
    def test_empty_result(self):
        """Test empty replanning result."""
        result = ReplanningResult(
            success=False,
            new_tasks=[],
            removed_task_ids=[],
            modified_task_ids=[],
            reasoning="No replanning needed",
            confidence=0.0,
            method="pattern_based"
        )
        
        assert result.success is False
        assert len(result.new_tasks) == 0
        assert len(result.removed_task_ids) == 0
        assert len(result.modified_task_ids) == 0


class TestReplannerErrorHandling:
    """Test error handling in Replanner."""
    
    def test_llm_exception_handling(self):
        """Test handling of LLM exceptions."""
        # Mock LLM that raises an exception
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM service unavailable")
        
        replanner = Replanner(llm=mock_llm)
        
        current_tasks = [create_test_task("Test task", "Test description", "task-1")]
        execution_context = {"goal": "Test goal"}
        
        trigger = ReplanningTrigger(
            trigger_type="validation_failure",
            details={},
            confidence=0.5,
            reasoning="Test exception handling"
        )
        
        # Should fall back to pattern-based replanning
        result = replanner.replan(
            current_goal="Test goal",
            current_tasks=current_tasks,
            execution_context=execution_context,
            trigger=trigger
        )
        
        assert result.success is True
        assert result.method == "pattern_based"
    
    def test_malformed_llm_response(self, mock_llm):
        """Test handling of malformed LLM responses."""
        # Mock LLM with invalid JSON response
        mock_llm.responses = {
            "replanning": "This is not valid JSON at all!"
        }
        
        replanner = Replanner(llm=mock_llm)
        
        current_tasks = [create_test_task("Test task", "Test description", "task-1")]
        execution_context = {"goal": "Test goal"}
        
        trigger = ReplanningTrigger(
            trigger_type="validation_failure",
            details={},
            confidence=0.5,
            reasoning="Test malformed response"
        )
        
        result = replanner.replan(
            current_goal="Test goal",
            current_tasks=current_tasks,
            execution_context=execution_context,
            trigger=trigger
        )
        
        # Should fall back to pattern-based replanning
        assert result.success is True
        assert result.method == "pattern_based"
    
    def test_empty_execution_context(self):
        """Test replanning with empty execution context."""
        replanner = Replanner()
        
        current_tasks = []
        execution_context = {}
        
        trigger = ReplanningTrigger(
            trigger_type="validation_failure",
            details={},
            confidence=0.5,
            reasoning="Empty context test"
        )
        
        result = replanner.replan(
            current_goal="",
            current_tasks=current_tasks,
            execution_context=execution_context,
            trigger=trigger
        )
        
        # Should handle gracefully
        assert isinstance(result, ReplanningResult)
        assert result.success is not None  # Should return some result