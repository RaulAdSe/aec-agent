"""
Unit tests for reactive agent components with NO FALLBACK mechanisms.

This test module verifies that all reactive agent components work with 
LLM-only intelligence and fail explicitly when LLM fails.
"""

import pytest
from unittest.mock import Mock
from typing import Dict, Any, List

from aec_agent.core.progress_evaluator import ProgressEvaluator, ProgressAssessment
from aec_agent.core.validator import ResultValidator
from aec_agent.core.replanner import Replanner, ReplanningTrigger
from aec_agent.core.tool_planner import ToolPlanner
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus, ExecutionResult
from tests.reactive_agent.fixtures.reactive_fixtures import (
    MockLLM, create_test_task, create_test_execution_result
)


class TestNoFallbackProgressEvaluator:
    """Test ProgressEvaluator with no fallbacks."""
    
    def test_successful_llm_evaluation(self):
        """Test that LLM evaluation works correctly."""
        mock_llm = MockLLM()
        evaluator = ProgressEvaluator(llm=mock_llm)
        
        execution_context = {
            "goal": "Count all doors in the building",
            "current_iteration": 3,
            "recent_execution_steps": [
                {
                    "task": "Load building data",
                    "tool": "load_building_data",
                    "success": True,
                    "confidence": 0.9
                }
            ],
            "discovered_context_summary": {
                "building_loaded": True,
                "total_doors": 15
            }
        }
        
        current_tasks = [create_test_task("Count doors", "Count all door elements", "task-1")]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        # Should work with MockLLM
        assessment = evaluator.evaluate_progress(
            original_goal="Count all doors in the building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved == True
        assert assessment.confidence == 0.9
    
    def test_llm_failure_raises_exception(self):
        """Test that LLM failure raises RuntimeError (no fallbacks)."""
        # Create a mock LLM that returns None (failure)
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        
        evaluator = ProgressEvaluator(llm=mock_llm)
        
        execution_context = {"goal": "Test goal"}
        current_tasks = [create_test_task("Test task", "Test description")]
        
        # Should raise RuntimeError when LLM fails
        with pytest.raises(RuntimeError, match="LLM evaluation failed.*No fallback mechanisms available"):
            evaluator.evaluate_progress(
                original_goal="Test goal",
                execution_context=execution_context,
                current_tasks=current_tasks
            )


class TestNoFallbackValidator:
    """Test Validator with no fallbacks."""
    
    def test_successful_llm_validation(self):
        """Test that LLM validation works correctly."""
        mock_llm = MockLLM()
        validator = ResultValidator(llm=mock_llm)
        
        task = create_test_task("Load data", "Load building data")
        execution_result = create_test_execution_result("load_building_data", True, {
            "status": "success", 
            "data": {"doors": 15}
        })
        
        # Should work with MockLLM
        validation_result = validator.validate_execution(task, execution_result)
        
        assert validation_result["success"] == True
        assert "successful" in validation_result["message"].lower()
    
    def test_llm_failure_raises_exception(self):
        """Test that LLM failure raises RuntimeError (no fallbacks)."""
        # Create a mock LLM that fails
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        
        validator = ResultValidator(llm=mock_llm)
        
        task = create_test_task("Test task", "Test description")
        execution_result = create_test_execution_result("test_tool", True)
        
        # Should raise RuntimeError when LLM fails
        with pytest.raises(RuntimeError, match="LLM validation failed.*No fallback mechanisms available"):
            validator.validate_execution(task, execution_result)


class TestNoFallbackReplanner:
    """Test Replanner with no fallbacks."""
    
    def test_successful_llm_replanning(self):
        """Test that LLM replanning works correctly."""
        mock_llm = MockLLM()
        replanner = Replanner(llm=mock_llm)
        
        current_goal = "Count doors in building"
        current_tasks = [create_test_task("Count doors", "Count all door elements")]
        execution_context = {
            "recent_execution_steps": [],
            "recent_failures": [],
            "plan_confidence": 0.5
        }
        trigger = ReplanningTrigger(
            trigger_type="validation_failure",
            details={"failed_task": "Count doors"},
            confidence=0.8,
            reasoning="Tool execution failed"
        )
        
        # Should work with MockLLM
        result = replanner.replan(current_goal, current_tasks, execution_context, trigger)
        
        assert result.success == True
        assert len(result.new_tasks) > 0
        assert "alternative approach" in result.reasoning.lower()
    
    def test_llm_failure_raises_exception(self):
        """Test that LLM failure raises RuntimeError (no fallbacks)."""
        # Create a mock LLM that fails
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        
        replanner = Replanner(llm=mock_llm)
        
        trigger = ReplanningTrigger(
            trigger_type="test_failure",
            details={},
            confidence=0.5,
            reasoning="Test"
        )
        
        # Should raise RuntimeError when LLM fails
        with pytest.raises(RuntimeError, match="LLM replanning failed.*No fallback mechanisms available"):
            replanner.replan("Test goal", [], {}, trigger)


class TestNoFallbackToolPlanner:
    """Test ToolPlanner with no fallbacks."""
    
    def test_successful_llm_tool_planning(self):
        """Test that LLM tool planning works correctly."""
        mock_llm = MockLLM()
        planner = ToolPlanner(llm=mock_llm)
        
        task = create_test_task("Load building data", "Load building data from file")
        context = {"building_data_loaded": False}
        
        # Should work with MockLLM
        result = planner.plan_tools(task, context)
        
        assert result["success"] == True
        assert len(result["tool_sequence"]) > 0
        assert result["method"] == "llm_reasoning"
    
    def test_llm_failure_raises_exception(self):
        """Test that LLM failure raises RuntimeError (no fallbacks)."""
        # Create a mock LLM that fails
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        
        planner = ToolPlanner(llm=mock_llm)
        
        task = create_test_task("Test task", "Test description")
        context = {}
        
        # Should raise RuntimeError when LLM fails
        with pytest.raises(RuntimeError, match="Tool planning failed.*No fallback mechanisms available"):
            planner.plan_tools(task, context)


class TestNoFallbackIntegration:
    """Integration tests verifying no fallback usage across components."""
    
    def test_all_components_use_mock_llm_successfully(self):
        """Test that all components work with MockLLM."""
        mock_llm = MockLLM()
        
        # Initialize all components with same MockLLM
        progress_evaluator = ProgressEvaluator(llm=mock_llm)
        validator = ResultValidator(llm=mock_llm)
        replanner = Replanner(llm=mock_llm)
        tool_planner = ToolPlanner(llm=mock_llm)
        
        # Test data
        task = create_test_task("Count doors", "Count all door elements")
        execution_result = create_test_execution_result("get_all_elements", True, {
            "status": "success",
            "elements": [{"type": "door", "id": "door1"}] * 15
        })
        execution_context = {
            "goal": "Count doors",
            "recent_execution_steps": [],
            "discovered_context_summary": {"total_doors": 15}
        }
        
        # All components should work
        validation_result = validator.validate_execution(task, execution_result)
        assert validation_result["success"] == True
        
        progress_assessment = progress_evaluator.evaluate_progress(
            "Count doors", execution_context, [task]
        )
        assert isinstance(progress_assessment, ProgressAssessment)
        
        tool_plan = tool_planner.plan_tools(task, {})
        assert tool_plan["success"] == True
        
        trigger = ReplanningTrigger("test", {}, 0.5, "test")
        replan_result = replanner.replan("Count doors", [task], execution_context, trigger)
        assert replan_result.success == True
        
        # Verify MockLLM was called multiple times
        assert mock_llm.call_count >= 4
    
    def test_all_components_fail_without_llm(self):
        """Test that all components fail gracefully when LLM fails."""
        # Create failing LLM mock
        failing_llm = Mock()
        failing_llm.invoke.side_effect = Exception("LLM unavailable")
        
        # Initialize components with failing LLM
        progress_evaluator = ProgressEvaluator(llm=failing_llm)
        validator = ResultValidator(llm=failing_llm)
        replanner = Replanner(llm=failing_llm)
        tool_planner = ToolPlanner(llm=failing_llm)
        
        # Test data
        task = create_test_task("Test task", "Test description")
        execution_result = create_test_execution_result("test_tool", True)
        trigger = ReplanningTrigger("test", {}, 0.5, "test")
        
        # All components should raise RuntimeError with no fallback message
        with pytest.raises(RuntimeError, match=".*No fallback mechanisms available"):
            progress_evaluator.evaluate_progress("Test goal", {}, [task])
        
        with pytest.raises(RuntimeError, match=".*No fallback mechanisms available"):
            validator.validate_execution(task, execution_result)
        
        with pytest.raises(RuntimeError, match=".*No fallback mechanisms available"):
            replanner.replan("Test goal", [task], {}, trigger)
        
        with pytest.raises(RuntimeError, match=".*No fallback mechanisms available"):
            tool_planner.plan_tools(task, {})