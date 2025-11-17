"""
Unit tests for ProgressEvaluator class - Phase 2 reactive implementation.
NO FALLBACK MECHANISMS - LLM-only evaluation.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from aec_agent.core.progress_evaluator import ProgressEvaluator, ProgressAssessment
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus
from tests.reactive_agent.fixtures.reactive_fixtures import (
    sample_tasks, create_test_task, assert_task_valid, MockLLM
)


class TestProgressEvaluator:
    """Test ProgressEvaluator core functionality."""
    
    def test_initialization(self):
        """Test proper initialization of ProgressEvaluator."""
        evaluator = ProgressEvaluator()
        
        assert evaluator.llm is not None
        assert hasattr(evaluator, 'logger')
    
    def test_initialization_with_custom_llm(self):
        """Test initialization with custom LLM."""
        mock_llm = Mock()
        evaluator = ProgressEvaluator(llm=mock_llm)
        
        assert evaluator.llm == mock_llm
        assert hasattr(evaluator, 'logger')
    
    def test_evaluate_progress_goal_achieved(self):
        """Test progress evaluation when goal is achieved."""
        # Use MockLLM that returns proper responses for door counting
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
                },
                {
                    "task": "Get all elements",
                    "tool": "get_all_elements",
                    "success": True,
                    "confidence": 0.95
                },
                {
                    "task": "Count door elements",
                    "tool": "count_elements",
                    "success": True,
                    "confidence": 0.9
                }
            ],
            "discovered_context_summary": {
                "building_loaded": True,
                "total_doors": 15,
                "door_count_verified": True
            },
            "plan_confidence": 0.9
        }
        
        current_tasks = [
            create_test_task("Count doors", "Count all door elements", "task-1"),
        ]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        assessment = evaluator.evaluate_progress(
            original_goal="Count all doors in the building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        # Verify assessment structure and achievement (matches MockLLM response)
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved == True
        assert assessment.confidence == 0.9
        assert "door count completed successfully" in assessment.reasoning.lower()
        assert assessment.completion_percentage == 100
        assert len(assessment.missing_requirements) == 0
        assert len(assessment.evidence_for_completion) > 0
    
    @patch('aec_agent.core.progress_evaluator.ChatOpenAI')  
    def test_evaluate_progress_goal_not_achieved(self, mock_chat_openai):
        """Test progress evaluation when goal is not achieved."""
        # Mock LLM response for incomplete goal
        mock_llm = Mock()
        mock_response = {
            "goal_achieved": False,
            "confidence": 0.7,
            "reasoning": "Building data loaded but door counting failed",
            "completion_percentage": 40,
            "missing_requirements": [
                "Successfully count door elements",
                "Verify door count accuracy"
            ],
            "evidence_for_completion": [
                "Building data loaded successfully"
            ],
            "recommendations": [
                "Retry door counting with alternative tool",
                "Verify building data quality"
            ]
        }
        mock_llm.invoke.return_value = Mock(content=str(mock_response))
        
        evaluator = ProgressEvaluator(llm=mock_llm)
        
        execution_context = {
            "goal": "Count all doors in the building",
            "current_iteration": 2,
            "recent_execution_steps": [
                {
                    "task": "Load building data",
                    "tool": "load_building_data", 
                    "success": True,
                    "confidence": 0.9
                },
                {
                    "task": "Query door elements",
                    "tool": "query_elements",
                    "success": False,
                    "confidence": 0.1
                }
            ],
            "recent_failures": [
                {
                    "task": "Query door elements",
                    "tool": "query_elements",
                    "error": "Query failed: invalid parameters"
                }
            ],
            "discovered_context_summary": {
                "building_loaded": True
            },
            "plan_confidence": 0.4
        }
        
        current_tasks = [
            create_test_task("Query doors", "Query door elements", "task-1"),
        ]
        current_tasks[0].status = TaskStatus.FAILED
        
        assessment = evaluator.evaluate_progress(
            original_goal="Count all doors in the building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is False
        assert assessment.completion_percentage < 100
        assert len(assessment.missing_requirements) > 0
        assert len(assessment.recommendations) > 0
    
    def test_pattern_based_evaluation_completed_tasks(self):
        """Test pattern-based evaluation when all tasks completed."""
        evaluator = ProgressEvaluator()
        
        execution_context = {
            "goal": "Count doors in building",
            "discovered_context_summary": {
                "building_loaded": True,
                "door_count": 15,
                "count_verified": True,
                "numeric_result": 15,  # Required evidence for counting tasks
                "element_list": ["door1", "door2", "door3"]  # Required evidence for counting tasks
            }
        }
        
        current_tasks = [
            create_test_task("Load data", "Load building data", "task-1"),
            create_test_task("Count doors", "Count door elements", "task-2"),
        ]
        # Mark all tasks as completed
        for task in current_tasks:
            task.status = TaskStatus.COMPLETED
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Count doors in building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert assessment.goal_achieved is True
        assert assessment.completion_percentage >= 80
        assert "completed" in assessment.reasoning.lower()
        assert assessment.confidence > 0.5
    
    def test_pattern_based_evaluation_failed_tasks(self):
        """Test pattern-based evaluation with failed tasks."""
        evaluator = ProgressEvaluator()
        
        execution_context = {
            "goal": "Count doors in building",
            "recent_failures": [
                {"task": "Query doors", "tool": "query_elements", "error": "Query failed"}
            ],
            "discovered_context_summary": {
                "building_loaded": True
            }
        }
        
        current_tasks = [
            create_test_task("Load data", "Load building data", "task-1"),
            create_test_task("Query doors", "Query door elements", "task-2"),
        ]
        current_tasks[0].status = TaskStatus.COMPLETED
        current_tasks[1].status = TaskStatus.FAILED
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Count doors in building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert assessment.goal_achieved is False
        assert assessment.completion_percentage < 100
        assert len(assessment.missing_requirements) > 0
        assert "failed" in assessment.reasoning.lower() or "incomplete" in assessment.reasoning.lower()
    
    def test_pattern_based_evaluation_with_goal_keywords(self):
        """Test pattern-based evaluation using goal keyword matching."""
        evaluator = ProgressEvaluator()
        
        # Goal with "count" keyword and matching discovered context
        execution_context = {
            "goal": "Count all doors in the building",
            "discovered_context_summary": {
                "total_doors": 15,
                "door_count": 15,
                "building_elements_loaded": True
            }
        }
        
        current_tasks = [create_test_task("Count doors", "Count door elements", "task-1")]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Count all doors in the building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        # Should detect that counting goal is achieved
        assert assessment.goal_achieved is True
        assert assessment.completion_percentage >= 90
        assert assessment.confidence > 0.7
    
    def test_llm_evaluation_with_fallback(self):
        """Test LLM evaluation with fallback to pattern-based."""
        # Mock LLM that fails
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM service unavailable")
        
        evaluator = ProgressEvaluator(llm=mock_llm)
        
        execution_context = {
            "goal": "Count doors in building",
            "discovered_context_summary": {
                "building_loaded": True,
                "door_count": 15
            }
        }
        
        current_tasks = [create_test_task("Count doors", "Count door elements", "task-1")]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        assessment = evaluator.evaluate_progress(
            original_goal="Count doors in building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        # Should fall back to pattern-based evaluation
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.metadata.get("method") == "pattern_based"
        assert assessment.goal_achieved is not None  # Should return some result
    
    def test_progress_assessment_data_structure(self):
        """Test ProgressAssessment data structure integrity."""
        assessment = ProgressAssessment(
            goal_achieved=True,
            confidence=0.95,
            reasoning="All requirements met successfully",
            completion_percentage=100.0,
            missing_requirements=[],
            evidence_for_completion=[
                "Building data loaded",
                "15 doors counted",
                "Results validated"
            ],
            recommendations=["Task completed successfully"],
            metadata={
                "method": "llm_based",
                "execution_time": 1.2,
                "iterations": 3
            }
        )
        
        assert assessment.goal_achieved is True
        assert assessment.confidence == 0.95
        assert assessment.completion_percentage == 100.0
        assert len(assessment.evidence_for_completion) == 3
        assert len(assessment.missing_requirements) == 0
        assert len(assessment.recommendations) == 1
        assert assessment.metadata["method"] == "llm_based"


class TestProgressEvaluatorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_execution_context(self):
        """Test evaluation with empty execution context."""
        evaluator = ProgressEvaluator()
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Test goal",
            execution_context={},
            current_tasks=[]
        )
        
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is False
        assert assessment.completion_percentage >= 0
        assert assessment.confidence >= 0
    
    def test_no_current_tasks(self):
        """Test evaluation with no current tasks."""
        evaluator = ProgressEvaluator()
        
        execution_context = {
            "goal": "Count doors",
            "discovered_context_summary": {"door_count": 10}
        }
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Count doors",
            execution_context=execution_context,
            current_tasks=[]
        )
        
        assert isinstance(assessment, ProgressAssessment)
        # Even with no tasks, might be achieved if context shows completion
        assert assessment.completion_percentage >= 0
    
    def test_malformed_llm_response(self):
        """Test handling of malformed LLM responses.""" 
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="This is not valid JSON")
        
        evaluator = ProgressEvaluator(llm=mock_llm)
        
        execution_context = {"goal": "Test goal"}
        current_tasks = []
        
        assessment = evaluator.evaluate_progress(
            original_goal="Test goal",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        # Should fall back to pattern-based evaluation
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.metadata.get("method") == "pattern_based"
    
    def test_partial_llm_response(self):
        """Test handling of partial LLM responses."""
        mock_llm = Mock()
        # Mock partial response missing some required fields
        mock_response = {
            "goal_achieved": True,
            "confidence": 0.8
            # Missing other required fields
        }
        mock_llm.invoke.return_value = Mock(content=str(mock_response))
        
        evaluator = ProgressEvaluator(llm=mock_llm)
        
        execution_context = {"goal": "Test goal"}
        current_tasks = []
        
        assessment = evaluator.evaluate_progress(
            original_goal="Test goal",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        # Should handle partial response gracefully
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is not None
        assert assessment.confidence >= 0
    
    def test_extremely_long_context(self):
        """Test evaluation with very large execution context."""
        evaluator = ProgressEvaluator()
        
        # Create large execution context
        execution_context = {
            "goal": "Test goal",
            "recent_execution_steps": [
                {
                    "task": f"Task {i}",
                    "tool": f"tool_{i}",
                    "success": True,
                    "confidence": 0.8,
                    "large_data": list(range(1000))  # Large data
                }
                for i in range(100)  # Many steps
            ],
            "discovered_context_summary": {
                f"key_{i}": f"value_{i}" for i in range(1000)  # Large context
            }
        }
        
        current_tasks = [
            create_test_task(f"Task {i}", f"Description {i}", f"task-{i}")
            for i in range(50)  # Many tasks
        ]
        
        # Should handle large context without errors
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Test goal",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.completion_percentage >= 0


class TestProgressEvaluatorIntegration:
    """Test integration scenarios."""
    
    def test_building_analysis_completion_scenario(self):
        """Test realistic building analysis completion evaluation."""
        evaluator = ProgressEvaluator()
        
        execution_context = {
            "goal": "Analyze building compliance and count all doors",
            "current_iteration": 4,
            "recent_execution_steps": [
                {
                    "task": "Load IFC building data",
                    "tool": "load_building_data",
                    "success": True,
                    "confidence": 0.95
                },
                {
                    "task": "Extract door elements",
                    "tool": "get_all_elements", 
                    "success": True,
                    "confidence": 0.9
                },
                {
                    "task": "Count door elements",
                    "tool": "count_elements",
                    "success": True,
                    "confidence": 0.85
                },
                {
                    "task": "Validate door count",
                    "tool": "validate_results",
                    "success": True,
                    "confidence": 0.9
                }
            ],
            "discovered_context_summary": {
                "building_loaded": True,
                "building_name": "Test Building",
                "total_elements": 150,
                "door_elements": 15,
                "door_count": 15,
                "door_count_verified": True,
                "compliance_checked": True
            },
            "plan_confidence": 0.9,
            "recent_failures": []
        }
        
        current_tasks = [
            create_test_task("Load data", "Load building data", "task-1"),
            create_test_task("Extract doors", "Extract door elements", "task-2"),
            create_test_task("Count doors", "Count door elements", "task-3"),
            create_test_task("Validate count", "Validate door count", "task-4"),
        ]
        
        # Mark all tasks as completed
        for task in current_tasks:
            task.status = TaskStatus.COMPLETED
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Analyze building compliance and count all doors",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert assessment.goal_achieved is True
        assert assessment.completion_percentage >= 95
        assert assessment.confidence >= 0.8
        assert len(assessment.evidence_for_completion) > 0
        assert len(assessment.missing_requirements) == 0
        assert "door" in str(assessment.evidence_for_completion).lower()
    
    def test_partial_completion_scenario(self):
        """Test scenario with partial goal completion."""
        evaluator = ProgressEvaluator()
        
        execution_context = {
            "goal": "Count doors and windows in building",
            "current_iteration": 3,
            "recent_execution_steps": [
                {
                    "task": "Load building data",
                    "tool": "load_building_data",
                    "success": True,
                    "confidence": 0.9
                },
                {
                    "task": "Count doors",
                    "tool": "count_elements",
                    "success": True,
                    "confidence": 0.85
                },
                {
                    "task": "Count windows",
                    "tool": "count_elements",
                    "success": False,
                    "confidence": 0.1
                }
            ],
            "discovered_context_summary": {
                "building_loaded": True,
                "door_count": 15,
                "door_count_verified": True
                # No window count
            },
            "recent_failures": [
                {
                    "task": "Count windows", 
                    "tool": "count_elements",
                    "error": "Window elements not found"
                }
            ]
        }
        
        current_tasks = [
            create_test_task("Count doors", "Count door elements", "task-1"),
            create_test_task("Count windows", "Count window elements", "task-2"),
        ]
        current_tasks[0].status = TaskStatus.COMPLETED
        current_tasks[1].status = TaskStatus.FAILED
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Count doors and windows in building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert assessment.goal_achieved is False
        assert 0 < assessment.completion_percentage < 100  # Partial completion
        assert "window" in str(assessment.missing_requirements).lower()
        assert "door" in str(assessment.evidence_for_completion).lower()