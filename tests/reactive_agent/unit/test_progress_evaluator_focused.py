"""
Focused unit tests for ProgressEvaluator class - Phase 2 reactive implementation.
Tests the actual implementation behavior instead of assumed behavior.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from aec_agent.core.progress_evaluator import ProgressEvaluator, ProgressAssessment
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus
from tests.reactive_agent.fixtures.reactive_fixtures import (
    create_test_task
)


class TestProgressEvaluatorCore:
    """Test core ProgressEvaluator functionality with actual implementation."""
    
    def test_initialization(self):
        """Test proper initialization of ProgressEvaluator."""
        evaluator = ProgressEvaluator()
        
        assert evaluator.llm is not None
        assert hasattr(evaluator, 'logger')
        assert hasattr(evaluator, 'evaluation_patterns')
        assert 'counting_tasks' in evaluator.evaluation_patterns
        assert 'compliance_analysis' in evaluator.evaluation_patterns
    
    def test_evaluation_patterns_structure(self):
        """Test that evaluation patterns have correct structure."""
        evaluator = ProgressEvaluator()
        
        for pattern_name, pattern_config in evaluator.evaluation_patterns.items():
            assert "completion_indicators" in pattern_config
            assert "evidence_requirements" in pattern_config
            assert "minimum_confidence" in pattern_config
            
            assert isinstance(pattern_config["completion_indicators"], list)
            assert isinstance(pattern_config["evidence_requirements"], list)
            assert isinstance(pattern_config["minimum_confidence"], (int, float))
    
    def test_conservative_assessment_all_completed(self):
        """Test conservative assessment when all tasks are completed."""
        evaluator = ProgressEvaluator()
        
        current_tasks = [
            create_test_task("Load data", "Load building data", "task-1"),
            create_test_task("Count doors", "Count door elements", "task-2"),
        ]
        
        # Mark all tasks as completed
        for task in current_tasks:
            task.status = TaskStatus.COMPLETED
        
        assessment = evaluator._conservative_assessment(
            original_goal="Count doors in building",
            execution_context={},
            current_tasks=current_tasks
        )
        
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is True
        assert assessment.completion_percentage == 100.0
        assert assessment.confidence == 0.5
        assert "2/2 tasks completed" in assessment.reasoning
        assert len(assessment.missing_requirements) == 0
    
    def test_conservative_assessment_partial_completion(self):
        """Test conservative assessment with partial task completion."""
        evaluator = ProgressEvaluator()
        
        current_tasks = [
            create_test_task("Load data", "Load building data", "task-1"),
            create_test_task("Count doors", "Count door elements", "task-2"),
        ]
        
        # Mark only first task as completed
        current_tasks[0].status = TaskStatus.COMPLETED
        current_tasks[1].status = TaskStatus.PENDING
        
        assessment = evaluator._conservative_assessment(
            original_goal="Count doors in building",
            execution_context={},
            current_tasks=current_tasks
        )
        
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is False
        assert assessment.completion_percentage == 50.0
        assert assessment.confidence == 0.2
        assert "1/2 tasks completed" in assessment.reasoning
        assert len(assessment.missing_requirements) > 0
        assert "Count doors" in str(assessment.missing_requirements)
    
    def test_pattern_based_evaluation_counting_tasks_success(self):
        """Test pattern-based evaluation for counting tasks with proper evidence."""
        evaluator = ProgressEvaluator()
        
        # Provide context with required evidence for counting tasks
        execution_context = {
            "discovered_context_summary": {
                "numeric_result": 15,  # Required evidence
                "element_list": ["door1", "door2", "door3"],  # Required evidence
                "building_loaded": True
            }
        }
        
        current_tasks = [
            create_test_task("Count doors", "Count door elements", "task-1")
        ]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Count doors in building",  # Contains "count" - counting task
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert assessment is not None
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is True
        assert assessment.completion_percentage == 100.0
        assert assessment.confidence >= 0.8  # High confidence for counting tasks
        assert assessment.metadata["goal_type"] == "counting_tasks"
        assert len(assessment.evidence_for_completion) == 2  # Both evidence types found
    
    def test_pattern_based_evaluation_counting_tasks_missing_evidence(self):
        """Test pattern-based evaluation for counting tasks without required evidence."""
        evaluator = ProgressEvaluator()
        
        # Context without required evidence
        execution_context = {
            "discovered_context_summary": {
                "building_loaded": True
                # Missing numeric_result and element_list
            }
        }
        
        current_tasks = [
            create_test_task("Count doors", "Count door elements", "task-1")
        ]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Count doors in building",  # Contains "count" - counting task
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert assessment is not None
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is False  # No evidence found
        assert assessment.completion_percentage == 100.0  # Tasks completed but no evidence
        assert assessment.confidence == 0.3  # Low confidence due to missing evidence
        assert len(assessment.missing_requirements) == 2  # Both evidence types missing
    
    def test_pattern_based_evaluation_compliance_analysis(self):
        """Test pattern-based evaluation for compliance analysis tasks."""
        evaluator = ProgressEvaluator()
        
        execution_context = {
            "discovered_context_summary": {
                "validation_result": "passed",  # Required evidence
                "rule_check": "compliant",  # Required evidence
                "building_analyzed": True
            }
        }
        
        current_tasks = [
            create_test_task("Check compliance", "Check building compliance", "task-1")
        ]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Check if building is compliant with regulations",  # Contains "compliant"
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert assessment is not None
        assert assessment.goal_achieved is True
        assert assessment.metadata["goal_type"] == "compliance_analysis"
        assert assessment.confidence >= 0.9  # High confidence for compliance analysis
    
    def test_pattern_based_evaluation_no_matching_pattern(self):
        """Test pattern-based evaluation when no pattern matches."""
        evaluator = ProgressEvaluator()
        
        execution_context = {"discovered_context_summary": {}}
        current_tasks = [create_test_task("Unknown task", "Do something unknown", "task-1")]
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Do some unknown task",  # No matching pattern
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert assessment is None  # Should return None when no pattern matches
    
    def test_evaluate_progress_fallback_chain(self):
        """Test that evaluate_progress properly falls through the fallback chain."""
        # Create mock LLM that fails
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM service unavailable")
        
        evaluator = ProgressEvaluator(llm=mock_llm)
        
        execution_context = {
            "discovered_context_summary": {
                "building_loaded": True
            }
        }
        
        current_tasks = [
            create_test_task("Unknown task", "Do something", "task-1")
        ]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        # Should fallback to conservative assessment since LLM fails and no pattern matches
        assessment = evaluator.evaluate_progress(
            original_goal="Do some unknown task",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is True  # Conservative: all tasks completed
        assert assessment.completion_percentage == 100.0
        assert "Conservative evaluation" in assessment.reasoning


class TestProgressEvaluatorDataExtraction:
    """Test specific pattern types."""
    
    def test_data_extraction_pattern(self):
        """Test data extraction pattern recognition and evaluation."""
        evaluator = ProgressEvaluator()
        
        execution_context = {
            "discovered_context_summary": {
                "data_output": {"elements": 150},  # Required evidence
                "element_properties": {"doors": 15},  # Required evidence
                "extraction_completed": True
            }
        }
        
        current_tasks = [
            create_test_task("Extract data", "Extract building data", "task-1")
        ]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Load extracted building data from IFC file",  # Contains "extracted"
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert assessment is not None
        assert assessment.goal_achieved is True
        assert assessment.metadata["goal_type"] == "data_extraction"
        assert assessment.confidence >= 0.7
    
    def test_search_tasks_pattern(self):
        """Test search tasks pattern recognition and evaluation."""
        evaluator = ProgressEvaluator()
        
        execution_context = {
            "discovered_context_summary": {
                "search_results": ["door1", "door2"],  # Required evidence
                "match_found": True,  # Required evidence
                "search_completed": True
            }
        }
        
        current_tasks = [
            create_test_task("Find doors", "Find door elements", "task-1")
        ]
        current_tasks[0].status = TaskStatus.COMPLETED
        
        assessment = evaluator._pattern_based_evaluation(
            original_goal="Elements need to be searched in building",  # Contains "searched"  
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert assessment is not None
        assert assessment.goal_achieved is True
        assert assessment.metadata["goal_type"] == "search_tasks"
        assert assessment.confidence >= 0.7


class TestProgressEvaluatorErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_tasks_conservative_assessment(self):
        """Test conservative assessment with no tasks."""
        evaluator = ProgressEvaluator()
        
        assessment = evaluator._conservative_assessment(
            original_goal="Test goal",
            execution_context={},
            current_tasks=[]
        )
        
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is False  # No tasks completed
        assert assessment.completion_percentage == 0.0
        assert assessment.confidence == 0.2
        assert "0/0 tasks completed" in assessment.reasoning
    
    def test_mixed_task_statuses(self):
        """Test evaluation with mixed task completion statuses."""
        evaluator = ProgressEvaluator()
        
        current_tasks = [
            create_test_task("Task 1", "Description 1", "task-1"),
            create_test_task("Task 2", "Description 2", "task-2"), 
            create_test_task("Task 3", "Description 3", "task-3"),
        ]
        
        current_tasks[0].status = TaskStatus.COMPLETED
        current_tasks[1].status = TaskStatus.FAILED
        current_tasks[2].status = TaskStatus.PENDING
        
        assessment = evaluator._conservative_assessment(
            original_goal="Complete all tasks",
            execution_context={},
            current_tasks=current_tasks
        )
        
        assert assessment.goal_achieved is False
        assert assessment.completion_percentage == (1/3) * 100  # Only 1 completed
        assert len(assessment.missing_requirements) == 2  # 2 non-completed tasks


class TestProgressEvaluatorIntegrationSimple:
    """Simple integration tests that work with actual implementation."""
    
    def test_door_counting_success_scenario(self):
        """Test successful door counting scenario."""
        evaluator = ProgressEvaluator()
        
        execution_context = {
            "goal": "Count all doors in the building",
            "discovered_context_summary": {
                "building_loaded": True,
                "total_doors": 15,
                "door_list": ["door1", "door2", "door3"],
                "numeric_result": 15,
                "element_list": ["door1", "door2", "door3", "door4", "door5"]
            }
        }
        
        current_tasks = [
            create_test_task("Load building", "Load IFC building data", "task-1"),
            create_test_task("Extract doors", "Extract door elements", "task-2"),
            create_test_task("Count doors", "Count door elements", "task-3"),
        ]
        
        # Mark all as completed
        for task in current_tasks:
            task.status = TaskStatus.COMPLETED
        
        # Should use pattern-based evaluation for counting task
        assessment = evaluator.evaluate_progress(
            original_goal="Count all doors in the building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is True
        assert assessment.completion_percentage == 100.0
        assert assessment.confidence >= 0.8
    
    def test_partial_completion_scenario(self):
        """Test partial completion scenario."""
        evaluator = ProgressEvaluator()
        
        execution_context = {
            "discovered_context_summary": {
                "building_loaded": True
                # Missing required evidence for counting
            }
        }
        
        current_tasks = [
            create_test_task("Load building", "Load building data", "task-1"),
            create_test_task("Count doors", "Count door elements", "task-2"),
        ]
        
        current_tasks[0].status = TaskStatus.COMPLETED
        current_tasks[1].status = TaskStatus.FAILED
        
        assessment = evaluator.evaluate_progress(
            original_goal="Count doors in building",
            execution_context=execution_context,
            current_tasks=current_tasks
        )
        
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.goal_achieved is False
        assert assessment.completion_percentage == 50.0  # 1 of 2 completed
        assert assessment.confidence <= 0.5