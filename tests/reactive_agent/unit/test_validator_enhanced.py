"""
Unit tests for enhanced Validator class - Phase 2 reactive implementation.
Tests the replanning assessment capabilities and enhanced validation.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from aec_agent.core.validator import ResultValidator, ValidationResult
from aec_agent.core.reasoning_utils import Task, Priority, TaskStatus, ExecutionResult
from tests.reactive_agent.fixtures.reactive_fixtures import (
    create_test_task, create_test_execution_result
)


class TestValidatorCore:
    """Test core Validator functionality."""
    
    def test_initialization(self):
        """Test proper initialization of ResultValidator."""
        validator = ResultValidator()
        
        assert validator.llm is not None
        assert hasattr(validator, 'logger')
    
    def test_initialization_with_custom_llm(self):
        """Test initialization with custom LLM."""
        mock_llm = Mock()
        validator = ResultValidator(llm=mock_llm)
        
        assert validator.llm == mock_llm
        assert hasattr(validator, 'logger')
    
    def test_validate_execution_success(self):
        """Test basic execution validation for successful result."""
        validator = ResultValidator()
        
        task = create_test_task("Load data", "Load building data", "task-1")
        execution_result = create_test_execution_result(
            "load_building_data", 
            success=True, 
            output={"elements": 150, "building_name": "Test Building"}
        )
        
        validation = validator.validate_execution(task, execution_result)
        
        assert isinstance(validation, dict)
        assert "success" in validation
        assert "confidence" in validation
        assert "validation_level" in validation
        assert "method" in validation
        assert "issues" in validation
        
        # Should be successful validation
        assert validation["success"] is True
        assert validation["confidence"] > 0.0
    
    def test_validate_execution_failure(self):
        """Test execution validation for failed result."""
        validator = ResultValidator()
        
        task = create_test_task("Query doors", "Query door elements", "task-1")
        execution_result = create_test_execution_result(
            "query_elements",
            success=False,
            output=None
        )
        execution_result.error_message = "Query failed: invalid parameters"
        
        validation = validator.validate_execution(task, execution_result)
        
        assert isinstance(validation, dict)
        assert validation["success"] is False
        assert validation["confidence"] <= 0.5
        assert len(validation["issues"]) > 0
        assert any("failed" in issue.lower() for issue in validation["issues"])
    
    def test_validate_with_replanning_assessment_success(self):
        """Test validation with replanning assessment for successful execution."""
        validator = ResultValidator()
        
        task = create_test_task("Count doors", "Count door elements", "task-1")
        execution_result = create_test_execution_result(
            "count_elements",
            success=True,
            output={"count": 15, "elements": ["door1", "door2"]}
        )
        
        execution_context = {
            "goal": "Count all doors in building",
            "current_iteration": 2,
            "recent_execution_steps": [
                {
                    "task": "Load data",
                    "tool": "load_building_data",
                    "success": True,
                    "confidence": 0.9
                }
            ],
            "plan_confidence": 0.8
        }
        
        validation = validator.validate_with_replanning_assessment(
            task, execution_result, execution_context
        )
        
        assert isinstance(validation, dict)
        assert validation["success"] is True
        
        # Should include replanning assessment
        assert "replanning_needed" in validation
        assert "replanning_confidence" in validation
        assert "replanning_reasoning" in validation
        
        # Successful execution should not trigger replanning
        assert validation["replanning_needed"] is False
    
    def test_validate_with_replanning_assessment_failure_triggers_replanning(self):
        """Test validation with replanning assessment that triggers replanning."""
        validator = ResultValidator()
        
        task = create_test_task("Query doors", "Query door elements", "task-1")
        execution_result = create_test_execution_result(
            "query_elements",
            success=False,
            output=None
        )
        execution_result.error_message = "Query failed: invalid parameters"
        
        execution_context = {
            "goal": "Count all doors in building",
            "current_iteration": 3,
            "recent_execution_steps": [
                {
                    "task": "Query doors",
                    "tool": "query_elements",
                    "success": False,
                    "confidence": 0.1
                },
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
                    "error": "Query failed"
                }
            ],
            "plan_confidence": 0.3
        }
        
        validation = validator.validate_with_replanning_assessment(
            task, execution_result, execution_context
        )
        
        assert isinstance(validation, dict)
        assert validation["success"] is False
        
        # Failed execution with context should trigger replanning
        assert "replanning_needed" in validation
        assert validation["replanning_needed"] is True
        assert validation["replanning_confidence"] > 0.5
        assert "replanning_reasoning" in validation
        assert len(validation["replanning_reasoning"]) > 0
    
    def test_validate_with_replanning_assessment_low_confidence_pattern(self):
        """Test replanning assessment for low confidence pattern."""
        validator = ResultValidator()
        
        task = create_test_task("Analyze data", "Analyze building data", "task-1")
        execution_result = create_test_execution_result(
            "analyze_data",
            success=True,
            output={"analysis": "preliminary results", "confidence": 0.2}
        )
        
        execution_context = {
            "goal": "Analyze building compliance",
            "current_iteration": 2,
            "recent_execution_steps": [
                {
                    "task": "Analyze data",
                    "tool": "analyze_data",
                    "success": True,
                    "confidence": 0.2  # Low confidence
                }
            ],
            "plan_confidence": 0.4
        }
        
        validation = validator.validate_with_replanning_assessment(
            task, execution_result, execution_context
        )
        
        assert isinstance(validation, dict)
        
        # Low confidence might trigger replanning
        assert "replanning_needed" in validation
        if validation["replanning_needed"]:
            assert "low confidence" in validation["replanning_reasoning"].lower()


class TestValidatorOutputValidation:
    """Test output format validation."""
    
    def test_validate_output_format_success(self):
        """Test output format validation for valid output."""
        validator = ResultValidator()
        
        execution_result = create_test_execution_result(
            "load_building_data",
            success=True,
            output={"elements": 150, "building_name": "Test Building"}
        )
        
        validation = validator._validate_output_format(execution_result)
        
        assert isinstance(validation, ValidationResult)
        assert validation.success is True
        assert validation.validation_level == "output"
        assert "format" in validation.message.lower() or "output" in validation.message.lower()
    
    def test_validate_output_format_failure(self):
        """Test output format validation for invalid output."""
        validator = ResultValidator()
        
        execution_result = create_test_execution_result(
            "load_building_data",
            success=False,
            output=None
        )
        execution_result.error_message = "Tool failed"
        
        validation = validator._validate_output_format(execution_result)
        
        assert isinstance(validation, ValidationResult)
        assert validation.success is False
        assert validation.validation_level == "output"
        assert len(validation.details) >= 0


class TestValidatorLogicalValidation:
    """Test logical consistency validation."""
    
    def test_validate_logical_consistency_building_data(self):
        """Test logical validation for building data."""
        validator = ResultValidator()
        
        task = create_test_task("Load building", "Load building data", "task-1")
        execution_result = create_test_execution_result(
            "load_building_data",
            success=True,
            output={
                "elements": 150,
                "doors": 15,
                "windows": 25,
                "building_name": "Test Building"
            }
        )
        
        validation = validator._validate_logical_consistency(execution_result, task)
        
        assert isinstance(validation, ValidationResult)
        assert validation.success is True
        assert validation.validation_level == "logical"
    
    def test_validate_logical_consistency_element_query(self):
        """Test logical validation for element query."""
        validator = ResultValidator()
        
        task = create_test_task("Query doors", "Query door elements", "task-1")
        execution_result = create_test_execution_result(
            "query_elements",
            success=True,
            output={
                "elements": [
                    {"id": "door1", "type": "IfcDoor"},
                    {"id": "door2", "type": "IfcDoor"}
                ],
                "count": 2
            }
        )
        
        validation = validator._validate_logical_consistency(execution_result, task)
        
        assert isinstance(validation, ValidationResult)
        assert validation.success is True
        assert validation.validation_level == "logical"
    
    def test_validate_logical_consistency_calculation(self):
        """Test logical validation for calculation results."""
        validator = ResultValidator()
        
        task = create_test_task("Calculate area", "Calculate total area", "task-1")
        execution_result = create_test_execution_result(
            "calculate_area",
            success=True,
            output={
                "total_area": 1500.5,
                "unit": "square_meters",
                "calculation_method": "sum_of_spaces"
            }
        )
        
        validation = validator._validate_logical_consistency(execution_result, task)
        
        assert isinstance(validation, ValidationResult)
        assert validation.success is True
        assert validation.validation_level == "logical"


class TestValidatorTaskProgress:
    """Test task progress validation."""
    
    def test_validate_task_progress_completed(self):
        """Test progress validation for completed task."""
        validator = ResultValidator()
        
        task = create_test_task("Count doors", "Count door elements", "task-1")
        task.status = TaskStatus.COMPLETED
        
        execution_result = create_test_execution_result(
            "count_elements",
            success=True,
            output={"count": 15, "elements": ["door1", "door2"]}
        )
        
        validation = validator._validate_task_progress(task, execution_result)
        
        assert isinstance(validation, ValidationResult)
        assert validation.success is True
        assert validation.validation_level == "progress"
        assert "progress" in validation.message.lower() or "completed" in validation.message.lower()
    
    def test_validate_task_progress_failed(self):
        """Test progress validation for failed task."""
        validator = ResultValidator()
        
        task = create_test_task("Query doors", "Query door elements", "task-1")
        task.status = TaskStatus.FAILED
        
        execution_result = create_test_execution_result(
            "query_elements",
            success=False,
            output=None
        )
        
        validation = validator._validate_task_progress(task, execution_result)
        
        assert isinstance(validation, ValidationResult)
        assert validation.success is False
        assert validation.validation_level == "progress"


class TestValidatorFallback:
    """Test fallback validation behavior."""
    
    def test_fallback_validate_result_success(self):
        """Test fallback validation for successful result."""
        validator = ResultValidator()
        
        task = create_test_task("Test task", "Test description", "task-1")
        execution_result = create_test_execution_result(
            "test_tool",
            success=True,
            output={"result": "success"}
        )
        
        validation = validator._fallback_validate_result(task, execution_result)
        
        assert isinstance(validation, dict)
        assert validation["success"] is True
        assert validation["method"] == "fallback"
        assert validation["confidence"] > 0.5
    
    def test_fallback_validate_result_failure(self):
        """Test fallback validation for failed result."""
        validator = ResultValidator()
        
        task = create_test_task("Test task", "Test description", "task-1")
        execution_result = create_test_execution_result(
            "test_tool",
            success=False,
            output=None
        )
        execution_result.error_message = "Tool failed"
        
        validation = validator._fallback_validate_result(task, execution_result)
        
        assert isinstance(validation, dict)
        assert validation["success"] is False
        assert validation["method"] == "fallback"
        assert validation["confidence"] <= 0.5
        assert len(validation["issues"]) > 0


class TestValidatorErrorHandling:
    """Test error handling in validation."""
    
    def test_validation_with_llm_failure(self):
        """Test validation when LLM fails."""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM service unavailable")
        
        validator = ResultValidator(llm=mock_llm)
        
        task = create_test_task("Test task", "Test description", "task-1")
        execution_result = create_test_execution_result(
            "test_tool",
            success=True,
            output={"result": "success"}
        )
        
        # Should fall back to rule-based validation
        validation = validator.validate_execution(task, execution_result)
        
        assert isinstance(validation, dict)
        assert validation["method"] == "fallback"
        assert validation["success"] is not None  # Should return some result
    
    def test_validation_with_malformed_output(self):
        """Test validation with malformed execution result output."""
        validator = ResultValidator()
        
        task = create_test_task("Test task", "Test description", "task-1")
        execution_result = create_test_execution_result(
            "test_tool",
            success=True,
            output="This is not a valid dictionary output"  # Wrong type
        )
        
        validation = validator.validate_execution(task, execution_result)
        
        assert isinstance(validation, dict)
        assert validation["success"] is not None  # Should handle gracefully
    
    def test_validation_with_empty_execution_context(self):
        """Test replanning assessment with empty execution context."""
        validator = ResultValidator()
        
        task = create_test_task("Test task", "Test description", "task-1")
        execution_result = create_test_execution_result(
            "test_tool",
            success=True,
            output={"result": "success"}
        )
        
        validation = validator.validate_with_replanning_assessment(
            task, execution_result, {}
        )
        
        assert isinstance(validation, dict)
        assert "replanning_needed" in validation
        assert "replanning_confidence" in validation
        assert "replanning_reasoning" in validation
        
        # Empty context should not trigger replanning  
        assert validation["replanning_needed"] is False


class TestValidationResultDataStructure:
    """Test ValidationResult data structure."""
    
    def test_validation_result_creation(self):
        """Test creating ValidationResult objects."""
        validation = ValidationResult(
            success=True,
            message="Validation passed successfully",
            validation_level="logical",
            details={"score": 0.9, "checks": ["format", "logic"]}
        )
        
        assert validation.success is True
        assert validation.message == "Validation passed successfully"
        assert validation.validation_level == "logical"
        assert validation.details["score"] == 0.9
        assert len(validation.details["checks"]) == 2
    
    def test_validation_result_defaults(self):
        """Test ValidationResult with default values."""
        validation = ValidationResult(
            success=False,
            message="Validation failed",
            validation_level="output"
        )
        
        assert validation.success is False
        assert validation.details == {}  # Default empty dict