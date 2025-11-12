"""
Result Validator - Validates tool outputs and reasoning progress.

This module provides multi-layer validation of tool execution results
and overall progress toward goals.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .reasoning_utils import ReasoningUtils, Task, TaskStatus, ExecutionResult


@dataclass
class ValidationResult:
    """Result of validation checks."""
    success: bool
    message: str
    validation_level: str  # "output", "logical", "progress", "goal"
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ResultValidator:
    """
    Validates tool execution results and reasoning progress.
    
    Provides multiple validation layers:
    1. Output validation - Check tool output format and completeness
    2. Logical validation - Verify results make logical sense
    3. Progress validation - Confirm steps advance toward goal
    """
    
    def __init__(self):
        """Initialize the result validator."""
        self.logger = ReasoningUtils.setup_logger(__name__)
    
    def validate_execution(self, task: Task, execution_result: ExecutionResult) -> Dict[str, Any]:
        """
        Validate the execution result for a task.
        
        Args:
            task: Task that was executed
            execution_result: Result of tool execution
            
        Returns:
            Dict containing overall validation result and detailed checks
        """
        self.logger.info(f"Validating execution of {execution_result.tool_name} for task {task.name}")
        
        validation_results = []
        overall_success = True
        
        # 1. Basic execution validation
        if not execution_result.success:
            validation_results.append(ValidationResult(
                success=False,
                message=f"Tool execution failed: {execution_result.error_message}",
                validation_level="execution"
            ))
            overall_success = False
        else:
            # 2. Output format validation
            output_validation = self._validate_output_format(execution_result)
            validation_results.append(output_validation)
            if not output_validation.success:
                overall_success = False
            
            # 3. Logical consistency validation
            logical_validation = self._validate_logical_consistency(execution_result, task)
            validation_results.append(logical_validation)
            if not logical_validation.success:
                overall_success = False
            
            # 4. Progress validation
            progress_validation = self._validate_task_progress(task, execution_result)
            validation_results.append(progress_validation)
            if not progress_validation.success:
                overall_success = False
        
        # Create summary
        if overall_success:
            message = f"All validations passed for {execution_result.tool_name}"
        else:
            failed_checks = [v.validation_level for v in validation_results if not v.success]
            message = f"Validation failed at: {', '.join(failed_checks)}"
        
        return {
            "success": overall_success,
            "message": message,
            "validation_results": [
                {
                    "level": v.validation_level,
                    "success": v.success,
                    "message": v.message,
                    "details": v.details
                }
                for v in validation_results
            ]
        }
    
    def _validate_output_format(self, execution_result: ExecutionResult) -> ValidationResult:
        """Validate the format and structure of tool output."""
        output = execution_result.output
        
        if not isinstance(output, dict):
            return ValidationResult(
                success=False,
                message=f"Expected dict output, got {type(output).__name__}",
                validation_level="output",
                details={"expected_type": "dict", "actual_type": type(output).__name__}
            )
        
        # Check required fields
        if "status" not in output:
            return ValidationResult(
                success=False,
                message="Missing required 'status' field",
                validation_level="output",
                details={"missing_fields": ["status"]}
            )
        
        return ValidationResult(
            success=True,
            message="Output format validation passed",
            validation_level="output"
        )
    
    def _validate_logical_consistency(self, execution_result: ExecutionResult, task: Task) -> ValidationResult:
        """Validate logical consistency of the results."""
        tool_name = execution_result.tool_name
        output = execution_result.output
        
        if not isinstance(output, dict) or output.get("status") != "success":
            # Skip logical validation for failed outputs
            return ValidationResult(
                success=True,
                message="Skipped logical validation (output not successful)",
                validation_level="logical"
            )
        
        # Tool-specific logical checks
        try:
            if tool_name == "load_building_data":
                return self._validate_building_data_logic(output)
            elif tool_name in ["get_all_elements", "query_elements"]:
                return self._validate_element_query_logic(output)
            elif tool_name == "calculate_metrics":
                return self._validate_calculation_logic(output)
            else:
                return ValidationResult(
                    success=True,
                    message="No specific logical validation for this tool",
                    validation_level="logical"
                )
                
        except Exception as e:
            return ValidationResult(
                success=False,
                message=f"Logical validation error: {str(e)}",
                validation_level="logical",
                details={"error": str(e)}
            )
    
    def _validate_building_data_logic(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate building data loading results."""
        data = output.get("data", {})
        file_info = data.get("file_info", {})
        
        # Check if total elements is reasonable
        total_elements = file_info.get("total_elements", 0)
        if total_elements <= 0:
            return ValidationResult(
                success=False,
                message="Building data shows no elements",
                validation_level="logical",
                details={"total_elements": total_elements}
            )
        
        return ValidationResult(
            success=True,
            message=f"Building data is logically consistent ({total_elements} elements)",
            validation_level="logical",
            details={"total_elements": total_elements}
        )
    
    def _validate_element_query_logic(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate element query results."""
        elements = output.get("elements") or output.get("data", [])
        
        if not isinstance(elements, list):
            return ValidationResult(
                success=False,
                message="Elements should be a list",
                validation_level="logical",
                details={"elements_type": type(elements).__name__}
            )
        
        # Check if element count is reasonable
        if len(elements) > 10000:
            return ValidationResult(
                success=False,
                message=f"Element count seems too high: {len(elements)}",
                validation_level="logical",
                details={"element_count": len(elements)}
            )
        
        return ValidationResult(
            success=True,
            message=f"Element query results are logical ({len(elements)} elements)",
            validation_level="logical",
            details={"element_count": len(elements)}
        )
    
    def _validate_calculation_logic(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate calculation results."""
        result = output.get("result") or output.get("data", {})
        
        if isinstance(result, dict):
            # Check for numeric values
            numeric_fields = []
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    numeric_fields.append(key)
                    # Check for reasonable ranges
                    if value < 0 and key in ["area", "volume", "count", "length", "width", "height"]:
                        return ValidationResult(
                            success=False,
                            message=f"Negative value for {key}: {value}",
                            validation_level="logical",
                            details={"field": key, "value": value}
                        )
            
            if not numeric_fields:
                return ValidationResult(
                    success=False,
                    message="No numeric results found in calculation output",
                    validation_level="logical",
                    details={"result_keys": list(result.keys())}
                )
        
        return ValidationResult(
            success=True,
            message="Calculation results are logically consistent",
            validation_level="logical"
        )
    
    def _validate_task_progress(self, task: Task, execution_result: ExecutionResult) -> ValidationResult:
        """Validate that execution made progress toward completing the task."""
        
        # If execution failed, no progress was made
        if not execution_result.success:
            return ValidationResult(
                success=False,
                message="No progress made due to execution failure",
                validation_level="progress"
            )
        
        tool_name = execution_result.tool_name
        output = execution_result.output
        
        # Tool-specific progress checks
        if tool_name == "load_building_data" and isinstance(output, dict):
            if output.get("status") == "success" and output.get("data"):
                return ValidationResult(
                    success=True,
                    message="Building data successfully loaded - enables further analysis",
                    validation_level="progress"
                )
        
        elif tool_name in ["get_all_elements", "query_elements"] and isinstance(output, dict):
            elements = output.get("elements") or output.get("data", [])
            if isinstance(elements, list) and len(elements) > 0:
                return ValidationResult(
                    success=True,
                    message=f"Found {len(elements)} elements - progress toward task completion",
                    validation_level="progress"
                )
        
        # Default: if tool executed successfully, assume some progress
        return ValidationResult(
            success=True,
            message="Tool executed successfully - assuming progress made",
            validation_level="progress"
        )