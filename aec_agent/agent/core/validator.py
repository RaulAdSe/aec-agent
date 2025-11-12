"""
Result Validator - Validates tool outputs and reasoning progress.

This module provides multi-layer validation of tool execution results
and overall progress toward goals.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .utils import ReasoningUtils, Task, TaskStatus, ExecutionResult


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
    4. Goal validation - Check if goal requirements are satisfied
    """
    
    def __init__(self):
        """Initialize the result validator."""
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        # Expected output patterns for different tools
        self.output_patterns = {
            "load_building_data": {
                "required_fields": ["status", "data"],
                "success_indicators": ["total_elements", "file_info"],
                "logical_checks": ["element_count_positive"]
            },
            "get_all_elements": {
                "required_fields": ["status"],
                "success_indicators": ["elements", "data"],
                "logical_checks": ["non_empty_elements"]
            },
            "query_elements": {
                "required_fields": ["status"],
                "success_indicators": ["elements", "data"], 
                "logical_checks": ["reasonable_element_count"]
            },
            "calculate_metrics": {
                "required_fields": ["status", "result"],
                "success_indicators": ["calculations"],
                "logical_checks": ["positive_numeric_results"]
            },
            "validate_compliance_rule": {
                "required_fields": ["status", "validation_results"],
                "success_indicators": ["compliance_status"],
                "logical_checks": ["boolean_compliance_status"]
            },
            "search_compliance_documents": {
                "required_fields": ["status"],
                "success_indicators": ["documents", "results"],
                "logical_checks": ["relevant_documents_found"]
            }
        }
    
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
        tool_name = execution_result.tool_name
        output = execution_result.output
        
        # Get expected pattern for this tool
        pattern = self.output_patterns.get(tool_name, {})
        
        if not isinstance(output, dict):
            return ValidationResult(
                success=False,
                message=f"Expected dict output, got {type(output).__name__}",
                validation_level="output",
                details={"expected_type": "dict", "actual_type": type(output).__name__}
            )
        
        # Check required fields
        required_fields = pattern.get("required_fields", ["status"])
        missing_fields = [field for field in required_fields if field not in output]
        
        if missing_fields:
            return ValidationResult(
                success=False,
                message=f"Missing required fields: {missing_fields}",
                validation_level="output",
                details={"missing_fields": missing_fields}
            )
        
        # Check success indicators (if status is success)
        if output.get("status") == "success":
            success_indicators = pattern.get("success_indicators", [])
            missing_indicators = [
                field for field in success_indicators 
                if field not in output or not output[field]
            ]
            
            if missing_indicators:
                return ValidationResult(
                    success=False,
                    message=f"Missing success indicators: {missing_indicators}",
                    validation_level="output",
                    details={"missing_indicators": missing_indicators}
                )
        
        return ValidationResult(
            success=True,
            message="Output format validation passed",
            validation_level="output",
            details={"validated_fields": required_fields}
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
            elif tool_name == "validate_compliance_rule":
                return self._validate_compliance_logic(output)
            elif tool_name == "search_compliance_documents":
                return self._validate_search_logic(output)
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
        
        # Check if we have some element types
        element_types = ["spaces", "doors", "walls", "slabs", "stairs"]
        found_types = [t for t in element_types if t in data and len(data[t]) > 0]
        
        if not found_types:
            return ValidationResult(
                success=False,
                message="No recognizable element types found in building data",
                validation_level="logical",
                details={"checked_types": element_types}
            )
        
        return ValidationResult(
            success=True,
            message=f"Building data is logically consistent ({total_elements} elements, {len(found_types)} types)",
            validation_level="logical",
            details={"total_elements": total_elements, "element_types": found_types}
        )
    
    def _validate_element_query_logic(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate element query results."""
        # Check for elements in data field
        elements = output.get("elements") or output.get("data", [])
        
        if not isinstance(elements, list):
            return ValidationResult(
                success=False,
                message="Elements should be a list",
                validation_level="logical",
                details={"elements_type": type(elements).__name__}
            )
        
        # Check if element count is reasonable (not too high)
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
    
    def _validate_compliance_logic(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate compliance validation results."""
        validation_results = output.get("validation_results", {})
        
        if not isinstance(validation_results, dict):
            return ValidationResult(
                success=False,
                message="Validation results should be a dictionary",
                validation_level="logical"
            )
        
        # Check for compliance status
        if "compliance_status" in validation_results:
            status = validation_results["compliance_status"]
            if status not in ["compliant", "non_compliant", "partial", "unknown"]:
                return ValidationResult(
                    success=False,
                    message=f"Invalid compliance status: {status}",
                    validation_level="logical",
                    details={"status": status}
                )
        
        return ValidationResult(
            success=True,
            message="Compliance validation results are logical",
            validation_level="logical"
        )
    
    def _validate_search_logic(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate document search results."""
        results = output.get("results") or output.get("documents", [])
        
        if not isinstance(results, list):
            return ValidationResult(
                success=False,
                message="Search results should be a list",
                validation_level="logical"
            )
        
        # Check if we have some results for successful search
        if len(results) == 0:
            return ValidationResult(
                success=False,
                message="No search results found - may indicate query issues",
                validation_level="logical",
                details={"result_count": 0}
            )
        
        return ValidationResult(
            success=True,
            message=f"Search results are logical ({len(results)} documents found)",
            validation_level="logical",
            details={"result_count": len(results)}
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
        
        elif tool_name == "calculate_metrics" and isinstance(output, dict):
            if output.get("status") == "success" and "result" in output:
                return ValidationResult(
                    success=True,
                    message="Calculations completed - progress toward task completion",
                    validation_level="progress"
                )
        
        elif tool_name == "search_compliance_documents" and isinstance(output, dict):
            results = output.get("results") or output.get("documents", [])
            if isinstance(results, list) and len(results) > 0:
                return ValidationResult(
                    success=True,
                    message=f"Found {len(results)} compliance documents - progress toward validation",
                    validation_level="progress"
                )
        
        # Default: if tool executed successfully, assume some progress
        return ValidationResult(
            success=True,
            message="Tool executed successfully - assuming progress made",
            validation_level="progress"
        )
    
    def validate_goal_achievement(self, tasks: List[Task], goal: str) -> ValidationResult:
        """Validate whether the overall goal has been achieved."""
        
        completed_tasks = [task for task in tasks if task.status == TaskStatus.COMPLETED]
        total_tasks = len(tasks)
        completion_rate = len(completed_tasks) / total_tasks if total_tasks > 0 else 0
        
        if completion_rate >= 1.0:
            return ValidationResult(
                success=True,
                message=f"Goal achieved: All {total_tasks} tasks completed",
                validation_level="goal",
                details={
                    "completed_tasks": len(completed_tasks),
                    "total_tasks": total_tasks,
                    "completion_rate": completion_rate
                }
            )
        elif completion_rate >= 0.8:
            return ValidationResult(
                success=True,
                message=f"Goal substantially achieved: {len(completed_tasks)}/{total_tasks} tasks completed",
                validation_level="goal",
                details={
                    "completed_tasks": len(completed_tasks),
                    "total_tasks": total_tasks,
                    "completion_rate": completion_rate
                }
            )
        else:
            failed_tasks = [task for task in tasks if task.status == TaskStatus.FAILED]
            return ValidationResult(
                success=False,
                message=f"Goal not achieved: Only {len(completed_tasks)}/{total_tasks} tasks completed, {len(failed_tasks)} failed",
                validation_level="goal",
                details={
                    "completed_tasks": len(completed_tasks),
                    "total_tasks": total_tasks,
                    "failed_tasks": len(failed_tasks),
                    "completion_rate": completion_rate
                }
            )