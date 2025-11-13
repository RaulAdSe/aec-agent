"""
Result Validator - Validates tool outputs and reasoning progress.

This module provides multi-layer validation of tool execution results
and overall progress toward goals.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .reasoning_utils import ReasoningUtils, Task, TaskStatus, ExecutionResult

# Import LangSmith tracing
from langsmith import traceable

# Import LLM components  
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import json


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
    
    def __init__(self, llm=None):
        """Initialize the result validator."""
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        # Setup LLM for intelligent validation
        if llm is None:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1000
            )
        else:
            self.llm = llm
    
    @traceable(name="result_validation", metadata={"component": "validator"})
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
        
        # 1. Basic execution check
        if not execution_result.success:
            return {
                "success": False,
                "message": f"Tool execution failed: {execution_result.error_message}",
                "validation_level": "execution",
                "method": "basic_check"
            }
        
        # 2. LLM-based intelligent validation
        llm_validation = self._llm_validate_result(task, execution_result)
        if llm_validation is not None:
            self.logger.info(f"Used LLM for validation of {task.name}")
            return llm_validation
        
        # 3. Fallback to rule-based validation  
        fallback_validation = self._fallback_validate_result(task, execution_result)
        self.logger.info(f"Used fallback validation for {task.name}")
        return fallback_validation
    
    @traceable(name="llm_result_validation")
    def _llm_validate_result(self, task: Task, execution_result: ExecutionResult) -> Optional[Dict[str, Any]]:
        """Use LLM to intelligently validate task execution results."""
        
        # Create validation prompt
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert validator for AEC (Architecture, Engineering, Construction) compliance analysis tools.

Your job is to validate whether a tool execution successfully accomplished its intended task.

Tool execution details:
- Tool: {tool_name}
- Task: {task_name}
- Task Description: {task_description}
- Execution Time: {execution_time}s
- Output Status: {output_status}
- Output Data Available: {has_output_data}

Output Preview: {output_preview}

Validation criteria:
1. Did the tool execute without errors?
2. Does the output format match what's expected for this tool?
3. Is the result logically consistent? (e.g., counts should be non-negative numbers)
4. Does this result advance progress toward completing the task?

Return ONLY a JSON response with:
{{
  "success": true/false,
  "message": "Brief explanation of validation result",
  "confidence": 0.0-1.0,
  "issues": ["list of any issues found"]
}}

Be strict but reasonable. If tool ran successfully and produced expected output format, it's likely valid."""),
            ("human", "Validate this tool execution result:")
        ])
        
        try:
            # Prepare output preview
            output = execution_result.output
            if isinstance(output, dict):
                output_preview = str(output)[:500] + ("..." if len(str(output)) > 500 else "")
                output_status = output.get("status", "unknown")
                has_output_data = "data" in output or "result" in output or len(output) > 1
            else:
                output_preview = str(output)[:500] if output else "No output"
                output_status = "unknown"
                has_output_data = output is not None
            
            # Execute LLM validation
            chain = validation_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "tool_name": execution_result.tool_name,
                "task_name": task.name,
                "task_description": task.description,
                "execution_time": execution_result.execution_time,
                "output_status": output_status,
                "has_output_data": has_output_data,
                "output_preview": output_preview
            })
            
            # Parse JSON response
            validation_result = json.loads(response.strip())
            
            if not isinstance(validation_result, dict):
                self.logger.warning("LLM returned invalid validation format")
                return None
            
            # Convert to our format
            return {
                "success": validation_result.get("success", False),
                "message": validation_result.get("message", "LLM validation completed"),
                "validation_level": "llm_intelligent",
                "method": "llm_reasoning",
                "confidence": validation_result.get("confidence", 0.0),
                "issues": validation_result.get("issues", [])
            }
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"LLM validation response was not valid JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"LLM validation failed for {task.name}: {e}")
            return None
    
    def _fallback_validate_result(self, task: Task, execution_result: ExecutionResult) -> Dict[str, Any]:
        """Fallback rule-based validation when LLM validation fails."""
        
        # Basic output format check
        output = execution_result.output
        if not isinstance(output, dict):
            return {
                "success": False,
                "message": "Tool output is not in expected dictionary format",
                "validation_level": "format",
                "method": "rule_based_fallback"
            }
        
        # Check for status field
        status = output.get("status")
        if status not in ["success", "error", "partial"]:
            return {
                "success": False,
                "message": f"Tool output has invalid status: {status}",
                "validation_level": "status",
                "method": "rule_based_fallback"
            }
        
        # If status is success, validate basic output
        if status == "success":
            return {
                "success": True,
                "message": "Basic validation passed - tool executed successfully",
                "validation_level": "basic",
                "method": "rule_based_fallback"
            }
        else:
            return {
                "success": False,
                "message": f"Tool execution status was: {status}",
                "validation_level": "status",
                "method": "rule_based_fallback"
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
        
        if not data or not isinstance(data, dict):
            return ValidationResult(
                success=False,
                message="Building data is missing or invalid",
                validation_level="logical",
                details={"data_type": type(data)}
            )
        
        # Try to get total elements from file_info first, then count manually
        file_info = data.get("file_info", {})
        total_elements = file_info.get("total_elements", 0)
        
        # If no file_info, count elements manually from data structure
        if total_elements == 0 or total_elements == "unknown":
            # Count elements from various possible keys
            element_count = 0
            if "elements" in data:
                element_count = len(data["elements"])
            else:
                # Count from typical IFC element types
                for element_type in ["spaces", "doors", "walls", "slabs", "stairs", "columns", "beams"]:
                    if element_type in data:
                        element_count += len(data[element_type])
            total_elements = element_count
        
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
        
        # No default assumption - require explicit progress validation for each tool
        return ValidationResult(
            success=False,
            message=f"No specific progress validation defined for tool: {tool_name}",
            validation_level="progress"
        )