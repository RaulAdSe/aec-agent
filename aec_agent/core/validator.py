"""
Result Validator - Validates tool outputs and reasoning progress.

This module provides multi-layer validation of tool execution results
and overall progress toward goals.
"""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

from .reasoning_utils import ReasoningUtils, Task, TaskStatus, ExecutionResult
from .llm_guardrails import default_llm_retry

# Import LangSmith tracing
from langsmith import traceable

# Import LLM components  
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

if TYPE_CHECKING:
    from ..config import AgentConfig


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
    
    def __init__(self, llm=None, config: Optional['AgentConfig'] = None):
        """Initialize the result validator.
        
        Args:
            llm: Optional pre-configured LLM instance (takes precedence)
            config: Optional AgentConfig to use for model configuration
        """
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        # Setup LLM for intelligent validation
        if llm is not None:
            self.llm = llm
        elif config is not None:
            # Use config to create LLM
            self.llm = ChatOpenAI(
                model=config.llm.get_component_model("validator"),
                temperature=config.llm.get_component_temperature("validator"),
                max_tokens=config.llm.get_component_max_tokens("validator")
            )
        else:
            # Fallback to defaults (for backward compatibility)
            self.llm = ChatOpenAI(
                model="gpt-5-mini",
                temperature=0.1,
                max_tokens=1000
            )
    
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
            validation_result = {
                "success": False,
                "message": f"Tool execution failed: {execution_result.error_message}",
                "validation_level": "execution",
                "method": "basic_check"
            }
            # Add replanning trigger assessment for failures
            validation_result.update(self._assess_replanning_trigger(task, execution_result, validation_result))
            return validation_result
        
        # 2. LLM-based intelligent validation ONLY - NO FALLBACKS
        llm_validation = self._llm_validate_result(task, execution_result)
        if llm_validation is not None:
            self.logger.info(f"Used LLM for validation of {task.name}")
            # Add replanning trigger assessment
            llm_validation.update(self._assess_replanning_trigger(task, execution_result, llm_validation))
            return llm_validation
        
        # NO FALLBACKS - Fail explicitly if LLM validation fails
        raise RuntimeError(f"LLM validation failed for task: {task.name}. No fallback mechanisms available.")
    
    @traceable(name="replanning_trigger_assessment")
    def validate_with_replanning_assessment(
        self, 
        task: Task, 
        execution_result: ExecutionResult,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced validation that includes comprehensive replanning trigger assessment.
        
        Args:
            task: Task that was executed
            execution_result: Result of tool execution
            execution_context: Optional execution context for enhanced assessment
            
        Returns:
            Dict containing validation result with replanning recommendations
        """
        # Get basic validation result
        validation_result = self.validate_execution(task, execution_result)
        
        # Enhanced replanning assessment with execution context
        if execution_context:
            enhanced_assessment = self._assess_replanning_with_context(
                task, execution_result, validation_result, execution_context
            )
            validation_result.update(enhanced_assessment)
        
        return validation_result
    
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
            
            # Execute LLM validation with retry guardrail
            chain = validation_prompt | self.llm | StrOutputParser()
            response = self._invoke_llm_with_retry(chain, {
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
    
    @default_llm_retry
    def _invoke_llm_with_retry(self, chain, inputs: Dict[str, Any]) -> str:
        """Invoke LLM chain with retry guardrail."""
        return chain.invoke(inputs)
    
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
    
    def _assess_replanning_trigger(
        self, 
        task: Task, 
        execution_result: ExecutionResult, 
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess whether validation results suggest replanning is needed.
        
        Args:
            task: Task that was executed
            execution_result: Result of tool execution
            validation_result: Current validation result
            
        Returns:
            Dict with replanning assessment fields
        """
        replanning_assessment = {
            "suggests_replanning": False,
            "replanning_confidence": 0.0,
            "replanning_reason": "",
            "replanning_trigger_type": None,
            "replanning_details": {}
        }
        
        # Check for execution failure
        if not execution_result.success:
            replanning_assessment.update({
                "suggests_replanning": True,
                "replanning_confidence": 0.8,
                "replanning_reason": f"Tool execution failed: {execution_result.error_message}",
                "replanning_trigger_type": "execution_error",
                "replanning_details": {
                    "tool_name": execution_result.tool_name,
                    "error_message": execution_result.error_message,
                    "task_name": task.name
                }
            })
            return replanning_assessment
        
        # Check for validation failure
        if not validation_result.get("success", False):
            replanning_assessment.update({
                "suggests_replanning": True,
                "replanning_confidence": 0.7,
                "replanning_reason": f"Validation failed: {validation_result.get('message', 'Unknown reason')}",
                "replanning_trigger_type": "validation_failure",
                "replanning_details": {
                    "validation_message": validation_result.get("message", ""),
                    "validation_method": validation_result.get("method", ""),
                    "task_name": task.name
                }
            })
            return replanning_assessment
        
        # Check for low confidence validation
        confidence = validation_result.get("confidence", 1.0)
        if confidence < 0.6:
            replanning_assessment.update({
                "suggests_replanning": True,
                "replanning_confidence": 0.5,
                "replanning_reason": f"Low validation confidence: {confidence:.1%}",
                "replanning_trigger_type": "low_confidence",
                "replanning_details": {
                    "validation_confidence": confidence,
                    "confidence_threshold": 0.6,
                    "task_name": task.name
                }
            })
            return replanning_assessment
        
        # Check for validation issues
        issues = validation_result.get("issues", [])
        if issues and len(issues) > 2:
            replanning_assessment.update({
                "suggests_replanning": True,
                "replanning_confidence": 0.4,
                "replanning_reason": f"Multiple validation issues: {len(issues)} issues found",
                "replanning_trigger_type": "validation_issues",
                "replanning_details": {
                    "issues": issues[:3],  # Include first 3 issues
                    "total_issues": len(issues),
                    "task_name": task.name
                }
            })
            return replanning_assessment
        
        return replanning_assessment
    
    def _assess_replanning_with_context(
        self,
        task: Task,
        execution_result: ExecutionResult,
        validation_result: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced replanning assessment using execution context.
        
        Args:
            task: Task that was executed
            execution_result: Result of tool execution
            validation_result: Current validation result
            execution_context: Rich execution context from memory
            
        Returns:
            Dict with enhanced replanning assessment
        """
        enhanced_assessment = {
            "context_suggests_replanning": False,
            "context_confidence": 0.0,
            "context_reason": "",
            "context_trigger_type": None,
            "context_details": {}
        }
        
        # Check for patterns in recent failures
        recent_failures = execution_context.get("recent_failures", [])
        if len(recent_failures) >= 2:
            # Multiple recent failures suggest systematic issue
            enhanced_assessment.update({
                "context_suggests_replanning": True,
                "context_confidence": 0.8,
                "context_reason": f"Pattern of recent failures: {len(recent_failures)} failures",
                "context_trigger_type": "failure_pattern",
                "context_details": {
                    "recent_failures": recent_failures,
                    "failure_count": len(recent_failures)
                }
            })
            return enhanced_assessment
        
        # Check for low plan confidence
        plan_confidence = execution_context.get("plan_confidence", 1.0)
        if plan_confidence < 0.5:
            enhanced_assessment.update({
                "context_suggests_replanning": True,
                "context_confidence": 0.7,
                "context_reason": f"Low plan confidence: {plan_confidence:.1%}",
                "context_trigger_type": "low_plan_confidence",
                "context_details": {
                    "plan_confidence": plan_confidence,
                    "threshold": 0.5
                }
            })
            return enhanced_assessment
        
        # Check for significant context discoveries
        recent_discoveries = execution_context.get("recent_context_discoveries", [])
        high_confidence_discoveries = [d for d in recent_discoveries if d.get("confidence", 0) > 0.8]
        
        if len(high_confidence_discoveries) >= 2:
            enhanced_assessment.update({
                "context_suggests_replanning": True,
                "context_confidence": 0.6,
                "context_reason": f"New context discovered: {len(high_confidence_discoveries)} high-confidence discoveries",
                "context_trigger_type": "context_discovery",
                "context_details": {
                    "discoveries": high_confidence_discoveries,
                    "discovery_count": len(high_confidence_discoveries)
                }
            })
            return enhanced_assessment
        
        # Check for stagnation (no progress over multiple steps)
        recent_steps = execution_context.get("recent_execution_steps", [])
        if len(recent_steps) >= 3:
            # Check if recent steps are all on the same failing task
            task_names = [step.get("task", "") for step in recent_steps]
            if len(set(task_names)) == 1 and not any(step.get("success", False) for step in recent_steps):
                enhanced_assessment.update({
                    "context_suggests_replanning": True,
                    "context_confidence": 0.6,
                    "context_reason": "Stagnation: Multiple attempts at same task failing",
                    "context_trigger_type": "task_stagnation",
                    "context_details": {
                        "stagnant_task": task_names[0],
                        "failed_attempts": len(recent_steps)
                    }
                })
                return enhanced_assessment
        
        return enhanced_assessment
    
    def should_trigger_replanning(self, validation_result: Dict[str, Any]) -> bool:
        """
        Determine if validation result suggests replanning should be triggered.
        
        Args:
            validation_result: Complete validation result with replanning assessment
            
        Returns:
            True if replanning should be triggered
        """
        # Check basic replanning suggestion
        basic_suggestion = validation_result.get("suggests_replanning", False)
        basic_confidence = validation_result.get("replanning_confidence", 0.0)
        
        # Check context-based replanning suggestion
        context_suggestion = validation_result.get("context_suggests_replanning", False)
        context_confidence = validation_result.get("context_confidence", 0.0)
        
        # Trigger replanning if either basic or context assessment suggests it with sufficient confidence
        should_replan = (
            (basic_suggestion and basic_confidence >= 0.6) or
            (context_suggestion and context_confidence >= 0.6)
        )
        
        if should_replan:
            trigger_type = validation_result.get("replanning_trigger_type") or validation_result.get("context_trigger_type")
            self.logger.info(f"Validation suggests replanning: {trigger_type}")
        
        return should_replan
    
    def get_replanning_trigger_details(self, validation_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract detailed replanning trigger information from validation result.
        
        Args:
            validation_result: Complete validation result with replanning assessment
            
        Returns:
            Trigger details if replanning is suggested, None otherwise
        """
        if not self.should_trigger_replanning(validation_result):
            return None
        
        # Prefer context-based trigger if available and confident
        if (validation_result.get("context_suggests_replanning", False) and 
            validation_result.get("context_confidence", 0.0) >= 0.6):
            return {
                "trigger_type": validation_result.get("context_trigger_type"),
                "confidence": validation_result.get("context_confidence"),
                "reason": validation_result.get("context_reason"),
                "details": validation_result.get("context_details", {})
            }
        
        # Fall back to basic trigger
        if (validation_result.get("suggests_replanning", False) and 
            validation_result.get("replanning_confidence", 0.0) >= 0.6):
            return {
                "trigger_type": validation_result.get("replanning_trigger_type"),
                "confidence": validation_result.get("replanning_confidence"),
                "reason": validation_result.get("replanning_reason"),
                "details": validation_result.get("replanning_details", {})
            }
        
        return None