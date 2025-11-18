"""
Recovery System - Intelligent error handling and task recovery for autonomous reasoning.

This module provides comprehensive recovery mechanisms when tasks fail:
1. Tool failure recovery with error analysis and retry logic
2. Task replanning when multiple failures indicate bad planning  
3. Graceful degradation for non-critical failures
4. Learning from failures to avoid repetition
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .reasoning_utils import Task, TaskStatus, ExecutionResult, ReasoningUtils
from .goal_decomposer import GoalDecomposer
from .tool_planner import ToolPlanner

# Import LangSmith tracing
from langsmith import traceable


class FailureType(Enum):
    """Types of task failures for recovery classification."""
    TOOL_EXECUTION_ERROR = "tool_execution_error"      # Tool failed to run
    PARAMETER_ERROR = "parameter_error"                # Wrong parameters passed
    DEPENDENCY_MISSING = "dependency_missing"          # Required data not available  
    TOOL_SELECTION_ERROR = "tool_selection_error"     # Wrong tool selected
    PLANNING_ERROR = "planning_error"                  # Bad task decomposition
    RESOURCE_ERROR = "resource_error"                  # External resource unavailable


@dataclass
class FailureAnalysis:
    """Analysis of a task failure for recovery planning."""
    failure_type: FailureType
    error_message: str
    task: Task
    can_retry: bool
    suggested_action: str
    alternative_tools: List[str] = field(default_factory=list)
    parameter_adjustments: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # How confident we are in the analysis


@dataclass 
class RecoveryAttempt:
    """Record of a recovery attempt for learning and avoiding loops."""
    task_id: str
    failure_type: FailureType
    recovery_action: str
    timestamp: float
    success: bool
    error_context: str


class RecoverySystem:
    """
    Intelligent recovery system for handling task failures.
    
    Provides multiple recovery strategies:
    - Tool failure recovery with error analysis
    - Parameter adjustment and alternative tool selection
    - Task replanning for fundamental planning errors
    - Graceful degradation for non-critical failures
    """
    
    def __init__(
        self,
        goal_decomposer: GoalDecomposer,
        tool_planner: ToolPlanner,
        llm,  # LLM for error analysis and recovery decisions
        max_retries: int = 2,
        max_replanning_attempts: int = 1
    ):
        """Initialize recovery system with configurable retry limits."""
        self.goal_decomposer = goal_decomposer
        self.tool_planner = tool_planner
        self.llm = llm
        self.max_retries = max_retries
        self.max_replanning_attempts = max_replanning_attempts
        
        # Track recovery attempts to avoid infinite loops
        self.recovery_attempts: List[RecoveryAttempt] = []
        self.replanning_attempts = 0
        
        self.logger = ReasoningUtils.setup_logger(__name__)
    
    @traceable(name="analyze_task_failure")
    def analyze_failure(self, task: Task, error_result: ExecutionResult, context: Dict[str, Any]) -> FailureAnalysis:
        """
        Analyze why a task failed to determine the best recovery strategy.
        
        Args:
            task: The failed task
            error_result: Execution result containing error information
            context: Current execution context
            
        Returns:
            FailureAnalysis with recovery recommendations
        """
        self.logger.info(f"Analyzing failure for task: {task.name}")
        
        error_message = error_result.error_message or "Unknown error"
        
        # First, try deterministic classification based on error patterns
        failure_analysis = self._classify_failure_deterministically(task, error_message, context)
        
        # If deterministic classification is uncertain, use LLM analysis
        if failure_analysis.confidence < 0.7:
            failure_analysis = self._analyze_failure_with_llm(task, error_message, context)
        
        self.logger.info(f"Failure analysis: {failure_analysis.failure_type.value} - {failure_analysis.suggested_action}")
        return failure_analysis
    
    def _classify_failure_deterministically(self, task: Task, error_message: str, context: Dict[str, Any]) -> FailureAnalysis:
        """Classify failure using deterministic error pattern matching."""
        error_lower = error_message.lower()
        
        # Parameter/input errors
        if any(pattern in error_lower for pattern in [
            "no input preparation method", "element_id_required", "invalid parameter",
            "missing required", "parameter error", "json", "decode"
        ]):
            return FailureAnalysis(
                failure_type=FailureType.PARAMETER_ERROR,
                error_message=error_message,
                task=task,
                can_retry=True,
                suggested_action="Adjust parameters and retry",
                confidence=0.9
            )
        
        # Tool selection errors  
        if any(pattern in error_lower for pattern in [
            "tool not found", "unknown tool", "invalid tool", "tool selection"
        ]):
            return FailureAnalysis(
                failure_type=FailureType.TOOL_SELECTION_ERROR,
                error_message=error_message,
                task=task,
                can_retry=True,
                suggested_action="Select alternative tool",
                alternative_tools=self._suggest_alternative_tools(task),
                confidence=0.8
            )
        
        # Dependency/data missing errors
        if any(pattern in error_lower for pattern in [
            "building data not loaded", "no data available", "dependency", "blocked"
        ]):
            return FailureAnalysis(
                failure_type=FailureType.DEPENDENCY_MISSING,
                error_message=error_message,
                task=task,
                can_retry=True,
                suggested_action="Ensure dependencies are satisfied",
                confidence=0.8
            )
        
        # Resource/external errors
        if any(pattern in error_lower for pattern in [
            "connection", "timeout", "network", "file not found", "permission"
        ]):
            return FailureAnalysis(
                failure_type=FailureType.RESOURCE_ERROR,
                error_message=error_message,
                task=task,
                can_retry=True,
                suggested_action="Retry after brief delay",
                confidence=0.7
            )
        
        # Default to tool execution error with low confidence
        return FailureAnalysis(
            failure_type=FailureType.TOOL_EXECUTION_ERROR,
            error_message=error_message,
            task=task,
            can_retry=False,
            suggested_action="Analyze with LLM for specific recovery strategy",
            confidence=0.3
        )
    
    @traceable(name="llm_failure_analysis")
    def _analyze_failure_with_llm(self, task: Task, error_message: str, context: Dict[str, Any]) -> FailureAnalysis:
        """Use LLM to analyze complex failure cases and suggest recovery strategies."""
        
        # Create analysis prompt
        analysis_prompt = f"""Analyze this task failure and suggest a recovery strategy:

Task: {task.name}
Description: {task.description}
Error: {error_message}
Context: {context.get('building_data_loaded', False)}, {len(context.get('doors_data', []))} doors, {len(context.get('stairs_data', []))} stairs

Available Tools: load_building_data, get_all_elements, get_all_properties, query_elements, calculate, find_related, validate_rule

Classify the failure type and suggest recovery:
1. TOOL_EXECUTION_ERROR - tool failed to run properly
2. PARAMETER_ERROR - wrong parameters provided  
3. DEPENDENCY_MISSING - required data not available
4. TOOL_SELECTION_ERROR - wrong tool chosen
5. PLANNING_ERROR - task breakdown was wrong
6. RESOURCE_ERROR - external resource issue

Respond in this format:
FAILURE_TYPE: [one of the above]
CAN_RETRY: [yes/no]
SUGGESTED_ACTION: [specific recovery action]
ALTERNATIVE_TOOLS: [comma-separated list if applicable]
CONFIDENCE: [0.0-1.0]"""
        
        try:
            response = self.llm.invoke(analysis_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse LLM response
            lines = content.strip().split('\n')
            analysis_data = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    analysis_data[key.strip()] = value.strip()
            
            # Map response to FailureAnalysis
            failure_type_str = analysis_data.get('FAILURE_TYPE', 'TOOL_EXECUTION_ERROR')
            failure_type = FailureType(failure_type_str.lower()) if failure_type_str.lower() in [ft.value for ft in FailureType] else FailureType.TOOL_EXECUTION_ERROR
            
            can_retry = analysis_data.get('CAN_RETRY', 'no').lower() == 'yes'
            suggested_action = analysis_data.get('SUGGESTED_ACTION', 'Manual intervention required')
            alternative_tools_str = analysis_data.get('ALTERNATIVE_TOOLS', '')
            alternative_tools = [tool.strip() for tool in alternative_tools_str.split(',') if tool.strip()]
            confidence = float(analysis_data.get('CONFIDENCE', '0.5'))
            
            return FailureAnalysis(
                failure_type=failure_type,
                error_message=error_message,
                task=task,
                can_retry=can_retry,
                suggested_action=suggested_action,
                alternative_tools=alternative_tools,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"LLM failure analysis failed: {e}")
            
            # Fallback to conservative analysis
            return FailureAnalysis(
                failure_type=FailureType.TOOL_EXECUTION_ERROR,
                error_message=error_message,
                task=task,
                can_retry=False,
                suggested_action="Manual review required",
                confidence=0.2
            )
    
    def _suggest_alternative_tools(self, task: Task) -> List[str]:
        """Suggest alternative tools that might accomplish the same task."""
        task_desc_lower = task.description.lower()
        
        alternatives = []
        
        # Distance/calculation alternatives
        if any(word in task_desc_lower for word in ["distance", "calculate", "measure"]):
            alternatives.extend(["calculate", "find_related"])
        
        # Element retrieval alternatives  
        if any(word in task_desc_lower for word in ["get", "find", "retrieve", "elements"]):
            alternatives.extend(["get_all_elements", "query_elements"])
        
        # Properties alternatives
        if any(word in task_desc_lower for word in ["properties", "characteristics", "attributes"]):
            alternatives.extend(["get_all_properties", "get_all_elements"])
        
        return list(set(alternatives))  # Remove duplicates
    
    @traceable(name="attempt_task_recovery", metadata={"component": "recovery_system"})
    def attempt_recovery(self, task: Task, failure_analysis: FailureAnalysis, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Attempt to recover from task failure based on analysis.
        
        Returns:
            Recovery result dict if successful, None if recovery not possible
        """
        # Check if we've already tried too many times
        task_attempts = [attempt for attempt in self.recovery_attempts if attempt.task_id == task.id]
        if len(task_attempts) >= self.max_retries:
            self.logger.warning(f"Max retries ({self.max_retries}) exceeded for task: {task.name}")
            return None
        
        # Record this recovery attempt
        attempt = RecoveryAttempt(
            task_id=task.id,
            failure_type=failure_analysis.failure_type,
            recovery_action=failure_analysis.suggested_action,
            timestamp=time.time(),
            success=False,  # Will update if successful
            error_context=failure_analysis.error_message
        )
        self.recovery_attempts.append(attempt)
        
        self.logger.info(f"Attempting recovery for {task.name}: {failure_analysis.suggested_action}")
        
        # Choose recovery strategy based on failure type
        if failure_analysis.failure_type == FailureType.PARAMETER_ERROR:
            return self._recover_parameter_error(task, failure_analysis, context)
        
        elif failure_analysis.failure_type == FailureType.TOOL_SELECTION_ERROR:
            return self._recover_tool_selection_error(task, failure_analysis, context)
        
        elif failure_analysis.failure_type == FailureType.DEPENDENCY_MISSING:
            return self._recover_dependency_error(task, failure_analysis, context)
        
        elif failure_analysis.failure_type == FailureType.RESOURCE_ERROR:
            return self._recover_resource_error(task, failure_analysis, context)
        
        elif failure_analysis.failure_type == FailureType.PLANNING_ERROR:
            return self._recover_planning_error(task, failure_analysis, context)
        
        else:
            self.logger.warning(f"No recovery strategy for failure type: {failure_analysis.failure_type}")
            return None
    
    def _recover_parameter_error(self, task: Task, analysis: FailureAnalysis, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recover from parameter errors by adjusting inputs."""
        self.logger.info(f"Attempting parameter recovery for: {task.name}")
        
        # Try to replan tools with current context - the tool planner might have better info now
        try:
            planning_result = self.tool_planner.plan_tools(task, context)
            if planning_result.get("success", False):
                return {
                    "success": True,
                    "recovery_type": "parameter_adjustment",
                    "new_tool_sequence": planning_result["tool_sequence"],
                    "message": f"Adjusted parameters for {task.name}"
                }
        except Exception as e:
            self.logger.error(f"Parameter recovery failed: {e}")
        
        return None
    
    def _recover_tool_selection_error(self, task: Task, analysis: FailureAnalysis, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recover from tool selection errors by trying alternative tools."""
        if not analysis.alternative_tools:
            return None
        
        self.logger.info(f"Trying alternative tools for {task.name}: {analysis.alternative_tools}")
        
        # Try each alternative tool
        for alt_tool in analysis.alternative_tools:
            try:
                # Create a modified task with alternative tool hint
                alt_task = Task(
                    id=task.id + "_alt",
                    name=task.name,
                    description=f"{task.description} (use {alt_tool} tool)",
                    priority=task.priority,
                    dependencies=task.dependencies
                )
                
                planning_result = self.tool_planner.plan_tools(alt_task, context)
                if planning_result.get("success", False):
                    tools = planning_result["tool_sequence"]
                    if alt_tool in tools:
                        return {
                            "success": True,
                            "recovery_type": "tool_substitution", 
                            "new_tool_sequence": tools,
                            "message": f"Using alternative tool {alt_tool} for {task.name}"
                        }
            except Exception as e:
                self.logger.warning(f"Alternative tool {alt_tool} failed: {e}")
                continue
        
        return None
    
    def _recover_dependency_error(self, task: Task, analysis: FailureAnalysis, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recover from dependency errors by checking if prerequisites can be satisfied."""
        # For now, just return None - dependency recovery is complex and might require replanning
        self.logger.info(f"Dependency error for {task.name} - may require replanning")
        return None
    
    def _recover_resource_error(self, task: Task, analysis: FailureAnalysis, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recover from resource errors with retry after delay."""
        self.logger.info(f"Resource error for {task.name} - retrying after delay")
        
        # Simple retry strategy - wait a bit and try again
        time.sleep(1)  # Brief delay
        
        return {
            "success": True,
            "recovery_type": "retry_after_delay",
            "new_tool_sequence": task.tool_sequence,
            "message": f"Retrying {task.name} after resource error"
        }
    
    def _recover_planning_error(self, task: Task, analysis: FailureAnalysis, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recover from planning errors - this might require full replanning."""
        if self.replanning_attempts >= self.max_replanning_attempts:
            self.logger.warning(f"Max replanning attempts exceeded")
            return None
        
        self.logger.info(f"Planning error detected for task '{task.name}' - triggering replanning")
        
        # Increment replanning attempts
        self.replanning_attempts += 1
        
        # Create a recovery strategy that signals the need for goal replanning
        return {
            "success": True,
            "strategy": "replan",
            "modified_task": None,  # No modified task - this will trigger full replanning
            "reason": f"Planning error detected: {analysis.suggested_action}. Full goal replanning required.",
            "requires_goal_replanning": True,
            "failed_task_context": {
                "task_name": task.name,
                "task_description": task.description,
                "failure_reason": analysis.error_message,
                "suggested_action": analysis.suggested_action
            }
        }
    
    @traceable(name="evaluate_replanning_need", metadata={"component": "recovery_system"})
    def should_trigger_replanning(self, failed_tasks: List[Task]) -> bool:
        """
        Determine if multiple task failures indicate a need for goal replanning.
        
        Args:
            failed_tasks: List of tasks that have failed
            
        Returns:
            True if replanning should be triggered
        """
        if len(failed_tasks) < 2:
            return False
        
        # Check for patterns that indicate planning problems
        planning_error_count = sum(1 for attempt in self.recovery_attempts 
                                 if attempt.failure_type == FailureType.PLANNING_ERROR)
        
        tool_selection_error_count = sum(1 for attempt in self.recovery_attempts 
                                       if attempt.failure_type == FailureType.TOOL_SELECTION_ERROR)
        
        # Trigger replanning if we have multiple planning/tool selection errors
        if planning_error_count >= 2 or tool_selection_error_count >= 3:
            self.logger.info(f"Multiple planning errors detected - triggering replanning")
            return True
        
        # Trigger if >50% of tasks failed
        total_tasks = len(failed_tasks) + sum(1 for attempt in self.recovery_attempts if attempt.success)
        failure_rate = len(failed_tasks) / total_tasks if total_tasks > 0 else 0
        
        if failure_rate > 0.5:
            self.logger.info(f"High failure rate ({failure_rate:.1%}) - triggering replanning")
            return True
        
        return False
    
    @traceable(name="evaluate_graceful_degradation", metadata={"component": "recovery_system"})
    def can_gracefully_degrade(self, task: Task, failure_analysis: FailureAnalysis) -> bool:
        """
        Determine if a task failure can be gracefully degraded without affecting critical functionality.
        
        Args:
            task: The failed task
            failure_analysis: Analysis of the failure
            
        Returns:
            True if the task can be gracefully degraded
        """
        # Check task priority - low priority tasks can usually be degraded
        if task.priority.value <= 1:  # LOW priority
            self.logger.info(f"Task '{task.name}' can be gracefully degraded (LOW priority)")
            return True
        
        # Check if task is conversational or auxiliary
        if (task.metadata.get("is_conversational", False) or 
            task.metadata.get("is_greeting", False) or
            task.metadata.get("is_auxiliary", False)):
            self.logger.info(f"Task '{task.name}' can be gracefully degraded (conversational/auxiliary)")
            return True
        
        # Check if this is a non-critical dependency error
        if failure_analysis.failure_type == FailureType.DEPENDENCY_MISSING:
            # If the missing dependency is optional or low priority, can degrade
            optional_dependencies = task.metadata.get("optional_dependencies", [])
            if task.name in optional_dependencies:
                self.logger.info(f"Task '{task.name}' can be gracefully degraded (optional dependency)")
                return True
        
        # Check if failure is resource-related and task has fallback options
        if failure_analysis.failure_type == FailureType.RESOURCE_ERROR:
            has_fallback = task.metadata.get("has_fallback", False)
            if has_fallback:
                self.logger.info(f"Task '{task.name}' can be gracefully degraded (has fallback)")
                return True
        
        # For calculation tasks, check if approximate results are acceptable
        if "calculate" in task.name.lower() or "measure" in task.name.lower():
            allow_approximation = task.metadata.get("allow_approximation", False)
            if allow_approximation:
                self.logger.info(f"Task '{task.name}' can be gracefully degraded (approximation allowed)")
                return True
        
        return False
    
    @traceable(name="create_degraded_response", metadata={"component": "recovery_system"})
    def create_degraded_response(self, task: Task, failure_analysis: FailureAnalysis) -> Dict[str, Any]:
        """
        Create a degraded response when a task cannot be completed normally.
        
        Args:
            task: The failed task
            failure_analysis: Analysis of the failure
            
        Returns:
            A response that provides partial value despite the failure
        """
        degraded_responses = {
            # For calculation tasks that failed
            "distance": "Unable to calculate exact distance due to data limitations. Consider checking building model integrity.",
            "area": "Unable to calculate exact area due to missing measurements. Manual verification recommended.",
            "compliance": "Compliance check could not be completed automatically. Manual review may be required.",
            "properties": "Some element properties could not be retrieved. Available data has been provided.",
            "relationship": "Element relationships could not be fully analyzed. Partial analysis available."
        }
        
        # Generate context-aware degraded response
        if "distance" in task.description.lower():
            base_message = degraded_responses["distance"]
        elif "area" in task.description.lower():
            base_message = degraded_responses["area"]
        elif "compliance" in task.description.lower():
            base_message = degraded_responses["compliance"]
        elif "properties" in task.description.lower():
            base_message = degraded_responses["properties"]
        elif "relation" in task.description.lower():
            base_message = degraded_responses["relationship"]
        else:
            base_message = f"Task '{task.name}' could not be completed due to {failure_analysis.failure_type.value}."
        
        # Add helpful context if available
        context_note = ""
        if failure_analysis.alternative_tools:
            context_note = f" Alternative approaches using {', '.join(failure_analysis.alternative_tools)} might be available."
        
        return {
            "success": False,
            "degraded": True,
            "message": base_message + context_note,
            "partial_data": task.metadata.get("partial_results", {}),
            "failure_reason": failure_analysis.error_message,
            "suggested_action": failure_analysis.suggested_action
        }
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts for monitoring and debugging."""
        total_attempts = len(self.recovery_attempts)
        successful_attempts = sum(1 for attempt in self.recovery_attempts if attempt.success)
        
        failure_types = {}
        for attempt in self.recovery_attempts:
            failure_type = attempt.failure_type.value
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        
        return {
            "total_recovery_attempts": total_attempts,
            "successful_recoveries": successful_attempts,
            "recovery_success_rate": successful_attempts / total_attempts if total_attempts > 0 else 0,
            "failure_type_breakdown": failure_types,
            "replanning_attempts": self.replanning_attempts
        }