"""
Execution Memory - Tracks reactive reasoning execution state and context.

This module manages execution-specific memory that accumulates during reactive
reasoning loops, including tool execution results, validation outcomes,
discovered context, plan modifications, and execution confidence metrics.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from pydantic import BaseModel, Field

from ..core.llm_guardrails import GuardrailConfig, ExecutionGuardrail, GuardrailViolationError


logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of an execution step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationLevel(str, Enum):
    """Level of validation assessment."""
    EXECUTION = "execution"
    BASIC = "basic"
    LOGICAL = "logical"
    PROGRESS = "progress"
    LLM_INTELLIGENT = "llm_intelligent"


class ExecutionStep(BaseModel):
    """Individual execution step within a reactive reasoning loop."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration: int = Field(..., description="Iteration number in reasoning loop")
    task_name: str = Field(..., description="Name of the task being executed")
    task_id: str = Field(..., description="ID of the task from reasoning system")
    tool_name: str = Field(..., description="Tool that was executed")
    
    # Execution details
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(default=None)
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    
    # Tool execution results
    tool_arguments: Dict[str, Any] = Field(default_factory=dict)
    tool_output: Optional[Any] = Field(default=None)
    tool_success: bool = Field(default=False)
    error_message: Optional[str] = Field(default=None)
    
    # Validation results
    validation_success: bool = Field(default=False)
    validation_level: Optional[ValidationLevel] = Field(default=None)
    validation_message: str = Field(default="")
    validation_confidence: float = Field(default=0.0, description="Validation confidence 0.0-1.0")
    validation_method: str = Field(default="", description="Method used for validation")
    validation_issues: List[str] = Field(default_factory=list)
    
    # Context discovered during execution
    discovered_context: Dict[str, Any] = Field(default_factory=dict, description="Context discovered during execution")
    
    def mark_completed(self, success: bool = True, error: Optional[str] = None):
        """Mark the execution step as completed."""
        self.status = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()
        if error:
            self.error_message = error
            
    def add_validation_result(self, validation_result: Dict[str, Any]):
        """Add validation result to the execution step."""
        self.validation_success = validation_result.get("success", False)
        self.validation_message = validation_result.get("message", "")
        self.validation_confidence = validation_result.get("confidence", 0.0)
        self.validation_method = validation_result.get("method", "")
        self.validation_level = ValidationLevel(validation_result.get("validation_level", "basic"))
        self.validation_issues = validation_result.get("issues", [])


class PlanModification(BaseModel):
    """Record of a plan modification during reactive execution."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration: int = Field(..., description="Iteration when modification occurred")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Trigger for modification
    trigger_type: str = Field(..., description="What triggered the replanning")
    trigger_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Original vs new plan
    original_plan_summary: str = Field(default="")
    new_plan_summary: str = Field(default="")
    modification_reasoning: str = Field(default="")
    
    # Tasks affected
    tasks_added: List[str] = Field(default_factory=list)
    tasks_removed: List[str] = Field(default_factory=list) 
    tasks_modified: List[str] = Field(default_factory=list)


class DiscoveredContext(BaseModel):
    """Context information discovered during execution."""
    
    discovery_type: str = Field(..., description="Type of discovery (file_path, element_count, etc.)")
    key: str = Field(..., description="Context key")
    value: Any = Field(..., description="Context value")
    iteration: int = Field(..., description="Iteration when discovered")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_tool: str = Field(..., description="Tool that discovered this context")
    confidence: float = Field(default=1.0, description="Confidence in this discovery")


class ExecutionMemory:
    """
    Memory system for reactive reasoning execution state.
    
    Tracks the evolution of context, plan modifications, and execution
    outcomes throughout a reactive reasoning session to enable intelligent
    replanning and adaptive behavior.
    """
    
    def __init__(self, session_id: str, goal: str, guardrail_config: Optional[GuardrailConfig] = None):
        """
        Initialize execution memory for a reasoning session.
        
        Args:
            session_id: ID of the reasoning session
            goal: Primary goal being pursued
            guardrail_config: Optional configuration for execution guardrails
        """
        self.session_id = session_id
        self.goal = goal
        self.logger = logging.getLogger(__name__)
        
        # Initialize execution guardrails
        config = guardrail_config or GuardrailConfig.from_env()
        self.execution_guardrail = ExecutionGuardrail(config)
        
        # Execution tracking
        self.execution_steps: List[ExecutionStep] = []
        self.plan_modifications: List[PlanModification] = []
        self.discovered_context: List[DiscoveredContext] = []
        
        # Current state
        self.current_iteration: int = 0
        self.current_context: Dict[str, Any] = {}
        self.current_plan_confidence: float = 1.0
        
        # Goal achievement tracking
        self.goal_achievement_assessments: List[Dict[str, Any]] = []
        self.is_goal_achieved: bool = False
        
        # Replanning tracking
        self.replanning_triggers: Set[str] = set()
        self.total_replanning_events: int = 0
        
        self.logger.info(f"ExecutionMemory initialized for goal: {goal[:50]}...")
    
    def start_iteration(self, iteration: int) -> None:
        """Start a new iteration in the reasoning loop."""
        self.current_iteration = iteration
        self.logger.debug(f"Started iteration {iteration}")
    
    def record_execution_step(
        self,
        task_name: str,
        task_id: str,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        tool_output: Any = None,
        tool_success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """
        Record a tool execution step.
        
        Args:
            task_name: Name of the task being executed
            task_id: ID of the task from reasoning system
            tool_name: Name of the tool executed
            tool_arguments: Arguments passed to the tool
            tool_output: Output from tool execution
            tool_success: Whether tool execution succeeded
            error_message: Error message if execution failed
            
        Returns:
            ID of the created execution step
            
        Raises:
            GuardrailViolationError: If execution guardrails are violated
        """
        # Check execution guardrails before recording
        self.execution_guardrail.record_execution_step()
        self.execution_guardrail.record_task_attempt(task_id)
        
        step = ExecutionStep(
            iteration=self.current_iteration,
            task_name=task_name,
            task_id=task_id,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_output=tool_output,
            tool_success=tool_success,
            error_message=error_message
        )
        
        step.mark_completed(tool_success, error_message)
        self.execution_steps.append(step)
        
        self.logger.debug(f"Recorded execution step: {tool_name} for task {task_name}")
        return step.id
    
    def add_validation_result(self, execution_step_id: str, validation_result: Dict[str, Any]) -> None:
        """
        Add validation result to an execution step.
        
        Args:
            execution_step_id: ID of the execution step
            validation_result: Validation result from validator
        """
        for step in self.execution_steps:
            if step.id == execution_step_id:
                step.add_validation_result(validation_result)
                self.logger.debug(f"Added validation result to step {execution_step_id}")
                return
        
        self.logger.warning(f"Execution step {execution_step_id} not found for validation")
    
    def discover_context(
        self,
        discovery_type: str,
        key: str,
        value: Any,
        source_tool: str,
        confidence: float = 1.0
    ) -> None:
        """
        Record discovered context information.
        
        Args:
            discovery_type: Type of discovery (file_path, element_count, etc.)
            key: Context key
            value: Context value
            source_tool: Tool that discovered this context
            confidence: Confidence in this discovery (0.0-1.0)
        """
        discovery = DiscoveredContext(
            discovery_type=discovery_type,
            key=key,
            value=value,
            iteration=self.current_iteration,
            source_tool=source_tool,
            confidence=confidence
        )
        
        self.discovered_context.append(discovery)
        
        # Update current context
        if key not in self.current_context or confidence >= 0.8:
            self.current_context[key] = value
        
        self.logger.debug(f"Discovered context: {key} = {value} (confidence: {confidence})")
    
    def record_plan_modification(
        self,
        trigger_type: str,
        trigger_details: Dict[str, Any],
        original_plan_summary: str,
        new_plan_summary: str,
        modification_reasoning: str,
        tasks_added: List[str] = None,
        tasks_removed: List[str] = None,
        tasks_modified: List[str] = None
    ) -> str:
        """
        Record a plan modification event.
        
        Args:
            trigger_type: What triggered the replanning
            trigger_details: Details about the trigger
            original_plan_summary: Summary of original plan
            new_plan_summary: Summary of new plan
            modification_reasoning: Reasoning for the modification
            tasks_added: List of task names added
            tasks_removed: List of task names removed
            tasks_modified: List of task names modified
            
        Returns:
            ID of the plan modification record
            
        Raises:
            GuardrailViolationError: If replanning guardrails are violated
        """
        # Check replanning guardrails before recording
        self.execution_guardrail.record_replanning_event()
        
        modification = PlanModification(
            iteration=self.current_iteration,
            trigger_type=trigger_type,
            trigger_details=trigger_details,
            original_plan_summary=original_plan_summary,
            new_plan_summary=new_plan_summary,
            modification_reasoning=modification_reasoning,
            tasks_added=tasks_added or [],
            tasks_removed=tasks_removed or [],
            tasks_modified=tasks_modified or []
        )
        
        self.plan_modifications.append(modification)
        self.replanning_triggers.add(trigger_type)
        self.total_replanning_events += 1
        
        # Adjust plan confidence based on modification
        if trigger_type in ["validation_failure", "execution_error"]:
            self.current_plan_confidence *= 0.8
        elif trigger_type == "context_discovery":
            self.current_plan_confidence = min(1.0, self.current_plan_confidence + 0.1)
        
        self.logger.info(f"Recorded plan modification: {trigger_type} -> {modification_reasoning[:50]}...")
        return modification.id
    
    def assess_goal_achievement(self, assessment: Dict[str, Any]) -> None:
        """
        Record a goal achievement assessment.
        
        Args:
            assessment: Assessment result from progress evaluator
        """
        assessment_record = {
            "iteration": self.current_iteration,
            "timestamp": datetime.now(timezone.utc),
            **assessment
        }
        
        self.goal_achievement_assessments.append(assessment_record)
        self.is_goal_achieved = assessment.get("goal_achieved", False)
        
        self.logger.debug(f"Goal achievement assessment: {self.is_goal_achieved}")
    
    def get_current_context(self) -> Dict[str, Any]:
        """
        Get the current accumulated context.
        
        Returns:
            Current context dictionary with all discovered information
        """
        context = {
            "goal": self.goal,
            "session_id": self.session_id,
            "current_iteration": self.current_iteration,
            "plan_confidence": self.current_plan_confidence,
            "total_execution_steps": len(self.execution_steps),
            "total_replanning_events": self.total_replanning_events,
            "replanning_triggers": list(self.replanning_triggers),
            "is_goal_achieved": self.is_goal_achieved,
            **self.current_context
        }
        
        return context
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of execution progress.
        
        Returns:
            Summary with key execution metrics and outcomes
        """
        successful_steps = [step for step in self.execution_steps if step.tool_success and step.validation_success]
        failed_steps = [step for step in self.execution_steps if not step.tool_success or not step.validation_success]
        
        total_time = sum(step.execution_time for step in self.execution_steps)
        avg_confidence = sum(step.validation_confidence for step in self.execution_steps) / len(self.execution_steps) if self.execution_steps else 0.0
        
        return {
            "total_iterations": self.current_iteration,
            "total_execution_steps": len(self.execution_steps),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "success_rate": len(successful_steps) / len(self.execution_steps) if self.execution_steps else 0.0,
            "total_execution_time": total_time,
            "average_validation_confidence": avg_confidence,
            "plan_modifications": len(self.plan_modifications),
            "plan_confidence": self.current_plan_confidence,
            "context_discoveries": len(self.discovered_context),
            "goal_achieved": self.is_goal_achieved,
            "replanning_triggers": list(self.replanning_triggers)
        }
    
    def get_context_for_replanning(self) -> Dict[str, Any]:
        """
        Get rich context information for replanning decisions.
        
        Returns:
            Context optimized for LLM-based replanning
        """
        recent_steps = self.execution_steps[-5:] if len(self.execution_steps) > 5 else self.execution_steps
        recent_failures = [step for step in recent_steps if not step.tool_success or not step.validation_success]
        recent_discoveries = self.discovered_context[-10:] if len(self.discovered_context) > 10 else self.discovered_context
        
        return {
            "goal": self.goal,
            "current_iteration": self.current_iteration,
            "recent_execution_steps": [
                {
                    "task": step.task_name,
                    "tool": step.tool_name,
                    "success": step.tool_success and step.validation_success,
                    "confidence": step.validation_confidence,
                    "issues": step.validation_issues
                }
                for step in recent_steps
            ],
            "recent_failures": [
                {
                    "task": step.task_name,
                    "tool": step.tool_name,
                    "error": step.error_message or "Validation failed",
                    "validation_issues": step.validation_issues
                }
                for step in recent_failures
            ],
            "recent_context_discoveries": [
                {
                    "type": disc.discovery_type,
                    "key": disc.key,
                    "value": str(disc.value)[:100],  # Truncate for readability
                    "source": disc.source_tool,
                    "confidence": disc.confidence
                }
                for disc in recent_discoveries
            ],
            "plan_confidence": self.current_plan_confidence,
            "accumulated_context": self.current_context,
            "guardrails_status": self.execution_guardrail.get_status()
        }
    
    def get_context_for_progress_evaluation(self) -> Dict[str, Any]:
        """
        Get context for progress evaluation decisions.
        
        Returns:
            Context optimized for LLM-based progress evaluation
        """
        all_tasks = set()
        successful_tasks = set()
        
        for step in self.execution_steps:
            all_tasks.add(step.task_name)
            if step.tool_success and step.validation_success:
                successful_tasks.add(step.task_name)
        
        return {
            "goal": self.goal,
            "total_iterations": self.current_iteration,
            "unique_tasks_attempted": len(all_tasks),
            "unique_tasks_completed": len(successful_tasks),
            "task_completion_rate": len(successful_tasks) / len(all_tasks) if all_tasks else 0.0,
            "execution_summary": self.get_execution_summary(),
            "recent_validation_confidence": [
                step.validation_confidence for step in self.execution_steps[-5:]
            ] if len(self.execution_steps) >= 5 else [step.validation_confidence for step in self.execution_steps],
            "discovered_context_summary": {
                disc.key: disc.value for disc in self.discovered_context[-5:]
            } if len(self.discovered_context) >= 5 else {disc.key: disc.value for disc in self.discovered_context},
            "plan_modifications_count": len(self.plan_modifications),
            "current_context": self.current_context
        }
    
    def should_trigger_replanning(self) -> bool:
        """
        Determine if current state suggests replanning is needed.
        
        Returns:
            True if replanning should be triggered
        """
        if not self.execution_steps:
            return False
        
        # Check recent failure rate
        recent_steps = self.execution_steps[-3:] if len(self.execution_steps) > 3 else self.execution_steps
        recent_failures = sum(1 for step in recent_steps if not step.tool_success or not step.validation_success)
        failure_rate = recent_failures / len(recent_steps)
        
        # Check validation confidence
        recent_confidence = [step.validation_confidence for step in recent_steps if step.validation_confidence > 0]
        avg_confidence = sum(recent_confidence) / len(recent_confidence) if recent_confidence else 0.0
        
        # Trigger replanning if high failure rate or low confidence
        should_replan = failure_rate > 0.5 or avg_confidence < 0.6 or self.current_plan_confidence < 0.5
        
        if should_replan:
            self.logger.info(f"Replanning triggered: failure_rate={failure_rate:.2f}, "
                           f"avg_confidence={avg_confidence:.2f}, plan_confidence={self.current_plan_confidence:.2f}")
        
        return should_replan
    
    def clear(self) -> None:
        """Clear all execution memory."""
        self.execution_steps.clear()
        self.plan_modifications.clear()
        self.discovered_context.clear()
        self.goal_achievement_assessments.clear()
        self.current_context.clear()
        self.replanning_triggers.clear()
        
        self.current_iteration = 0
        self.current_plan_confidence = 1.0
        self.is_goal_achieved = False
        self.total_replanning_events = 0
        
        self.logger.info("Execution memory cleared")