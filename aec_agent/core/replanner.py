"""
Replanner - LLM-powered dynamic plan revision for reactive reasoning.

This module analyzes execution results and validation outcomes to intelligently
revise task plans when current approaches are not working effectively.
"""

import uuid
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

from .reasoning_utils import ReasoningUtils, Task, TaskStatus, Priority
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
class ReplanningTrigger:
    """Information about what triggered a replanning event."""
    trigger_type: str
    details: Dict[str, Any]
    confidence: float
    reasoning: str


@dataclass
class ReplanningResult:
    """Result of a replanning operation."""
    success: bool
    new_tasks: List[Task]
    removed_task_ids: List[str]
    modified_task_ids: List[str]
    reasoning: str
    confidence: float
    trigger: ReplanningTrigger
    metadata: Dict[str, Any]


class Replanner:
    """
    LLM-powered plan revision for reactive reasoning.
    
    Analyzes execution context and failures to intelligently revise
    task plans when current approaches are not working effectively.
    """
    
    def __init__(self, llm=None, config: Optional['AgentConfig'] = None):
        """Initialize the replanner.
        
        Args:
            llm: Optional pre-configured LLM instance (takes precedence)
            config: Optional AgentConfig to use for model configuration
        """
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        # Setup LLM for intelligent replanning
        if llm is not None:
            self.llm = llm
        elif config is not None:
            # Use config to create LLM
            self.llm = ChatOpenAI(
                model=config.llm.get_component_model("replanner"),
                temperature=config.llm.get_component_temperature("replanner"),
                max_tokens=config.llm.get_component_max_tokens("replanner")
            )
        else:
            # Fallback to defaults (for backward compatibility)
            self.llm = ChatOpenAI(
                model="gpt-5-mini",
                temperature=0.2,  # Slightly higher for creative replanning
                max_tokens=2000
            )
        
    
    @traceable(name="replan_execution", metadata={"component": "replanner"})
    def replan(
        self,
        current_goal: str,
        current_tasks: List[Task],
        execution_context: Dict[str, Any],
        trigger: ReplanningTrigger
    ) -> ReplanningResult:
        """
        Revise the current plan based on execution context and trigger.
        
        Args:
            current_goal: The original goal being pursued
            current_tasks: Current list of tasks in the plan
            execution_context: Rich context from execution memory
            trigger: Information about what triggered this replanning
            
        Returns:
            ReplanningResult with new plan or failure information
        """
        self.logger.info(f"Replanning triggered by: {trigger.trigger_type}")
        
        # Use ONLY LLM-powered intelligent replanning - NO FALLBACKS
        llm_result = self._llm_replan(current_goal, current_tasks, execution_context, trigger)
        if llm_result and llm_result.success:
            self.logger.info(f"LLM replanning successful: {len(llm_result.new_tasks)} tasks")
            return llm_result
        
        # NO FALLBACKS - Fail explicitly if LLM replanning fails
        raise RuntimeError(f"LLM replanning failed for trigger: {trigger.trigger_type}. No fallback mechanisms available.")
    
    @traceable(name="llm_replanning")
    def _llm_replan(
        self,
        current_goal: str,
        current_tasks: List[Task],
        execution_context: Dict[str, Any],
        trigger: ReplanningTrigger
    ) -> Optional[ReplanningResult]:
        """Use LLM to intelligently replan based on execution context."""
        
        # Prepare context summary for LLM
        context_summary = self._prepare_context_summary(execution_context)
        
        # Create replanning prompt
        replanning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AEC (Architecture, Engineering, Construction) compliance reasoning specialist.

Your job is to revise task plans when current approaches are not working effectively.

Available tools for task execution:
- load_building_data: Load IFC JSON building data files (input: file_path)
- get_all_elements: Get all elements of a specific type like spaces, doors, walls, slabs, stairs (input: element_type)
- get_element_properties: Get detailed properties of specific elements (input: element_id)
- query_elements: Filter elements with specific criteria (input: JSON with element_type and filters)
- calculate_metrics: Perform calculations like counts, areas, volumes (input: JSON with operation and parameters)
- find_related_elements: Find spatial relationships between elements (input: JSON with element_id and relationship_type)
- validate_compliance_rule: Check elements against compliance rules (input: JSON with rule_type, element_id, criteria)
- search_compliance_documents: Search building codes and regulations (input: query_string)

Current Goal: {goal}
Current Tasks: {current_tasks}
Execution Context: {execution_context}
Trigger Type: {trigger_type}
Trigger Details: {trigger_details}

Based on the execution context and what triggered this replanning, create a revised task plan that addresses the issues.

Guidelines:
1. Keep successful tasks if they're still relevant
2. Remove or modify failing tasks
3. Add new tasks to address discovered issues
4. Ensure logical dependencies between tasks
5. Leverage any discovered context or information
6. Focus on achieving the original goal more effectively

Return ONLY a JSON response with:
{{
  "reasoning": "Brief explanation of why this replanning approach will work better",
  "confidence": 0.0-1.0,
  "tasks_to_remove": ["list of task IDs to remove"],
  "tasks_to_modify": [
    {{"id": "task_id", "new_name": "updated name", "new_description": "updated description"}}
  ],
  "tasks_to_add": [
    {{"name": "Task name", "description": "Detailed description", "priority": "HIGH|MEDIUM|LOW"}}
  ]
}}

Be pragmatic and focus on what will actually work given the execution context."""),
            ("human", "Please replan the task execution:")
        ])
        
        try:
            # Format current tasks for LLM
            tasks_for_llm = []
            for task in current_tasks:
                tasks_for_llm.append({
                    "id": task.id,
                    "name": task.name,
                    "description": task.description,
                    "status": task.status.value,
                    "priority": task.priority.name
                })
            
            # Execute LLM replanning with retry guardrail
            chain = replanning_prompt | self.llm | StrOutputParser()
            response = self._invoke_llm_with_retry(chain, {
                "goal": current_goal,
                "current_tasks": json.dumps(tasks_for_llm, indent=2),
                "execution_context": context_summary,
                "trigger_type": trigger.trigger_type,
                "trigger_details": json.dumps(trigger.details, indent=2)
            })
            
            # Parse JSON response
            replan_data = json.loads(response.strip())
            
            if not isinstance(replan_data, dict):
                self.logger.warning("LLM returned invalid replanning format")
                return None
            
            # Convert to our format
            new_tasks = []
            for task_info in replan_data.get("tasks_to_add", []):
                task_id = str(uuid.uuid4())
                priority = Priority.__members__.get(task_info.get("priority", "MEDIUM"), Priority.MEDIUM)
                
                task = Task(
                    id=task_id,
                    name=task_info.get("name", "Unnamed task"),
                    description=task_info.get("description", ""),
                    priority=priority,
                    metadata={
                        "replanned": True,
                        "trigger_type": trigger.trigger_type,
                        "original_goal": current_goal
                    }
                )
                new_tasks.append(task)
            
            # Apply modifications to existing tasks
            modified_task_ids = []
            for mod in replan_data.get("tasks_to_modify", []):
                task_id = mod.get("id")
                for task in current_tasks:
                    if task.id == task_id:
                        task.name = mod.get("new_name", task.name)
                        task.description = mod.get("new_description", task.description)
                        modified_task_ids.append(task_id)
                        break
            
            return ReplanningResult(
                success=True,
                new_tasks=new_tasks,
                removed_task_ids=replan_data.get("tasks_to_remove", []),
                modified_task_ids=modified_task_ids,
                reasoning=replan_data.get("reasoning", "LLM replanning completed"),
                confidence=replan_data.get("confidence", 0.7),
                trigger=trigger,
                metadata={"method": "llm_reasoning", "llm_response": replan_data}
            )
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"LLM replanning response was not valid JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"LLM replanning failed: {e}")
            return None
    
    @default_llm_retry
    def _invoke_llm_with_retry(self, chain, inputs: Dict[str, Any]) -> str:
        """Invoke LLM chain with retry guardrail."""
        return chain.invoke(inputs)
    
    def _prepare_context_summary(self, execution_context: Dict[str, Any]) -> str:
        """Prepare a concise summary of execution context for the LLM."""
        
        # Extract key information
        recent_steps = execution_context.get("recent_execution_steps", [])
        recent_failures = execution_context.get("recent_failures", [])
        discoveries = execution_context.get("recent_context_discoveries", [])
        plan_confidence = execution_context.get("plan_confidence", 1.0)
        
        summary_parts = [
            f"Current iteration: {execution_context.get('current_iteration', 0)}",
            f"Plan confidence: {plan_confidence:.1%}"
        ]
        
        if recent_steps:
            summary_parts.append(f"Recent execution steps: {len(recent_steps)} total")
            for step in recent_steps[-3:]:  # Show last 3
                status = "✓" if step.get("success", False) else "✗"
                summary_parts.append(f"  {status} {step.get('task', 'Unknown')} using {step.get('tool', 'unknown tool')}")
        
        if recent_failures:
            summary_parts.append(f"Recent failures: {len(recent_failures)}")
            for failure in recent_failures[-2:]:  # Show last 2
                summary_parts.append(f"  • {failure.get('task', 'Unknown task')}: {failure.get('error', 'Unknown error')}")
        
        if discoveries:
            summary_parts.append("Recent discoveries:")
            for discovery in discoveries[-3:]:  # Show last 3
                summary_parts.append(f"  • {discovery.get('key', 'Unknown')}: {str(discovery.get('value', 'Unknown'))[:50]}")
        
        return "\n".join(summary_parts)
    
    def assess_replanning_need(self, execution_context: Dict[str, Any]) -> Optional[ReplanningTrigger]:
        """
        Assess whether replanning is needed based on execution context.
        
        Args:
            execution_context: Current execution context from execution memory
            
        Returns:
            ReplanningTrigger if replanning is recommended, None otherwise
        """
        recent_failures = execution_context.get("recent_failures", [])
        plan_confidence = execution_context.get("plan_confidence", 1.0)
        recent_steps = execution_context.get("recent_execution_steps", [])
        
        # Check for high failure rate
        if len(recent_failures) >= 2:
            failure_rate = len(recent_failures) / max(len(recent_steps), 1)
            if failure_rate > 0.5:
                return ReplanningTrigger(
                    trigger_type="validation_failure",
                    details={
                        "failure_rate": failure_rate,
                        "recent_failures": recent_failures,
                        "failure_count": len(recent_failures)
                    },
                    confidence=0.8,
                    reasoning=f"High failure rate detected: {failure_rate:.1%}"
                )
        
        # Check for low plan confidence
        if plan_confidence < 0.5:
            return ReplanningTrigger(
                trigger_type="low_confidence", 
                details={
                    "plan_confidence": plan_confidence,
                    "confidence_threshold": 0.5
                },
                confidence=0.7,
                reasoning=f"Plan confidence too low: {plan_confidence:.1%}"
            )
        
        # Check for significant context discoveries that could improve planning
        discoveries = execution_context.get("recent_context_discoveries", [])
        high_confidence_discoveries = [d for d in discoveries if d.get("confidence", 0) > 0.8]
        if len(high_confidence_discoveries) >= 2:
            return ReplanningTrigger(
                trigger_type="context_discovery",
                details={
                    "discoveries": high_confidence_discoveries,
                    "discovery_count": len(high_confidence_discoveries)
                },
                confidence=0.6,
                reasoning=f"Significant context discovered: {len(high_confidence_discoveries)} high-confidence discoveries"
            )
        
        return None
    
    
    def validate_replan_result(self, result: ReplanningResult, original_goal: str) -> bool:
        """
        Validate that a replanning result is logical and achieves the goal.
        
        Args:
            result: The replanning result to validate
            original_goal: The original goal being pursued
            
        Returns:
            True if the replan result is valid
        """
        if not result.success:
            return False
        
        # Check that we have either new tasks or modifications
        if not result.new_tasks and not result.modified_task_ids:
            self.logger.warning("Replanning result has no new tasks or modifications")
            return False
        
        # Check that new tasks have valid structure
        for task in result.new_tasks:
            if not task.name or not task.description:
                self.logger.warning(f"Invalid task structure: {task}")
                return False
        
        # Check confidence is reasonable
        if result.confidence < 0.3:
            self.logger.warning(f"Replanning confidence too low: {result.confidence}")
            return False
        
        return True