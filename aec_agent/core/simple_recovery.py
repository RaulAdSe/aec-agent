"""
Simple Modular Recovery System

A clean, focused recovery system that uses LLM for decision making
without unnecessary complexity. Each recovery strategy is a separate module.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .reasoning_utils import Task, ExecutionResult, ReasoningUtils
from langsmith import traceable


@dataclass
class RecoveryResult:
    """Simple result from recovery attempt."""
    success: bool
    strategy: str
    modified_task: Optional[Task] = None
    message: str = ""


class RecoveryDecisionMaker:
    """Uses LLM to make simple recovery decisions."""
    
    def __init__(self, llm):
        self.llm = llm
        self.logger = ReasoningUtils.setup_logger(__name__)
    
    @traceable(name="decide_recovery")
    def decide_what_to_do(self, task: Task, error: str, context: Dict[str, Any]) -> str:
        """Ask LLM what to do about this failure. Returns simple strategy name."""
        
        available_files = context.get('available_files', [])
        
        prompt = f"""A task failed. You should try to FIX it, not give up easily!

Task: {task.name}
Description: {task.description}  
Error: {error}
Available tools: {context.get('available_tools', [])}
Available files: {available_files}

ANALYZE THE ERROR and choose the BEST fix:

1. retry_with_different_input - Change parameters, file path, or tool arguments
2. try_different_tool - Current tool failed, try an alternative tool
3. replan_goal - Task doesn't make sense anymore, need to rethink the approach
4. skip_gracefully - ONLY if task is truly impossible or non-critical

Be PROACTIVE and FLEXIBLE:
- Option 1: Try different inputs/parameters first (files: {available_files})
- Option 2: Switch tools if current approach is wrong
- Option 3: Replan if the whole approach needs rethinking  
- Option 4: Skip only as last resort

Your decision:"""

        try:
            response = self.llm.invoke(prompt)
            decision = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            self.logger.info(f"LLM recovery decision: {decision}")
            
            # Extract strategy from decision
            if "retry_with_different_input" in decision.lower():
                return "retry_with_different_input"
            elif "try_different_tool" in decision.lower():
                return "try_different_tool"
            elif "replan_goal" in decision.lower():
                return "replan_goal"
            else:
                return "skip_gracefully"
                
        except Exception as e:
            self.logger.error(f"LLM decision failed: {e}")
            return "skip_gracefully"


class SimpleRecovery:
    """Simple modular recovery system."""
    
    def __init__(self, llm):
        self.decision_maker = RecoveryDecisionMaker(llm)
        self.logger = ReasoningUtils.setup_logger(__name__)
    
    def recover(self, task: Task, error_result: ExecutionResult, context: Dict[str, Any]) -> RecoveryResult:
        """Main recovery entry point - simple and clean."""
        
        # Ask LLM what to do
        strategy = self.decision_maker.decide_what_to_do(
            task=task,
            error=error_result.error_message,
            context=context
        )
        
        # Execute the strategy
        if strategy == "retry_with_different_input":
            return self._retry_with_different_input(task, context, error_result.error_message)
        elif strategy == "try_different_tool":
            return self._try_different_tool(task, context)
        elif strategy == "replan_goal":
            return self._replan_goal(task, context, error_result.error_message)
        else:  # skip_gracefully or unknown
            return self._skip_gracefully(task, error_result.error_message)
    
    def _retry_with_different_input(self, task: Task, context: Dict[str, Any], error: str) -> RecoveryResult:
        """Modify task with different input/parameters based on error."""
        
        # Check available resources
        available_files = context.get('available_files', []) or context.get('active_files', [])
        
        # Flexible input modification
        new_description = task.description
        modifications = []
        
        # Add file path if missing and available
        if ("file" in error.lower() or "path" in error.lower()) and available_files:
            file_path = available_files[0]
            new_description = f"{task.description} (File: {file_path})"
            modifications.append(f"added file: {file_path}")
        
        # Add more specific parameters if error suggests it
        if "parameter" in error.lower() or "argument" in error.lower():
            new_description = f"{task.description} (with default parameters)"
            modifications.append("added default parameters")
            
        if not modifications:
            return RecoveryResult(
                success=False,
                strategy="retry_with_different_input",
                message="No suitable input modifications found"
            )
        
        modified_task = Task(
            id=task.id + "_modified",
            name=task.name,
            description=new_description,
            tool_sequence=task.tool_sequence,
            priority=task.priority,
            dependencies=task.dependencies
        )
        
        self.logger.info(f"Modified task input: {', '.join(modifications)}")
        
        return RecoveryResult(
            success=True,
            strategy="retry_with_different_input", 
            modified_task=modified_task,
            message=f"Retrying with modifications: {', '.join(modifications)}"
        )
    
    def _try_different_tool(self, task: Task, context: Dict[str, Any]) -> RecoveryResult:
        """Try a different tool for this task."""
        
        available_tools = context.get('available_tools', [])
        current_tool = task.tool_sequence[0] if task.tool_sequence else None
        
        # Find alternative tools (improved logic)
        alternatives = [t for t in available_tools if t != current_tool]
        
        if not alternatives:
            return RecoveryResult(
                success=False,
                strategy="try_different_tool",
                message="No alternative tools available"
            )
        
        # Use first alternative
        alt_tool = alternatives[0]
        modified_task = Task(
            id=task.id + "_alt_tool",
            name=task.name,
            description=f"{task.description} (using {alt_tool})",
            tool_sequence=[alt_tool],
            priority=task.priority,
            dependencies=task.dependencies
        )
        
        self.logger.info(f"Switching to alternative tool: {alt_tool}")
        
        return RecoveryResult(
            success=True,
            strategy="try_different_tool",
            modified_task=modified_task,
            message=f"Trying alternative tool: {alt_tool}"
        )
    
    def _replan_goal(self, task: Task, context: Dict[str, Any], error: str) -> RecoveryResult:
        """Signal that goal needs replanning."""
        
        self.logger.info(f"Task requires goal replanning: {task.name}")
        
        return RecoveryResult(
            success=True,
            strategy="replan_goal",
            message=f"Task '{task.name}' needs goal replanning due to: {error[:50]}..."
        )
    
    def _skip_gracefully(self, task: Task, error_message: str) -> RecoveryResult:
        """Skip task with helpful message."""
        
        self.logger.info(f"Gracefully skipping task: {task.name}")
        
        return RecoveryResult(
            success=True,
            strategy="skip_gracefully",
            message=f"Skipped {task.name} - {error_message[:50]}..."
        )