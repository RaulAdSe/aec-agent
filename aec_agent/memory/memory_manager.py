"""
Memory manager coordinator for the AEC compliance agent.

This module provides a unified interface to combine short-term conversation memory
and intermediate session memory, coordinating between different memory layers
and providing a clean API for agent integration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .short_term_memory import ShortTermMemory, ShortTermMemoryConfig
from .session_memory import SessionMemory, TaskStatus, SubTask, ToolExecution, GoalStatus, Goal
from .execution_memory import ExecutionMemory


logger = logging.getLogger(__name__)


class MemoryManagerConfig:
    """Configuration for the memory manager."""
    
    def __init__(
        self,
        short_term_config: Optional[ShortTermMemoryConfig] = None,
        session_persistence_path: Optional[Path] = None,
        auto_save_interval: int = 10,  # Save session every N operations
        enable_persistence: bool = True
    ):
        self.short_term_config = short_term_config or ShortTermMemoryConfig()
        self.session_persistence_path = session_persistence_path
        self.auto_save_interval = auto_save_interval
        self.enable_persistence = enable_persistence


class MemoryManager:
    """
    Central memory coordinator combining short-term and intermediate memory.
    
    This class provides a unified interface for:
    - Conversation context management (short-term)
    - Session state tracking (intermediate)
    - Memory persistence and retrieval
    - Context injection for LLM prompts
    """
    
    def __init__(
        self, 
        config: Optional[MemoryManagerConfig] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize the memory manager.
        
        Args:
            config: Memory manager configuration
            session_id: Optional existing session ID to resume
        """
        self.config = config or MemoryManagerConfig()
        self.logger = logging.getLogger(__name__)
        self._operation_count = 0
        
        # Initialize memory components
        self.short_term = ShortTermMemory(self.config.short_term_config)
        
        persistence_path = self.config.session_persistence_path if self.config.enable_persistence else None
        self.session = SessionMemory(session_id=session_id, persistence_path=persistence_path)
        
        # Initialize execution memory (for reactive reasoning)
        self.execution: Optional[ExecutionMemory] = None
        
        self.logger.info(f"MemoryManager initialized with session_id={self.session.state.session_id}")
    
    # Conversation Management (Short-term Memory)
    def add_conversation_turn(self, user_input: str, ai_output: str) -> None:
        """
        Add a conversation turn to short-term memory.
        
        Args:
            user_input: The user's input message
            ai_output: The AI agent's response
        """
        self.short_term.add_conversation_turn(user_input, ai_output)
        self._increment_operation_count()
        self.logger.debug("Added conversation turn to memory")
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for prompt injection."""
        return self.short_term.get_conversation_context()
    
    # Session Management (Intermediate Memory)
    def set_session_goal(self, goal: str, context: str = "") -> str:
        """
        Set the main goal for this session.
        
        Args:
            goal: Primary goal description
            context: Additional context information
            
        Returns:
            ID of the new goal
        """
        goal_id = self.session.set_goal(goal, context)
        self._increment_operation_count()
        self.logger.info(f"Session goal set: {goal[:50]}... (goal_id: {goal_id})")
        return goal_id
    
    def complete_current_goal(self, success: bool = True) -> None:
        """
        Mark the current goal as completed and archive it.
        
        Args:
            success: Whether the goal was completed successfully
        """
        self.session.complete_current_goal(success)
        self._increment_operation_count()
        self.logger.info(f"Current goal marked as {'completed' if success else 'failed'}")
    
    def add_subtask(self, name: str, dependencies: Optional[List[str]] = None) -> str:
        """
        Add a new subtask to track.
        
        Args:
            name: Description of the subtask
            dependencies: Optional list of dependent subtask IDs
            
        Returns:
            ID of the created subtask
        """
        task_id = self.session.add_subtask(name, dependencies)
        self._increment_operation_count()
        return task_id
    
    def update_subtask_status(self, subtask_id: str, status: TaskStatus, notes: Optional[str] = None) -> bool:
        """
        Update subtask status.
        
        Args:
            subtask_id: ID of the subtask
            status: New status
            notes: Optional notes
            
        Returns:
            True if updated successfully
        """
        success = self.session.update_subtask_status(subtask_id, status, notes)
        if success:
            self._increment_operation_count()
        return success
    
    def set_current_subtask(self, subtask_id: str) -> bool:
        """Set the currently active subtask."""
        success = self.session.set_current_subtask(subtask_id)
        if success:
            self._increment_operation_count()
        return success
    
    # File Management
    def track_active_file(self, file_path: str) -> None:
        """Add a file to active files tracking."""
        self.session.add_active_file(file_path)
        self._increment_operation_count()
    
    def untrack_active_file(self, file_path: str) -> None:
        """Remove a file from active files tracking."""
        self.session.remove_active_file(file_path)
        self._increment_operation_count()
    
    def mark_file_modified(self, file_path: str) -> None:
        """Mark a file as modified in this session."""
        self.session.mark_file_modified(file_path)
        self._increment_operation_count()
    
    # Tool Usage Tracking
    def record_tool_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool = True,
        result_summary: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> str:
        """
        Record a tool execution in session memory.
        
        Args:
            tool_name: Name of the executed tool
            arguments: Tool arguments
            success: Whether execution was successful
            result_summary: Summary of the result
            error_message: Error message if failed
            
        Returns:
            ID of the tool execution record
        """
        execution_id = self.session.record_tool_execution(
            tool_name=tool_name,
            arguments=arguments,
            success=success,
            result_summary=result_summary,
            error_message=error_message
        )
        self._increment_operation_count()
        return execution_id
    
    # Building Data Context (AEC-specific)
    def set_building_data_context(self, data_path: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Set the currently loaded building data context."""
        self.session.set_building_data(data_path, context)
        self._increment_operation_count()
    
    def add_compliance_rule(self, rule: str) -> None:
        """Add a compliance rule to track."""
        self.session.add_compliance_rule(rule)
        self._increment_operation_count()
    
    def update_analysis_results(self, results: Dict[str, Any]) -> None:
        """Update the latest analysis results."""
        self.session.update_analysis_results(results)
        self._increment_operation_count()
    
    # Execution Memory Management (for Reactive Reasoning)
    def start_reactive_execution(self, goal: str) -> str:
        """
        Start a reactive execution session.
        
        Args:
            goal: Primary goal for the reactive execution
            
        Returns:
            Session ID for the execution memory
        """
        self.execution = ExecutionMemory(
            session_id=self.session.state.session_id,
            goal=goal
        )
        
        # Update session memory to track reactive mode
        self.session.set_reasoning_mode("reactive")
        self.session.set_active_execution_memory(self.execution.session_id)
        
        self._increment_operation_count()
        self.logger.info(f"Started reactive execution for goal: {goal[:50]}...")
        return self.execution.session_id
    
    def end_reactive_execution(self) -> Dict[str, Any]:
        """
        End the reactive execution session and return summary.
        
        Returns:
            Final execution summary
        """
        if not self.execution:
            return {"message": "No active execution memory"}
        
        summary = self.execution.get_execution_summary()
        
        # Store execution summary in session memory
        self.session.update_execution_summary(summary)
        self.session.update_accumulated_context(self.execution.get_current_context())
        
        # Clear active execution memory
        execution_summary = summary.copy()
        self.execution = None
        self.session.set_reasoning_mode("static")
        self.session.set_active_execution_memory(None)
        
        self._increment_operation_count()
        self.logger.info("Ended reactive execution session")
        return execution_summary
    
    def record_execution_step(
        self,
        task_name: str,
        task_id: str,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        tool_output: Any = None,
        tool_success: bool = True,
        error_message: Optional[str] = None
    ) -> Optional[str]:
        """
        Record an execution step in reactive mode.
        
        Args:
            task_name: Name of the task being executed
            task_id: ID of the task from reasoning system
            tool_name: Name of the tool executed
            tool_arguments: Arguments passed to the tool
            tool_output: Output from tool execution
            tool_success: Whether tool execution succeeded
            error_message: Error message if execution failed
            
        Returns:
            Execution step ID if in reactive mode, None otherwise
        """
        if not self.execution:
            return None
        
        step_id = self.execution.record_execution_step(
            task_name=task_name,
            task_id=task_id,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_output=tool_output,
            tool_success=tool_success,
            error_message=error_message
        )
        
        self._increment_operation_count()
        return step_id
    
    def add_execution_validation(self, step_id: str, validation_result: Dict[str, Any]) -> None:
        """
        Add validation result to an execution step.
        
        Args:
            step_id: ID of the execution step
            validation_result: Validation result from validator
        """
        if self.execution:
            self.execution.add_validation_result(step_id, validation_result)
            self._increment_operation_count()
    
    def discover_execution_context(
        self,
        discovery_type: str,
        key: str,
        value: Any,
        source_tool: str,
        confidence: float = 1.0
    ) -> None:
        """
        Record discovered context during reactive execution.
        
        Args:
            discovery_type: Type of discovery (file_path, element_count, etc.)
            key: Context key
            value: Context value
            source_tool: Tool that discovered this context
            confidence: Confidence in this discovery (0.0-1.0)
        """
        if self.execution:
            self.execution.discover_context(discovery_type, key, value, source_tool, confidence)
            self._increment_operation_count()
    
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
    ) -> Optional[str]:
        """
        Record a plan modification during reactive execution.
        
        Returns:
            Plan modification ID if in reactive mode, None otherwise
        """
        if not self.execution:
            return None
        
        modification_id = self.execution.record_plan_modification(
            trigger_type=trigger_type,
            trigger_details=trigger_details,
            original_plan_summary=original_plan_summary,
            new_plan_summary=new_plan_summary,
            modification_reasoning=modification_reasoning,
            tasks_added=tasks_added,
            tasks_removed=tasks_removed,
            tasks_modified=tasks_modified
        )
        
        self._increment_operation_count()
        return modification_id
    
    def assess_goal_achievement(self, assessment: Dict[str, Any]) -> None:
        """Record goal achievement assessment during reactive execution."""
        if self.execution:
            self.execution.assess_goal_achievement(assessment)
            self._increment_operation_count()
    
    def get_execution_context_for_replanning(self) -> Optional[Dict[str, Any]]:
        """Get rich context for replanning decisions."""
        if not self.execution:
            return None
        return self.execution.get_context_for_replanning()
    
    def get_execution_context_for_progress_evaluation(self) -> Optional[Dict[str, Any]]:
        """Get context for progress evaluation."""
        if not self.execution:
            return None
        return self.execution.get_context_for_progress_evaluation()
    
    def should_trigger_replanning(self) -> bool:
        """Check if current execution state suggests replanning is needed."""
        if not self.execution:
            return False
        return self.execution.should_trigger_replanning()
    
    def start_execution_iteration(self, iteration: int) -> None:
        """Start a new iteration in reactive execution."""
        if self.execution:
            self.execution.start_iteration(iteration)
    
    def is_reactive_mode(self) -> bool:
        """Check if currently in reactive execution mode."""
        return self.execution is not None
    
    def get_reactive_execution_summary(self) -> Optional[Dict[str, Any]]:
        """Get current reactive execution summary."""
        if not self.execution:
            return None
        return self.execution.get_execution_summary()
    
    # Unified Context for LLM Prompts
    def get_full_context_for_prompt(self) -> str:
        """
        Get complete memory context formatted for LLM prompt injection.
        
        Returns:
            Formatted string with all relevant memory context
        """
        context_parts = []
        
        # Session context (structured state)
        session_context = self.session.get_context_for_prompt()
        if session_context:
            context_parts.append(f"=== Session Context ===\n{session_context}")
        
        # Execution context (if in reactive mode)
        if self.execution:
            execution_context = self.execution.get_current_context()
            execution_summary = self.execution.get_execution_summary()
            
            exec_context_str = f"Reasoning Mode: Reactive (Iteration {execution_context.get('current_iteration', 0)})\n"
            exec_context_str += f"Goal: {execution_context.get('goal', 'Unknown')}\n"
            exec_context_str += f"Execution Steps: {execution_summary.get('total_execution_steps', 0)}\n"
            exec_context_str += f"Success Rate: {execution_summary.get('success_rate', 0.0):.1%}\n"
            exec_context_str += f"Plan Confidence: {execution_context.get('plan_confidence', 1.0):.1%}\n"
            
            if execution_summary.get('plan_modifications', 0) > 0:
                exec_context_str += f"Plan Modifications: {execution_summary.get('plan_modifications')}\n"
            
            # Include recent discovered context
            recent_context = {k: v for k, v in execution_context.items() 
                            if k not in ['goal', 'session_id', 'current_iteration', 
                                       'plan_confidence', 'total_execution_steps', 
                                       'total_replanning_events', 'replanning_triggers', 'is_goal_achieved']}
            if recent_context:
                exec_context_str += f"Discovered Context: {recent_context}"
            
            context_parts.append(f"=== Reactive Execution Context ===\n{exec_context_str}")
        
        # Conversation context (recent messages + summary)
        conversation_context = self.short_term.get_conversation_context()
        if conversation_context:
            context_parts.append(f"=== Conversation History ===\n{conversation_context}")
        
        return "\n\n".join(context_parts)
    
    def get_memory_variables_for_langchain(self) -> Dict[str, Any]:
        """
        Get memory variables in LangChain-compatible format.
        
        Returns:
            Dictionary with memory variables for LangChain chains
        """
        # Get short-term memory variables
        short_term_vars = self.short_term.get_memory_variables()
        
        # Get session context
        session_summary = self.session.get_session_summary()
        session_context = self.session.get_context_for_prompt()
        
        # Combine all variables
        memory_vars = {
            # Short-term memory
            "recent_conversation": short_term_vars.get("recent_conversation", []),
            "conversation_summary": short_term_vars.get("conversation_summary", ""),
            
            # Session memory
            "session_goal": self.session.state.goal or "",
            "session_context": session_context,
            "active_files": self.session.state.active_files,
            "current_subtask": self.session.get_current_subtask().name if self.session.get_current_subtask() else "",
            "building_data_path": self.session.state.loaded_building_data or "",
            
            # Full context
            "full_memory_context": self.get_full_context_for_prompt()
        }
        
        # Add execution memory variables if in reactive mode
        if self.execution:
            execution_summary = self.execution.get_execution_summary()
            memory_vars.update({
                "reasoning_mode": "reactive",
                "execution_iteration": self.execution.current_iteration,
                "execution_steps": execution_summary.get("total_execution_steps", 0),
                "execution_success_rate": execution_summary.get("success_rate", 0.0),
                "plan_confidence": execution_summary.get("plan_confidence", 1.0),
                "plan_modifications": execution_summary.get("plan_modifications", 0),
                "context_discoveries": execution_summary.get("context_discoveries", 0)
            })
        else:
            memory_vars["reasoning_mode"] = "static"
        
        return memory_vars
    
    # Information Retrieval
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current session."""
        session_summary = self.session.get_session_summary()
        memory_stats = self.short_term.get_memory_stats()
        
        return {
            **session_summary,
            "memory_stats": memory_stats,
            "total_operations": self._operation_count
        }
    
    def get_pending_subtasks(self) -> List[SubTask]:
        """Get all pending subtasks."""
        return self.session.get_pending_subtasks()
    
    def get_recent_tool_executions(self, limit: int = 10) -> List[ToolExecution]:
        """Get recent tool executions."""
        return self.session.get_recent_tool_executions(limit)
    
    def get_active_files(self) -> List[str]:
        """Get currently active files."""
        return self.session.state.active_files.copy()
    
    def get_modified_files(self) -> List[str]:
        """Get files modified in this session."""
        return self.session.state.modified_files.copy()
    
    # Session Persistence
    def save_session(self) -> None:
        """Manually save the current session."""
        if self.config.enable_persistence:
            self.session.save_session()
            self.logger.debug("Session manually saved")
    
    def _auto_save_if_needed(self) -> None:
        """Automatically save session if auto-save interval is reached."""
        if (self.config.enable_persistence and 
            self.config.auto_save_interval > 0 and 
            self._operation_count % self.config.auto_save_interval == 0):
            self.session.save_session()
            self.logger.debug(f"Auto-saved session after {self._operation_count} operations")
    
    def _increment_operation_count(self) -> None:
        """Increment operation counter and trigger auto-save/compaction if needed."""
        self._operation_count += 1
        
        # Token-based automatic compaction (primary trigger)
        # Check every 5 operations to avoid too much overhead
        if self._operation_count % 5 == 0:
            compaction_triggered = self.session.check_and_trigger_compaction()
            
            # If token-based didn't trigger, check periodic backup
            if not compaction_triggered and self._operation_count % 50 == 0:
                # Periodic maintenance compaction
                self.session.compact_session_memory()
        
        self._auto_save_if_needed()
    
    # Memory Management
    def clear_conversation_memory(self) -> None:
        """Clear only the conversation memory, keeping session state."""
        self.short_term.clear_memory()
        self.logger.info("Conversation memory cleared")
    
    def clear_session_memory(self) -> None:
        """Clear only the session memory, keeping conversation history."""
        # Clear execution memory if active
        if self.execution:
            self.execution.clear()
            self.execution = None
        
        self.session.clear_session()
        self._operation_count = 0
        self.logger.info("Session memory cleared")
    
    def clear_execution_memory(self) -> None:
        """Clear only the execution memory, keeping session and conversation state."""
        if self.execution:
            self.execution.clear()
            self.execution = None
            self.session.clear_execution_state()
            self.logger.info("Execution memory cleared")
    
    def clear_all_memory(self) -> None:
        """Clear all memory components."""
        self.clear_conversation_memory()
        self.clear_session_memory()
        self.logger.info("All memory cleared")
    
    # Configuration Updates
    def update_short_term_config(self, new_config: ShortTermMemoryConfig) -> None:
        """Update short-term memory configuration."""
        self.short_term.update_config(new_config)
        self.config.short_term_config = new_config
        self.logger.info("Short-term memory configuration updated")
    
    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self.session.state.session_id
    
    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """Get detailed memory usage statistics."""
        return {
            "session_summary": self.get_session_summary(),
            "memory_variables_size": len(str(self.get_memory_variables_for_langchain())),
            "full_context_size": len(self.get_full_context_for_prompt()),
            "conversation_context_size": len(self.get_conversation_context())
        }