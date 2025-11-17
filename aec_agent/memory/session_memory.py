"""
Session memory implementation for structured state tracking.

This module manages intermediate memory that persists throughout a session,
including goals, active files, subtasks, tool usage history, and other
structured knowledge about the current working context.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field
from langsmith import traceable


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status options for subtasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class GoalStatus(str, Enum):
    """Status of a goal."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Goal(BaseModel):
    """Represents a goal in the session."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_text: str = Field(..., description="The goal description")
    status: GoalStatus = Field(default=GoalStatus.ACTIVE)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(default=None)
    
    # Summary of goal's execution (created when goal completes)
    tool_executions_count: int = 0
    subtasks_count: int = 0
    success_rate: float = 1.0
    
    # Summarized data (created when goal completes)
    tool_history_summary: Optional[Dict[str, Any]] = None
    subtasks_summary: Optional[Dict[str, Any]] = None
    context_summary: Optional[str] = None


class SubTask(BaseModel):
    """Individual subtask within a session."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: Optional[str] = Field(default=None, description="ID of goal this subtask belongs to")
    name: str = Field(..., description="Description of the subtask")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dependencies: List[str] = Field(default_factory=list, description="IDs of dependent subtasks")
    notes: Optional[str] = Field(default=None, description="Additional notes")
    
    def mark_status(self, status: TaskStatus, notes: Optional[str] = None):
        """Update task status and timestamp."""
        self.status = status
        self.updated_at = datetime.now(timezone.utc)
        if notes:
            self.notes = notes


class ToolExecution(BaseModel):
    """Record of a tool execution."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal_id: Optional[str] = Field(default=None, description="ID of goal this execution belongs to")
    tool_name: str = Field(..., description="Name of the tool executed")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to tool")
    result_summary: Optional[str] = Field(default=None, description="Summary of tool result")
    execution_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = Field(default=True, description="Whether execution was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class SessionState(BaseModel):
    """Complete session state structure."""
    
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # High-level session context (legacy - keeping for backward compatibility)
    goal: Optional[str] = Field(default=None, description="Main goal for this session (legacy)")
    context: str = Field(default="", description="Additional context information")
    
    # Goal tracking (new goal-based system)
    current_goal_id: Optional[str] = Field(default=None, description="ID of currently active goal")
    goals: List[Goal] = Field(default_factory=list, description="All goals (active + completed)")
    completed_goals: List[Goal] = Field(default_factory=list, description="Completed goals with summaries")
    
    # File tracking
    active_files: List[str] = Field(default_factory=list, description="Currently active/opened files")
    modified_files: List[str] = Field(default_factory=list, description="Files modified in this session")
    
    # Task management
    subtasks: List[SubTask] = Field(default_factory=list, description="List of subtasks")
    current_subtask_id: Optional[str] = Field(default=None, description="ID of current active subtask")
    
    # Tool usage tracking
    tool_history: List[ToolExecution] = Field(default_factory=list, description="History of tool executions")
    
    # Building data context (AEC-specific)
    loaded_building_data: Optional[str] = Field(default=None, description="Path to currently loaded building data")
    building_context: Dict[str, Any] = Field(default_factory=dict, description="Building-specific context")
    
    # Compliance analysis context
    compliance_rules: List[str] = Field(default_factory=list, description="Active compliance rules")
    analysis_results: Dict[str, Any] = Field(default_factory=dict, description="Recent analysis results")
    
    # Reactive execution state tracking
    current_reasoning_mode: str = Field(default="static", description="Current reasoning mode (static/reactive)")
    active_execution_memory_id: Optional[str] = Field(default=None, description="ID of active execution memory session")
    execution_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of reactive execution state")
    accumulated_context: Dict[str, Any] = Field(default_factory=dict, description="Context accumulated during reactive execution")
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now(timezone.utc)


class SessionMemory:
    """
    Session memory manager for intermediate state tracking.
    
    Manages structured session state including goals, files, tasks, and tool usage.
    Provides persistence and retrieval capabilities for session data.
    """
    
    def __init__(self, session_id: Optional[str] = None, persistence_path: Optional[Path] = None):
        """
        Initialize session memory.
        
        Args:
            session_id: Optional existing session ID to load
            persistence_path: Optional path for session persistence
        """
        self.logger = logging.getLogger(__name__)
        self.persistence_path = persistence_path
        
        # Initialize or load session state
        if session_id and self.persistence_path:
            self.state = self._load_session(session_id)
        else:
            self.state = SessionState(session_id=session_id) if session_id else SessionState()
        
        self.logger.info(f"SessionMemory initialized with session_id={self.state.session_id}")
    
    # Goal and Context Management
    def set_goal(self, goal: str, context: str = "") -> str:
        """
        Set a new goal for the session.
        
        If there's an active goal, archive it first.
        Then create a new goal and set it as active.
        
        Args:
            goal: Goal description
            context: Additional context for the goal
            
        Returns:
            ID of the new goal
        """
        # Archive previous goal if exists
        if self.state.current_goal_id:
            self._archive_current_goal()
        
        # Create new goal
        new_goal = Goal(
            goal_text=goal,
            status=GoalStatus.ACTIVE
        )
        self.state.goals.append(new_goal)
        self.state.current_goal_id = new_goal.id
        
        # Reset context for new goal
        self.state.context = context
        self.state.update_timestamp()
        
        # Maintain backward compatibility
        self.state.goal = goal
        
        self.logger.info(f"New goal set: {goal[:50]}... (goal_id: {new_goal.id})")
        return new_goal.id
    
    def get_goal(self) -> Optional[str]:
        """Get the current session goal."""
        return self.state.goal
    
    def update_context(self, additional_context: str) -> None:
        """Add additional context information."""
        if self.state.context:
            self.state.context += f"\n{additional_context}"
        else:
            self.state.context = additional_context
        self.state.update_timestamp()
    
    def complete_current_goal(self, success: bool = True) -> None:
        """
        Mark the current goal as completed and archive it.
        
        Args:
            success: Whether the goal was completed successfully
        """
        if not self.state.current_goal_id:
            self.logger.warning("No active goal to complete")
            return
        
        goal = self._get_goal_by_id(self.state.current_goal_id)
        if not goal:
            return
        
        goal.status = GoalStatus.COMPLETED if success else GoalStatus.FAILED
        goal.completed_at = datetime.now(timezone.utc)
        
        # Archive the goal's data
        self._archive_goal_data(goal)
        
        # Move to completed goals
        self.state.completed_goals.append(goal)
        self.state.current_goal_id = None
        
        self.logger.info(f"Goal completed: {goal.goal_text[:50]}... (success: {success})")
    
    def _archive_current_goal(self) -> None:
        """Archive the current active goal when a new goal starts."""
        if not self.state.current_goal_id:
            return
        
        goal = self._get_goal_by_id(self.state.current_goal_id)
        if not goal:
            return
        
        # Mark as completed (or cancelled if we're replacing it)
        goal.status = GoalStatus.COMPLETED
        goal.completed_at = datetime.now(timezone.utc)
        
        # Archive the goal's data
        self._archive_goal_data(goal)
        
        # Move to completed goals
        self.state.completed_goals.append(goal)
        
        self.logger.info(f"Archived previous goal: {goal.goal_text[:50]}...")
    
    def _get_goal_by_id(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by its ID."""
        for goal in self.state.goals:
            if goal.id == goal_id:
                return goal
        return None
    
    @traceable(name="_archive_goal_data", metadata={"component": "session_memory"})
    def _archive_goal_data(self, goal: Goal) -> None:
        """
        Archive a goal's data (tool_history, subtasks, context).
        
        Creates summaries and removes detailed data from active lists.
        """
        goal_id = goal.id
        
        # Get all tool executions for this goal
        goal_tools = [t for t in self.state.tool_history if t.goal_id == goal_id]
        
        # Get all subtasks for this goal
        goal_subtasks = [s for s in self.state.subtasks if s.goal_id == goal_id]
        
        # Create summaries using rule-based approach
        if goal_tools:
            goal.tool_history_summary = self._create_tool_history_summary(goal_tools)
            goal.tool_executions_count = len(goal_tools)
            goal.success_rate = sum(1 for t in goal_tools if t.success) / len(goal_tools)
        
        if goal_subtasks:
            goal.subtasks_summary = self._create_subtask_summary(goal_subtasks)
            goal.subtasks_count = len(goal_subtasks)
        
        # Archive context
        if self.state.context:
            goal.context_summary = self.state.context[:500]  # Keep first 500 chars
        
        # Remove from active lists (keep only for active goal)
        self.state.tool_history = [t for t in self.state.tool_history if t.goal_id != goal_id]
        self.state.subtasks = [s for s in self.state.subtasks if s.goal_id != goal_id]
        
        self.logger.info(
            f"Archived goal data: {len(goal_tools)} tools, {len(goal_subtasks)} subtasks"
        )
    
    def _create_tool_history_summary(self, executions: List[ToolExecution]) -> Dict[str, Any]:
        """
        Create a statistical summary of tool executions.
        
        Returns a dict that can be stored as a summary in the goal.
        """
        if not executions:
            return {}
        
        # Group by tool name
        by_tool = {}
        for exec in executions:
            tool = exec.tool_name
            if tool not in by_tool:
                by_tool[tool] = []
            by_tool[tool].append(exec)
        
        # Calculate statistics
        total = len(executions)
        successful = sum(1 for e in executions if e.success)
        success_rate = successful / total if total > 0 else 0.0
        
        # Tool usage counts
        tool_counts = {tool: len(execs) for tool, execs in by_tool.items()}
        
        # Error patterns
        errors = [e.error_message for e in executions if e.error_message]
        error_patterns = {}
        for error in errors:
            # Extract error type (first part before colon or common patterns)
            error_type = error.split(':')[0] if ':' in error else error[:50]
            error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        # Time range
        times = [e.execution_time for e in executions if e.execution_time]
        time_range = {}
        if times:
            time_range = {
                "first": min(times).isoformat(),
                "last": max(times).isoformat()
            }
        
        return {
            "_type": "tool_history_summary",
            "total_executions": total,
            "success_rate": success_rate,
            "tool_usage": tool_counts,
            "error_patterns": error_patterns,
            "time_range": time_range,
            "summary_created_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _create_subtask_summary(self, tasks: List[SubTask]) -> Dict[str, Any]:
        """
        Create a theme-based summary of subtasks.
        
        Returns a dict that can be stored as a summary in the goal.
        """
        if not tasks:
            return {}
        
        # Group by common patterns in task names
        # Extract key themes (e.g., "Load", "Analyze", "Check")
        task_themes = {}
        for task in tasks:
            # Simple heuristic: first word or common prefix
            first_word = task.name.split()[0] if task.name else "Other"
            if first_word not in task_themes:
                task_themes[first_word] = []
            task_themes[first_word].append(task.name)
        
        # Calculate statistics
        total_tasks = len(tasks)
        status_counts = {}
        for task in tasks:
            status = task.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Theme counts
        theme_counts = {theme: len(names) for theme, names in task_themes.items()}
        
        # Time range
        times = [t.updated_at for t in tasks if t.updated_at]
        time_range = {}
        if times:
            time_range = {
                "first": min(times).isoformat(),
                "last": max(times).isoformat()
            }
        
        return {
            "_type": "subtask_summary",
            "total_tasks": total_tasks,
            "status_distribution": status_counts,
            "task_themes": theme_counts,
            "time_range": time_range,
            "summary_created_at": datetime.now(timezone.utc).isoformat()
        }
    
    # File Management
    def add_active_file(self, file_path: str) -> None:
        """Add a file to the active files list."""
        if file_path not in self.state.active_files:
            self.state.active_files.append(file_path)
            self.state.update_timestamp()
            self.logger.debug(f"Added active file: {file_path}")
    
    def remove_active_file(self, file_path: str) -> None:
        """Remove a file from the active files list."""
        if file_path in self.state.active_files:
            self.state.active_files.remove(file_path)
            self.state.update_timestamp()
            self.logger.debug(f"Removed active file: {file_path}")
    
    def mark_file_modified(self, file_path: str) -> None:
        """Mark a file as modified in this session."""
        if file_path not in self.state.modified_files:
            self.state.modified_files.append(file_path)
            self.state.update_timestamp()
            self.logger.debug(f"Marked file as modified: {file_path}")
    
    # Task Management
    def add_subtask(self, name: str, dependencies: Optional[List[str]] = None) -> str:
        """
        Add a new subtask to the session, automatically associating it with current goal.
        
        Args:
            name: Description of the subtask
            dependencies: Optional list of subtask IDs this task depends on
            
        Returns:
            ID of the created subtask
        """
        subtask = SubTask(
            goal_id=self.state.current_goal_id,  # Automatically set
            name=name, 
            dependencies=dependencies or []
        )
        self.state.subtasks.append(subtask)
        self.state.update_timestamp()
        self.logger.info(f"Added subtask: {name} (goal_id: {subtask.goal_id})")
        return subtask.id
    
    def update_subtask_status(self, subtask_id: str, status: TaskStatus, notes: Optional[str] = None) -> bool:
        """
        Update the status of a subtask.
        
        Args:
            subtask_id: ID of the subtask to update
            status: New status
            notes: Optional notes about the status change
            
        Returns:
            True if subtask was found and updated, False otherwise
        """
        for subtask in self.state.subtasks:
            if subtask.id == subtask_id:
                subtask.mark_status(status, notes)
                self.state.update_timestamp()
                self.logger.info(f"Updated subtask {subtask_id} to {status}")
                return True
        return False
    
    def set_current_subtask(self, subtask_id: str) -> bool:
        """Set the currently active subtask."""
        if any(task.id == subtask_id for task in self.state.subtasks):
            self.state.current_subtask_id = subtask_id
            self.state.update_timestamp()
            return True
        return False
    
    def get_pending_subtasks(self) -> List[SubTask]:
        """Get all pending subtasks."""
        return [task for task in self.state.subtasks if task.status == TaskStatus.PENDING]
    
    def get_current_subtask(self) -> Optional[SubTask]:
        """Get the currently active subtask."""
        if self.state.current_subtask_id:
            for task in self.state.subtasks:
                if task.id == self.state.current_subtask_id:
                    return task
        return None
    
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
        Record a tool execution, automatically associating it with current goal.
        
        Args:
            tool_name: Name of the executed tool
            arguments: Arguments passed to the tool
            success: Whether execution was successful
            result_summary: Summary of the result
            error_message: Error message if failed
            
        Returns:
            ID of the tool execution record
        """
        execution = ToolExecution(
            goal_id=self.state.current_goal_id,  # Automatically set
            tool_name=tool_name,
            arguments=arguments,
            success=success,
            result_summary=result_summary,
            error_message=error_message
        )
        self.state.tool_history.append(execution)
        self.state.update_timestamp()
        self.logger.debug(f"Recorded tool execution: {tool_name} (goal_id: {execution.goal_id})")
        return execution.id
    
    def get_recent_tool_executions(self, limit: int = 10, goal_id: Optional[str] = None) -> List[ToolExecution]:
        """
        Get the most recent tool executions.
        
        Args:
            limit: Maximum number of executions to return
            goal_id: Optional goal ID to filter by. If None, uses current goal.
        """
        target_goal_id = goal_id or self.state.current_goal_id
        
        if target_goal_id:
            # Filter by goal
            executions = [t for t in self.state.tool_history if t.goal_id == target_goal_id]
        else:
            # No goal filter, return all
            executions = self.state.tool_history
        
        return sorted(
            executions, 
            key=lambda x: x.execution_time, 
            reverse=True
        )[:limit]
    
    def get_current_goal_subtasks(self) -> List[SubTask]:
        """Get subtasks for the current goal."""
        if not self.state.current_goal_id:
            return []
        
        return [
            task for task in self.state.subtasks 
            if task.goal_id == self.state.current_goal_id
        ]
    
    def get_completed_goals_summary(self) -> List[Dict[str, Any]]:
        """Get summaries of completed goals."""
        return [
            {
                "goal_id": goal.id,
                "goal_text": goal.goal_text,
                "status": goal.status,
                "completed_at": goal.completed_at,
                "tool_executions_count": goal.tool_executions_count,
                "subtasks_count": goal.subtasks_count,
                "success_rate": goal.success_rate,
                "tool_history_summary": goal.tool_history_summary,
                "subtasks_summary": goal.subtasks_summary
            }
            for goal in self.state.completed_goals
        ]
    
    def get_current_goal_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current active goal."""
        if not self.state.current_goal_id:
            return None
        
        goal = self._get_goal_by_id(self.state.current_goal_id)
        if not goal:
            return None
        
        return {
            "goal_id": goal.id,
            "goal_text": goal.goal_text,
            "status": goal.status,
            "created_at": goal.created_at,
            "current_subtasks": len(self.get_current_goal_subtasks()),
            "current_tool_executions": len([t for t in self.state.tool_history if t.goal_id == goal.id])
        }
    
    # Building Data Context (AEC-specific)
    def set_building_data(self, data_path: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Set the currently loaded building data."""
        self.state.loaded_building_data = data_path
        if context:
            self.state.building_context.update(context)
        self.state.update_timestamp()
        self.logger.info(f"Set building data: {data_path}")
    
    def update_building_context(self, context: Dict[str, Any]) -> None:
        """Update building-specific context."""
        self.state.building_context.update(context)
        self.state.update_timestamp()
    
    def add_compliance_rule(self, rule: str) -> None:
        """Add a compliance rule to the active rules list."""
        if rule not in self.state.compliance_rules:
            self.state.compliance_rules.append(rule)
            self.state.update_timestamp()
    
    def update_analysis_results(self, results: Dict[str, Any]) -> None:
        """Update the latest analysis results."""
        self.state.analysis_results.update(results)
        self.state.update_timestamp()
    
    # Execution State Management (for Reactive Reasoning)
    def set_reasoning_mode(self, mode: str) -> None:
        """Set the current reasoning mode (static/reactive)."""
        self.state.current_reasoning_mode = mode
        self.state.update_timestamp()
        self.logger.debug(f"Reasoning mode set to: {mode}")
    
    def get_reasoning_mode(self) -> str:
        """Get the current reasoning mode."""
        return self.state.current_reasoning_mode
    
    def set_active_execution_memory(self, execution_memory_id: str) -> None:
        """Set the active execution memory session ID."""
        self.state.active_execution_memory_id = execution_memory_id
        self.state.update_timestamp()
        self.logger.debug(f"Active execution memory set: {execution_memory_id}")
    
    def get_active_execution_memory_id(self) -> Optional[str]:
        """Get the active execution memory session ID."""
        return self.state.active_execution_memory_id
    
    def update_execution_summary(self, summary: Dict[str, Any]) -> None:
        """Update the execution summary with latest reactive execution state."""
        self.state.execution_summary.update(summary)
        self.state.update_timestamp()
        self.logger.debug("Execution summary updated")
    
    def update_accumulated_context(self, context: Dict[str, Any]) -> None:
        """Update accumulated context from reactive execution."""
        self.state.accumulated_context.update(context)
        self.state.update_timestamp()
        self.logger.debug(f"Accumulated context updated with {len(context)} new entries")
    
    def get_accumulated_context(self) -> Dict[str, Any]:
        """Get the accumulated context from reactive execution."""
        return self.state.accumulated_context.copy()
    
    def clear_execution_state(self) -> None:
        """Clear reactive execution state."""
        self.state.current_reasoning_mode = "static"
        self.state.active_execution_memory_id = None
        self.state.execution_summary.clear()
        self.state.accumulated_context.clear()
        self.state.update_timestamp()
        self.logger.info("Execution state cleared")
    
    # Session Management
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session state."""
        return {
            "session_id": self.state.session_id,
            "goal": self.state.goal,
            "active_files_count": len(self.state.active_files),
            "modified_files_count": len(self.state.modified_files),
            "total_subtasks": len(self.state.subtasks),
            "pending_subtasks": len(self.get_pending_subtasks()),
            "current_subtask": self.get_current_subtask().name if self.get_current_subtask() else None,
            "tool_executions": len(self.state.tool_history),
            "building_data_loaded": bool(self.state.loaded_building_data),
            "compliance_rules_active": len(self.state.compliance_rules),
            "reasoning_mode": self.state.current_reasoning_mode,
            "active_execution_memory": bool(self.state.active_execution_memory_id),
            "accumulated_context_size": len(self.state.accumulated_context),
            "created_at": self.state.created_at,
            "updated_at": self.state.updated_at
        }
    
    def get_context_for_prompt(self) -> str:
        """
        Get formatted context information for LLM prompt injection.
        
        Returns:
            Formatted string with session context
        """
        context_parts = []
        
        # Goal and context
        if self.state.goal:
            context_parts.append(f"Session Goal: {self.state.goal}")
        if self.state.context:
            context_parts.append(f"Context: {self.state.context}")
        
        # Current subtask
        current_task = self.get_current_subtask()
        if current_task:
            context_parts.append(f"Current Task: {current_task.name} (Status: {current_task.status})")
        
        # File context
        if self.state.active_files:
            context_parts.append(f"Active Files: {', '.join(self.state.active_files[-3:])}")  # Show last 3
        
        # Building data context
        if self.state.loaded_building_data:
            context_parts.append(f"Loaded Building Data: {self.state.loaded_building_data}")
            # Include summary if available (not the full JSON)
            if self.state.building_context:
                project_name = self.state.building_context.get("project_name", "Unknown")
                total_elements = self.state.building_context.get("total_elements", 0)
                
                # Format element type counts (e.g., "5 stairs, 2 walls, 10 doors")
                element_types = self.state.building_context.get("available_element_types", {})
                element_counts = []
                for elem_type, count in element_types.items():
                    if count > 0:
                        # Format with proper pluralization
                        if count == 1:
                            # Singular forms
                            singular_map = {
                                "spaces": "space",
                                "doors": "door",
                                "walls": "wall",
                                "slabs": "slab",
                                "stairs": "stair"
                            }
                            elem_name = singular_map.get(elem_type, elem_type.rstrip('s'))
                        else:
                            elem_name = elem_type
                        element_counts.append(f"{count} {elem_name}")
                
                element_summary = ", ".join(element_counts) if element_counts else "no elements"
                context_parts.append(
                    f"Building Context: {project_name} with {total_elements} total elements "
                    f"({element_summary})"
                )
        
        # Recent tool usage
        recent_tools = self.get_recent_tool_executions(3)
        if recent_tools:
            tool_names = [t.tool_name for t in recent_tools]
            context_parts.append(f"Recent Tools: {', '.join(tool_names)}")
        
        return "\n".join(context_parts)
    
    # Persistence
    def save_session(self) -> None:
        """Save session state to disk if persistence path is set."""
        if not self.persistence_path:
            return
        
        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            session_file = self.persistence_path / f"{self.state.session_id}.json"
            
            with session_file.open("w") as f:
                json.dump(self.state.model_dump(), f, indent=2, default=str)
            
            self.logger.debug(f"Session saved to {session_file}")
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
    
    def _load_session(self, session_id: str) -> SessionState:
        """Load session state from disk."""
        if not self.persistence_path:
            return SessionState(session_id=session_id)
        
        try:
            session_file = self.persistence_path / f"{session_id}.json"
            if session_file.exists():
                with session_file.open("r") as f:
                    data = json.load(f)
                return SessionState(**data)
            else:
                self.logger.warning(f"Session file not found: {session_file}")
                return SessionState(session_id=session_id)
        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")
            return SessionState(session_id=session_id)
    
    def clear_session(self) -> None:
        """Clear the current session state."""
        old_session_id = self.state.session_id
        self.state = SessionState()
        self.logger.info(f"Session cleared (was {old_session_id})")
    
    # Token-Based Memory Management
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Simple heuristic: ~4 characters per token (conservative estimate).
        For more accuracy, could use tiktoken library.
        """
        return len(text) // 4
    
    def estimate_session_memory_tokens(self) -> int:
        """
        Estimate total token count for session memory.
        
        Includes:
        - tool_history (all executions)
        - subtasks (all tasks)
        - context string
        - completed_goals summaries
        - active_files, modified_files (small)
        - accumulated_context
        """
        total = 0
        
        # Tool history
        for exec in self.state.tool_history:
            total += self.estimate_tokens(str(exec.model_dump()))
        
        # Subtasks
        for task in self.state.subtasks:
            total += self.estimate_tokens(str(task.model_dump()))
        
        # Context
        total += self.estimate_tokens(self.state.context)
        
        # Completed goals summaries
        for goal in self.state.completed_goals:
            if goal.tool_history_summary:
                total += self.estimate_tokens(str(goal.tool_history_summary))
            if goal.subtasks_summary:
                total += self.estimate_tokens(str(goal.subtasks_summary))
            if goal.context_summary:
                total += self.estimate_tokens(goal.context_summary)
        
        # Accumulated context
        total += self.estimate_tokens(str(self.state.accumulated_context))
        
        return total
    
    def check_and_trigger_compaction(self, session_token_cap: int = 12000, 
                                   session_token_warning_threshold: int = 10000) -> bool:
        """
        Check if memory exceeds token cap and trigger compaction if needed.
        
        Args:
            session_token_cap: Maximum tokens allowed before triggering compaction
            session_token_warning_threshold: Warning threshold
        
        Returns:
            True if compaction was triggered, False otherwise
        """
        current_tokens = self.estimate_session_memory_tokens()
        
        # Check if we've exceeded the cap
        if current_tokens > session_token_cap:
            self.logger.warning(
                f"Session memory token cap exceeded: {current_tokens} > {session_token_cap}. "
                f"Triggering compaction... (Memory should stay under cap to leave room for prompts/responses)"
            )
            
            # Trigger comprehensive compaction
            self.compact_session_memory()
            return True
        
        # Warning threshold
        elif current_tokens > session_token_warning_threshold:
            self.logger.info(
                f"Session memory approaching token cap: {current_tokens}/{session_token_cap} "
                f"({current_tokens/session_token_cap*100:.1f}%)"
            )
        
        return False
    
    @traceable(name="compact_session_memory", metadata={"component": "session_memory"})
    def compact_session_memory(self) -> Dict[str, Any]:
        """
        Compact session memory by applying goal-based archiving and cleanup.
        
        Note: Most compaction happens automatically through goal lifecycle.
        This method handles additional cleanup for edge cases.
        
        Returns:
            Dict with compaction statistics
        """
        stats = {
            "tool_history_before": len(self.state.tool_history),
            "subtasks_before": len(self.state.subtasks),
            "active_files_before": len(self.state.active_files),
            "modified_files_before": len(self.state.modified_files),
            "context_length_before": len(self.state.context),
            "accumulated_context_before": len(self.state.accumulated_context),
            "completed_goals_before": len(self.state.completed_goals)
        }
        
        # Compact file tracking
        self._compact_file_tracking()
        
        # Compact context if too long
        self._compact_context()
        
        # Compact accumulated context
        self._compact_accumulated_context()
        
        # Compact completed goals if needed
        self._compact_completed_goals()
        
        stats.update({
            "tool_history_after": len(self.state.tool_history),
            "subtasks_after": len(self.state.subtasks),
            "active_files_after": len(self.state.active_files),
            "modified_files_after": len(self.state.modified_files),
            "context_length_after": len(self.state.context),
            "accumulated_context_after": len(self.state.accumulated_context),
            "completed_goals_after": len(self.state.completed_goals)
        })
        
        self.logger.info(f"Session memory compacted. Token savings: {stats}")
        return stats
    
    def _compact_file_tracking(self, keep_active: int = 10, keep_modified: int = 20) -> None:
        """Compact file tracking lists."""
        # Active files: simple truncation (FIFO)
        if len(self.state.active_files) > keep_active:
            removed = self.state.active_files[:-keep_active]
            self.state.active_files = self.state.active_files[-keep_active:]
            self.logger.debug(f"Removed {len(removed)} old active files")
        
        # Modified files: keep recent
        if len(self.state.modified_files) > keep_modified:
            recent_modified = self.state.modified_files[-keep_modified:]
            older_modified = self.state.modified_files[:-keep_modified]
            self.state.modified_files = recent_modified
            self.logger.debug(f"Compacted modified files: {len(older_modified)} older files removed")
        
        self.state.update_timestamp()
    
    def _compact_context(self, max_length: int = 2000) -> None:
        """Compact context string if it exceeds max_length."""
        if len(self.state.context) <= max_length:
            return
        
        # Split by newlines to preserve structure
        lines = self.state.context.split('\n')
        
        # Keep recent lines (last N lines that fit in limit)
        recent_lines = []
        current_length = 0
        for line in reversed(lines):
            if current_length + len(line) + 1 <= max_length * 0.7:  # Use 70% for recent
                recent_lines.insert(0, line)
                current_length += len(line) + 1
            else:
                break
        
        # Older lines to summarize
        older_lines = lines[:len(lines) - len(recent_lines)]
        
        if older_lines:
            # Simple truncation approach
            older_summary = f"[Previous context summarized: {len(older_lines)} lines]\n"
            self.state.context = older_summary + '\n'.join(recent_lines)
            self.logger.info(f"Compacted context: {len(older_lines)} lines summarized")
        
        self.state.update_timestamp()
    
    def _compact_accumulated_context(self, max_entries: int = 50) -> None:
        """Compact accumulated_context dict."""
        if len(self.state.accumulated_context) <= max_entries:
            return
        
        # Convert to list of (key, value) tuples
        items = list(self.state.accumulated_context.items())
        
        # Keep most recent N entries
        keep_items = items[-max_entries:]
        
        # Rebuild dict
        self.state.accumulated_context = dict(keep_items)
        
        self.logger.info(
            f"Compacted accumulated_context: {len(items) - max_entries} entries removed, "
            f"{max_entries} kept"
        )
        self.state.update_timestamp()
    
    def _compact_completed_goals(self, keep_recent: int = 10) -> None:
        """Compact completed goals list if it gets too large."""
        if len(self.state.completed_goals) <= keep_recent:
            return
        
        recent = self.state.completed_goals[-keep_recent:]
        older = self.state.completed_goals[:-keep_recent]
        
        self.state.completed_goals = recent
        
        self.logger.info(
            f"Compacted completed goals: removed {len(older)} older goals, "
            f"keeping {len(recent)} recent"
        )