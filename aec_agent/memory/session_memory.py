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


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status options for subtasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class SubTask(BaseModel):
    """Individual subtask within a session."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
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
    
    # High-level session context
    goal: Optional[str] = Field(default=None, description="Main goal for this session")
    context: str = Field(default="", description="Additional context information")
    
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
    def set_goal(self, goal: str, context: str = "") -> None:
        """Set the main goal for this session."""
        self.state.goal = goal
        self.state.context = context
        self.state.update_timestamp()
        self.logger.info(f"Session goal set: {goal[:50]}...")
    
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
        Add a new subtask to the session.
        
        Args:
            name: Description of the subtask
            dependencies: Optional list of subtask IDs this task depends on
            
        Returns:
            ID of the created subtask
        """
        subtask = SubTask(name=name, dependencies=dependencies or [])
        self.state.subtasks.append(subtask)
        self.state.update_timestamp()
        self.logger.info(f"Added subtask: {name}")
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
        Record a tool execution.
        
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
            tool_name=tool_name,
            arguments=arguments,
            success=success,
            result_summary=result_summary,
            error_message=error_message
        )
        self.state.tool_history.append(execution)
        self.state.update_timestamp()
        self.logger.debug(f"Recorded tool execution: {tool_name}")
        return execution.id
    
    def get_recent_tool_executions(self, limit: int = 10) -> List[ToolExecution]:
        """Get the most recent tool executions."""
        return sorted(
            self.state.tool_history, 
            key=lambda x: x.execution_time, 
            reverse=True
        )[:limit]
    
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