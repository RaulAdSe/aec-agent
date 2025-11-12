"""
Shared utilities for the reasoning system.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import tiktoken


class TaskStatus(Enum):
    """Status of a task or subtask."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Priority(Enum):
    """Priority levels for tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a task in the reasoning system."""
    id: str
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    dependencies: List[str] = None
    tool_sequence: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tool_sequence is None:
            self.tool_sequence = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionResult:
    """Result of tool execution."""
    success: bool
    tool_name: str
    output: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ReasoningUtils:
    """Utility functions for the reasoning system."""
    
    @staticmethod
    def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
        """Set up a logger for a reasoning component."""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Avoid duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """Count tokens in text for the given model."""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback estimation: roughly 4 characters per token
            return len(text) // 4
    
    @staticmethod
    def format_task_summary(tasks: List[Task]) -> str:
        """Format a summary of tasks."""
        if not tasks:
            return "No tasks"
        
        status_counts = {}
        for task in tasks:
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        summary_parts = []
        for status, count in status_counts.items():
            summary_parts.append(f"{count} {status}")
        
        return f"Tasks: {', '.join(summary_parts)} (total: {len(tasks)})"
    
    @staticmethod
    def validate_tool_output(output: Any, expected_format: str = "dict") -> bool:
        """Validate that tool output matches expected format."""
        if expected_format == "dict":
            return isinstance(output, dict) and "status" in output
        elif expected_format == "list":
            return isinstance(output, list)
        elif expected_format == "str":
            return isinstance(output, str)
        return True
    
    @staticmethod
    def extract_error_info(error: Exception) -> Dict[str, str]:
        """Extract useful information from an exception."""
        return {
            "type": type(error).__name__,
            "message": str(error),
            "module": getattr(error, "__module__", "unknown")
        }
    
    @staticmethod
    def calculate_task_progress(tasks: List[Task]) -> float:
        """Calculate overall progress percentage for a list of tasks."""
        if not tasks:
            return 0.0
        
        completed = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        return (completed / len(tasks)) * 100.0
    
    @staticmethod
    def find_ready_tasks(tasks: List[Task]) -> List[Task]:
        """Find tasks that are ready to execute (no pending dependencies)."""
        ready_tasks = []
        completed_task_ids = {
            task.id for task in tasks 
            if task.status == TaskStatus.COMPLETED
        }
        
        for task in tasks:
            if task.status == TaskStatus.PENDING:
                if not task.dependencies or all(
                    dep_id in completed_task_ids 
                    for dep_id in task.dependencies
                ):
                    ready_tasks.append(task)
        
        # Sort by priority
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        return ready_tasks