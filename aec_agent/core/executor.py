"""
Tool Executor - Executes tools with monitoring and error handling.

This module runs individual tools while providing real-time monitoring,
error recovery, and execution tracking for the reasoning system.
"""

import time
import json
from typing import Any, Dict, Callable, Optional

from .reasoning_utils import ReasoningUtils, Task, ExecutionResult


class ToolExecutor:
    """
    Executes tools with monitoring, error handling, and result validation.
    
    Provides a controlled environment for tool execution with:
    - Real-time monitoring
    - Error recovery
    - Execution tracking
    - Resource management
    """
    
    def __init__(self, tool_registry: Dict[str, Callable], timeout: float = 60.0):
        """
        Initialize the tool executor.
        
        Args:
            tool_registry: Map of tool names to callable functions
            timeout: Maximum execution time per tool call in seconds
        """
        self.tool_registry = tool_registry
        self.timeout = timeout
        self.logger = ReasoningUtils.setup_logger(__name__)
        self.execution_history = []
    
    def execute_tool(
        self, 
        tool_name: str, 
        task: Task, 
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """
        Execute a single tool for a given task.
        
        Args:
            tool_name: Name of the tool to execute
            task: Task being executed
            context: Execution context and parameters
            
        Returns:
            ExecutionResult with success status, output, and metadata
        """
        start_time = time.time()
        
        self.logger.info(f"Executing tool '{tool_name}' for task '{task.name}'")
        
        # Validate tool exists
        if tool_name not in self.tool_registry:
            return ExecutionResult(
                success=False,
                tool_name=tool_name,
                output=None,
                error_message=f"Tool '{tool_name}' not found in registry",
                execution_time=0.0
            )
        
        # Get tool function
        tool_function = self.tool_registry[tool_name]
        
        try:
            # Prepare tool input based on tool type
            tool_input = self._prepare_tool_input(tool_name, task, context)
            
            self.logger.debug(f"Tool input for {tool_name}: {str(tool_input)[:200]}...")
            
            # Execute tool
            output = tool_function(tool_input)
            
            execution_time = time.time() - start_time
            
            # Validate output format
            is_valid, validation_msg = self._validate_output(tool_name, output)
            
            if not is_valid:
                return ExecutionResult(
                    success=False,
                    tool_name=tool_name,
                    output=output,
                    error_message=f"Invalid output format: {validation_msg}",
                    execution_time=execution_time
                )
            
            # Create successful result
            result = ExecutionResult(
                success=True,
                tool_name=tool_name,
                output=output,
                execution_time=execution_time,
                metadata={
                    "task_id": task.id,
                    "tool_input_preview": str(tool_input)[:200],
                    "output_size": len(str(output))
                }
            )
            
            self.logger.info(f"Tool '{tool_name}' executed successfully in {execution_time:.2f}s")
            self._record_execution(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool execution failed: {str(e)}"
            self.logger.error(f"{error_msg} ({tool_name})")
            
            result = ExecutionResult(
                success=False,
                tool_name=tool_name,
                output=None,
                error_message=error_msg,
                execution_time=execution_time,
                metadata={"error_type": type(e).__name__}
            )
            self._record_execution(result)
            return result
    
    def _prepare_tool_input(self, tool_name: str, task: Task, context: Dict[str, Any]) -> Any:
        """Prepare appropriate input for the tool based on its requirements."""
        
        # Tool-specific input preparation
        if tool_name == "load_building_data":
            return self._extract_file_path(task, context)
            
        elif tool_name == "get_all_elements":
            return self._extract_element_type(task)
            
        elif tool_name == "query_elements":
            return self._prepare_query_input(task, context)
            
        elif tool_name == "calculate_metrics":
            return self._prepare_calculation_input(task, context)
            
        elif tool_name == "find_related_elements":
            return self._prepare_relationship_input(task, context)
            
        elif tool_name == "validate_compliance_rule":
            return self._prepare_validation_input(task, context)
            
        elif tool_name == "search_compliance_documents":
            return self._extract_search_query(task)
            
        else:
            # No generic fallback - fail explicitly for unknown tools
            raise ValueError(f"No input preparation method for tool: {tool_name}")
    
    def _extract_file_path(self, task: Task, context: Dict[str, Any]) -> str:
        """Extract file path for building data loading."""
        # Check task metadata first
        if "file_path" in task.metadata:
            return task.metadata["file_path"]
        
        # Check context for active files
        active_files = context.get("active_files", [])
        if active_files:
            for file_path in active_files:
                if file_path.endswith('.json'):
                    return file_path
        
        # Look for path in task description
        desc = task.description.lower()
        if "data/" in desc or ".json" in desc:
            words = desc.split()
            for word in words:
                if ".json" in word:
                    return word.strip('",.')
        
        # No fallback - fail explicitly if no path found
        raise ValueError(f"No file path found in task: {task.name} - {task.description}")
    
    def _extract_element_type(self, task: Task) -> str:
        """Extract element type from task description."""
        desc = task.description.lower()
        
        # Check if task is asking for all elements (count, extract all, etc.)
        if any(term in desc for term in ["count", "all", "total", "extract", "process"]):
            # Look at the original goal or task metadata for more context
            original_goal = task.metadata.get("original_goal", "").lower()
            if "count" in original_goal or "all" in original_goal:
                return "spaces"  # Default to spaces for all-element queries
        
        if any(term in desc for term in ["door", "doors"]):
            return "doors"
        elif any(term in desc for term in ["wall", "walls"]):
            return "walls"
        elif any(term in desc for term in ["space", "spaces", "room", "rooms"]):
            return "spaces"
        elif any(term in desc for term in ["stair", "stairs"]):
            return "stairs"
        elif any(term in desc for term in ["slab", "slabs", "floor", "floors"]):
            return "slabs"
        else:
            raise ValueError(f"Cannot determine element type from task description: {desc}")
    
    def _prepare_query_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare JSON input for query_elements tool."""
        element_type = self._extract_element_type(task)
        filters = {}
        desc = task.description.lower()
        
        # Fire-related filters
        if any(term in desc for term in ["fire", "emergency", "exit"]):
            filters["fire_related"] = True
        
        # Size filters
        if "wider than" in desc or "width" in desc:
            words = desc.split()
            for i, word in enumerate(words):
                if word in ["wider", "width"] and i + 2 < len(words):
                    try:
                        width = float(words[i + 2].replace("m", ""))
                        filters["min_width"] = width
                    except ValueError:
                        pass
        
        return json.dumps({
            "element_type": element_type,
            "filters": filters
        })
    
    def _prepare_calculation_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare JSON input for calculate_metrics tool."""
        desc = task.description.lower()
        
        if any(term in desc for term in ["area", "floor area", "surface"]):
            operation = "total_area"
            params = {"element_type": "spaces"}
        elif any(term in desc for term in ["volume", "cubic"]):
            operation = "total_volume"
            params = {"element_type": "spaces"}
        elif any(term in desc for term in ["count", "number", "how many"]):
            operation = "count"
            params = {"element_type": self._extract_element_type(task)}
        else:
            operation = "total_area"
            params = {"element_type": "spaces"}
        
        return json.dumps({
            "operation": operation,
            **params
        })
    
    def _prepare_relationship_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare JSON input for find_related_elements tool."""
        return json.dumps({
            "element_id": "element_to_be_determined",
            "relationship_type": "adjacent",
            "parameters": {}
        })
    
    def _prepare_validation_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare JSON input for validate_compliance_rule tool."""
        desc = task.description.lower()
        
        if any(term in desc for term in ["fire", "emergency"]):
            rule_type = "fire_safety"
        elif any(term in desc for term in ["accessibility", "ada"]):
            rule_type = "accessibility"
        elif any(term in desc for term in ["structural", "load"]):
            rule_type = "structural"
        else:
            rule_type = "general"
        
        return json.dumps({
            "rule_type": rule_type,
            "element_id": "element_to_be_determined", 
            "criteria": {"extracted_from": task.description}
        })
    
    def _extract_search_query(self, task: Task) -> str:
        """Extract search query from task description."""
        desc = task.description
        
        # Remove common prefixes
        prefixes_to_remove = [
            "retrieve ", "search for ", "find ", "get ",
            "search compliance documents for ",
            "retrieve compliance rules for "
        ]
        
        query = desc.lower()
        for prefix in prefixes_to_remove:
            if query.startswith(prefix):
                query = query[len(prefix):]
                break
        
        return query.strip()
    
    def _validate_output(self, tool_name: str, output: Any) -> tuple[bool, str]:
        """Validate that tool output matches expected format."""
        
        if not isinstance(output, dict):
            return False, f"Expected dict output, got {type(output).__name__}"
        
        if "status" not in output:
            return False, "Missing 'status' field in output"
        
        status = output.get("status")
        if status not in ["success", "error", "partial"]:
            return False, f"Invalid status value: {status}"
        
        # Tool-specific validations
        if status == "success":
            if tool_name == "load_building_data":
                if "data" not in output:
                    return False, "Missing 'data' field for successful data load"
        
        return True, "Output validation passed"
    
    def _record_execution(self, result: ExecutionResult) -> None:
        """Record execution result in history."""
        execution_record = {
            "timestamp": time.time(),
            "tool_name": result.tool_name,
            "success": result.success,
            "execution_time": result.execution_time,
            "error_message": result.error_message
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about tool execution history."""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total = len(self.execution_history)
        successful = sum(1 for record in self.execution_history if record["success"])
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": (successful / total) * 100 if total > 0 else 0
        }