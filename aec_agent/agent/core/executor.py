"""
Tool Executor - Executes tools with monitoring and error handling.

This module runs individual tools while providing real-time monitoring,
error recovery, and execution tracking for the reasoning system.
"""

import time
import json
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from .utils import ReasoningUtils, Task, ExecutionResult


@dataclass
class ExecutionContext:
    """Context information for tool execution."""
    task: Task
    goal: str
    iteration: int
    session_context: Dict[str, Any]
    execution_history: list


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
            
            self.logger.debug(f"Tool input for {tool_name}: {tool_input}")
            
            # Execute tool with timeout monitoring
            output = self._execute_with_timeout(tool_function, tool_input, tool_name)
            
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
                    "tool_input": str(tool_input)[:200],  # Truncate for logging
                    "output_size": len(str(output))
                }
            )
            
            self.logger.info(f"Tool '{tool_name}' executed successfully in {execution_time:.2f}s")
            self._record_execution(result)
            
            return result
            
        except TimeoutError as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool execution timed out after {self.timeout}s"
            self.logger.error(f"{error_msg}: {tool_name}")
            
            result = ExecutionResult(
                success=False,
                tool_name=tool_name,
                output=None,
                error_message=error_msg,
                execution_time=execution_time
            )
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
            # Need file path - try to get from task metadata or context
            file_path = self._extract_file_path(task, context)
            return file_path
            
        elif tool_name == "get_all_elements":
            # Need element type - extract from task description
            element_type = self._extract_element_type(task)
            return element_type
            
        elif tool_name == "query_elements":
            # Need JSON with element_type and filters
            return self._prepare_query_input(task, context)
            
        elif tool_name == "calculate_metrics":
            # Need JSON with operation and parameters
            return self._prepare_calculation_input(task, context)
            
        elif tool_name == "find_related_elements":
            # Need JSON with element_id, relationship_type, and parameters
            return self._prepare_relationship_input(task, context)
            
        elif tool_name == "validate_compliance_rule":
            # Need JSON with rule_type, element_id, and criteria
            return self._prepare_validation_input(task, context)
            
        elif tool_name == "search_compliance_documents":
            # Need query text
            return self._extract_search_query(task)
            
        else:
            # Default: use task description as input
            return task.description
    
    def _extract_file_path(self, task: Task, context: Dict[str, Any]) -> str:
        """Extract file path for building data loading."""
        # Check task metadata first
        if "file_path" in task.metadata:
            return task.metadata["file_path"]
        
        # Check context for active files
        active_files = context.get("active_files", [])
        if active_files:
            # Use first JSON file found
            for file_path in active_files:
                if file_path.endswith('.json'):
                    return file_path
        
        # Look for path in task description
        desc = task.description.lower()
        if "data/" in desc or ".json" in desc:
            # Try to extract path from description
            words = desc.split()
            for word in words:
                if ".json" in word:
                    return word.strip('",.')
        
        # Default fallback
        return "data/out/building_data.json"
    
    def _extract_element_type(self, task: Task) -> str:
        """Extract element type from task description."""
        desc = task.description.lower()
        
        # Common element type mappings
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
        elif any(term in desc for term in ["window", "windows"]):
            return "windows"
        else:
            # Default to spaces as most generic
            return "spaces"
    
    def _prepare_query_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare JSON input for query_elements tool."""
        element_type = self._extract_element_type(task)
        
        # Extract filters from task description
        filters = {}
        desc = task.description.lower()
        
        # Fire-related filters
        if any(term in desc for term in ["fire", "emergency", "exit"]):
            filters["fire_related"] = True
        
        # Size/dimension filters
        if "wider than" in desc or "width" in desc:
            # Try to extract numeric value
            words = desc.split()
            for i, word in enumerate(words):
                if word in ["wider", "width"] and i + 2 < len(words):
                    try:
                        width = float(words[i + 2].replace("m", ""))
                        filters["min_width"] = width
                    except ValueError:
                        pass
        
        query_input = {
            "element_type": element_type,
            "filters": filters
        }
        
        return json.dumps(query_input)
    
    def _prepare_calculation_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare JSON input for calculate_metrics tool."""
        desc = task.description.lower()
        
        # Determine calculation type
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
            # Default to area calculation
            operation = "total_area"
            params = {"element_type": "spaces"}
        
        calc_input = {
            "operation": operation,
            **params
        }
        
        return json.dumps(calc_input)
    
    def _prepare_relationship_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare JSON input for find_related_elements tool."""
        # This is more complex - would need specific element IDs
        # For now, create a basic structure
        rel_input = {
            "element_id": "element_to_be_determined",
            "relationship_type": "adjacent",
            "parameters": {}
        }
        
        return json.dumps(rel_input)
    
    def _prepare_validation_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare JSON input for validate_compliance_rule tool."""
        desc = task.description.lower()
        
        # Determine rule type
        if any(term in desc for term in ["fire", "emergency"]):
            rule_type = "fire_safety"
        elif any(term in desc for term in ["accessibility", "ada"]):
            rule_type = "accessibility"
        elif any(term in desc for term in ["structural", "load"]):
            rule_type = "structural"
        else:
            rule_type = "general"
        
        val_input = {
            "rule_type": rule_type,
            "element_id": "element_to_be_determined", 
            "criteria": {"extracted_from": task.description}
        }
        
        return json.dumps(val_input)
    
    def _extract_search_query(self, task: Task) -> str:
        """Extract search query from task description."""
        desc = task.description
        
        # Remove common prefixes to get core query
        prefixes_to_remove = [
            "retrieve ",
            "search for ",
            "find ",
            "get ",
            "search compliance documents for ",
            "retrieve compliance rules for "
        ]
        
        query = desc.lower()
        for prefix in prefixes_to_remove:
            if query.startswith(prefix):
                query = query[len(prefix):]
                break
        
        return query.strip()
    
    def _execute_with_timeout(self, tool_function: Callable, tool_input: Any, tool_name: str) -> Any:
        """Execute tool function with timeout protection."""
        # For now, just execute directly
        # TODO: Implement proper timeout mechanism (threading/asyncio)
        
        try:
            if isinstance(tool_input, str) and tool_name != "load_building_data" and tool_name != "search_compliance_documents":
                # Most tools expect string input, except data loading and search
                return tool_function(tool_input)
            else:
                return tool_function(tool_input)
        except Exception as e:
            raise e
    
    def _validate_output(self, tool_name: str, output: Any) -> tuple[bool, str]:
        """Validate that tool output matches expected format."""
        
        # All tools should return dictionaries with status field
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
            elif tool_name in ["get_all_elements", "query_elements"]:
                if "elements" not in output and "data" not in output:
                    return False, "Missing 'elements' or 'data' field"
            elif tool_name == "calculate_metrics":
                if "result" not in output and "data" not in output:
                    return False, "Missing 'result' or 'data' field for calculation"
        
        return True, "Output validation passed"
    
    def _record_execution(self, result: ExecutionResult) -> None:
        """Record execution result in history."""
        execution_record = {
            "timestamp": time.time(),
            "tool_name": result.tool_name,
            "success": result.success,
            "execution_time": result.execution_time,
            "error_message": result.error_message,
            "output_size": len(str(result.output)) if result.output else 0
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
        failed = total - successful
        
        # Calculate average execution times by tool
        tool_times = {}
        for record in self.execution_history:
            tool = record["tool_name"]
            if tool not in tool_times:
                tool_times[tool] = []
            tool_times[tool].append(record["execution_time"])
        
        avg_times = {
            tool: sum(times) / len(times) 
            for tool, times in tool_times.items()
        }
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total) * 100 if total > 0 else 0,
            "average_execution_times": avg_times,
            "most_used_tools": list(tool_times.keys())
        }
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
        self.logger.info("Execution history cleared")