"""
Tool Executor - Executes tools with monitoring and error handling.

This module runs individual tools while providing real-time monitoring,
error recovery, and execution tracking for the reasoning system.
"""

import time
import json
from typing import Any, Dict, Callable, Optional

from .reasoning_utils import ReasoningUtils, Task, ExecutionResult

# Import LangSmith tracing
from langsmith import traceable


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
    
    @traceable(name="tool_execution", metadata={"component": "executor"})
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
            
            self.logger.info(f"Tool '{tool_name}' executed successfully in {execution_time:.4f}s ({execution_time*1000:.1f}ms)")
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
            
        elif tool_name == "get_all_properties":
            return self._prepare_properties_input(task, context)
            
        elif tool_name == "query_elements":
            return self._prepare_query_input(task, context)
            
        elif tool_name == "calculate":
            calc_params = self._prepare_calculation_input(task, context)
            return json.dumps(calc_params)
            
        elif tool_name == "find_related":
            return self._prepare_relationship_input(task, context)
            
        elif tool_name == "validate_rule":
            return self._prepare_validation_input(task, context)
            
        elif tool_name == "search_compliance_documents":
            return self._extract_search_query(task)
            
        elif tool_name == "document_findings":
            doc_params = self._prepare_documentation_input(task, context)
            return json.dumps(doc_params)
            
        elif tool_name == "simple_response":
            # Extract the original query from the task
            return task.metadata.get("original_goal", task.description)
            
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
    
    def _prepare_calculation_input(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for calculate_metrics tool with proper parameter handling."""
        desc = task.description.lower()
        
        # DISTANCE CALCULATIONS
        if any(term in desc for term in ["distance", "distances", "between", "from", "to", "apart"]):
            return self._prepare_distance_calculation(task, context, desc)
        
        # AREA CALCULATIONS
        elif any(term in desc for term in ["area", "floor area", "surface"]):
            return {"operation": "total_area", "element_type": "spaces"}
            
        # VOLUME CALCULATIONS  
        elif any(term in desc for term in ["volume", "cubic"]):
            return {"operation": "total_volume", "element_type": "spaces"}
            
        # COUNT CALCULATIONS
        elif any(term in desc for term in ["count", "number", "how many"]):
            return {"operation": "count", "element_type": self._extract_element_type(task)}
            
        # STATISTICS
        elif any(term in desc for term in ["statistics", "average", "min", "max", "sum"]):
            return {"operation": "statistics", "values": []}  # Values will need to be provided by agent
            
        # DEFAULT - Let the agent figure out what's needed
        else:
            raise ValueError(f"Cannot determine calculation operation from task description: {desc}")
    
    def _prepare_distance_calculation(self, task: Task, context: Dict[str, Any], desc: str) -> Dict[str, Any]:
        """Prepare distance calculation parameters based on task description and context."""
        
        # Check if we need to calculate distances between all doors (multiple distances)
        if any(phrase in desc for phrase in [
            "between all doors", "all doors", "distance between doors", 
            "distances between doors", "all distances", "door distances",
            "between all", "all door distances"
        ]):
            # This is a complex distance analysis requiring multiple element IDs
            # The agent will need to first get all doors, then calculate pairwise distances
            
            # Check if we have door data in context from previous operations  
            doors_data = context.get("doors_data", [])
            if doors_data and len(doors_data) >= 2:
                # We have door data - set up pairwise distance calculation for first two doors as example
                door_ids = [door.get("id") for door in doors_data if door.get("id")]
                if len(door_ids) >= 2:
                    return {
                        "operation": "distance_between_elements",
                        "element1_id": door_ids[0],
                        "element2_id": door_ids[1]
                    }
            
            # No door data yet - indicate that door IDs are needed
            return {
                "operation": "distance_between_elements",
                "element1_id": "DOOR_ID_REQUIRED",  # Signal that door IDs need to be obtained first
                "element2_id": "DOOR_ID_REQUIRED",
                "note": "Door IDs must be obtained from get_all_elements('doors') first"
            }
        
        # Check for distance between specific elements (if IDs are mentioned)
        elif "distance between" in desc:
            # Try to extract element IDs from description
            words = desc.split()
            possible_ids = []
            for word in words:
                # Look for ID-like patterns (containing letters and numbers)
                if any(char.isdigit() for char in word) and any(char.isalpha() for char in word):
                    possible_ids.append(word.strip(',.'))
            
            if len(possible_ids) >= 2:
                return {
                    "operation": "distance_between_elements",
                    "element1_id": possible_ids[0],
                    "element2_id": possible_ids[1]
                }
            else:
                # Generic distance calculation - IDs need to be determined
                return {
                    "operation": "distance_between_elements", 
                    "element1_id": "ELEMENT_ID_REQUIRED",
                    "element2_id": "ELEMENT_ID_REQUIRED",
                    "note": "Element IDs must be obtained from get_all_elements or query_elements first"
                }
        
        # Distance from point to point (if coordinates are mentioned)
        elif any(term in desc for term in ["point", "coordinate", "location"]):
            return {
                "operation": "distance_2d",  # Default to 2D unless 3D is specified
                "point1": [0, 0],  # Default coordinates - will need to be updated
                "point2": [0, 0],
                "note": "Coordinates need to be extracted from context or element geometry"
            }
        
        # Generic distance calculation
        else:
            return {
                "operation": "distance_between_elements",
                "element1_id": "ELEMENT_ID_REQUIRED",
                "element2_id": "ELEMENT_ID_REQUIRED", 
                "note": "Element IDs must be obtained first using get_all_elements or query_elements"
            }
    
    def _prepare_documentation_input(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for document_findings tool."""
        desc = task.description
        
        # Extract title from task description
        title = desc if len(desc) < 50 else f"{desc[:47]}..."
        
        # Determine document type based on task content
        desc_lower = desc.lower()
        if "distance" in desc_lower or "calculation" in desc_lower:
            doc_type = "distance_analysis"
        elif "compliance" in desc_lower or "rule" in desc_lower:
            doc_type = "compliance_report"
        elif "analysis" in desc_lower:
            doc_type = "analysis_report"
        else:
            doc_type = "general_report"
        
        # Get analysis results from context if available
        analysis_results = context.get("analysis_results", [])
        building_data = context.get("building_data_loaded", False)
        
        # Create basic document structure
        content = {
            "title": title,
            "type": doc_type,
            "sections": ["Overview", "Analysis Results", "Summary"],
            "summary": f"Analysis completed for: {desc}",
            "recommendations": ["Analysis completed successfully"]
        }
        
        # Enhance content with available context
        if analysis_results:
            content["sections"].append("Detailed Results")
            content["recommendations"] = ["Review detailed results", "Consider next steps"]
        
        if building_data:
            content["sections"].insert(1, "Building Data Overview")
            
        return content
    
    def _prepare_properties_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare input for get_all_properties tool - needs an element_id."""
        desc = task.description.lower()
        
        # Try to get element data from context
        doors_data = context.get("doors_data", [])
        stairs_data = context.get("stairs_data", [])
        
        # Determine element type from task description
        if any(word in desc for word in ["door", "doors"]):
            if doors_data:
                # Use first door element
                door_id = doors_data[0].get("id") if doors_data[0] else None
                if door_id:
                    return door_id
                    
        elif any(word in desc for word in ["stair", "stairs"]):
            if stairs_data:
                # Use first stair element
                stair_id = stairs_data[0].get("id") if stairs_data[0] else None
                if stair_id:
                    return stair_id
        
        # Try to extract element ID from description
        words = desc.split()
        for word in words:
            # Look for ID-like patterns
            if any(char.isdigit() for char in word) and any(char.isalpha() for char in word):
                return word.strip(',.()[]')
        
        # Default - return a placeholder that indicates we need an element ID
        return "ELEMENT_ID_REQUIRED"
    
    def _prepare_calculation_input(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for calculate tool - handles distance, area, and volume calculations."""
        desc = task.description.lower()
        
        # Determine calculation type based on description
        if any(term in desc for term in ["distance", "far", "between", "apart"]):
            return self._prepare_distance_calculation(task, context)
        elif any(term in desc for term in ["area", "volume", "square", "cubic", "m2", "m3"]):
            return self._prepare_area_calculation(task, context)
        else:
            # Default to distance if unclear
            return self._prepare_distance_calculation(task, context)
    
    def _prepare_distance_calculation(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare distance calculation parameters."""
        desc = task.description.lower()
        
        # Check if we have door data in context from previous operations  
        doors_data = context.get("doors_data", [])
        stairs_data = context.get("stairs_data", [])
        
        # Handle door-to-stair distance
        if doors_data and stairs_data:
            door_ids = [door.get("id") for door in doors_data if door.get("id")]
            stair_ids = [stair.get("id") for stair in stairs_data if stair.get("id")]
            if door_ids and stair_ids:
                return {
                    "operation": "distance_between_elements",
                    "element1_id": door_ids[0],  # First door
                    "element2_id": stair_ids[0]  # First stair
                }
        
        # Handle door-to-door distance
        elif doors_data and len(doors_data) >= 2:
            door_ids = [door.get("id") for door in doors_data if door.get("id")]
            if len(door_ids) >= 2:
                return {
                    "operation": "distance_between_elements",
                    "element1_id": door_ids[0],
                    "element2_id": door_ids[1]
                }
        
        # Try to extract element IDs from description
        words = desc.split()
        possible_ids = []
        for word in words:
            # Look for ID-like patterns (containing letters and numbers)
            if any(char.isdigit() for char in word) and any(char.isalpha() for char in word):
                possible_ids.append(word.strip(',.'))
        
        if len(possible_ids) >= 2:
            return {
                "operation": "distance_between_elements",
                "element1_id": possible_ids[0],
                "element2_id": possible_ids[1]
            }
        
        # Default - indicate that element IDs are needed
        return {
            "operation": "distance_between_elements",
            "element1_id": "ELEMENT_ID_REQUIRED",
            "element2_id": "ELEMENT_ID_REQUIRED",
            "note": "Element IDs must be obtained from get_all_elements first"
        }
    
    def _prepare_area_calculation(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare area/volume calculation parameters."""
        desc = task.description.lower()
        
        # Try to get element data from context
        spaces_data = context.get("spaces_data", [])
        if spaces_data:
            element_ids = [space.get("id") for space in spaces_data if space.get("id")]
            if element_ids:
                return {
                    "operation": "area_sum",
                    "element_ids": element_ids
                }
        
        # Default - indicate that element IDs are needed
        return {
            "operation": "area_sum",
            "element_ids": ["ELEMENT_IDS_REQUIRED"],
            "note": "Element IDs must be obtained from get_all_elements first"
        }
    
    def _prepare_relationship_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare JSON input for find_related_elements tool."""
        # Try to extract element ID from context data
        element_id = self._extract_element_id_from_context(context, task.description)
        
        # Determine relationship type from task description
        desc = task.description.lower()
        relationship_type = "adjacent"  # default
        
        if "spatial" in desc or "space" in desc:
            relationship_type = "spatial"
        elif "connect" in desc:
            relationship_type = "connected"
        elif "adjacent" in desc:
            relationship_type = "adjacent"
        
        return json.dumps({
            "element_id": element_id,
            "relationship_type": relationship_type,
            "parameters": {}
        })
    
    def _prepare_validation_input(self, task: Task, context: Dict[str, Any]) -> str:
        """Prepare JSON input for validate_compliance_rule tool."""
        desc = task.description.lower()
        
        # Determine rule type from task description
        if any(term in desc for term in ["fire", "emergency"]):
            rule_type = "fire_safety"
        elif any(term in desc for term in ["accessibility", "ada"]):
            rule_type = "accessibility"
        elif any(term in desc for term in ["structural", "load"]):
            rule_type = "structural"
        elif any(term in desc for term in ["stair", "step"]):
            rule_type = "stair_compliance"
        elif any(term in desc for term in ["door"]):
            rule_type = "door_compliance"
        else:
            rule_type = "general"
        
        # Try to extract element ID from context data
        element_id = self._extract_element_id_from_context(context, desc)
        
        return json.dumps({
            "rule_type": rule_type,
            "element_id": element_id,
            "criteria": {"extracted_from": task.description}
        })
    
    def _extract_element_id_from_context(self, context: Dict[str, Any], description: str) -> str:
        """Extract appropriate element ID from context based on task description."""
        desc_lower = description.lower()
        
        # Priority order: try to match element type in description to context data
        if "stair" in desc_lower:
            # Look for stair elements
            if "stairs_data" in context and context["stairs_data"]:
                elements = context["stairs_data"]
                if isinstance(elements, list) and elements and elements[0].get("id"):
                    return elements[0]["id"]
        
        elif "door" in desc_lower:
            # Look for door elements
            if "doors_data" in context and context["doors_data"]:
                elements = context["doors_data"]
                if isinstance(elements, list) and elements and elements[0].get("id"):
                    return elements[0]["id"]
        
        elif "wall" in desc_lower:
            # Look for wall elements
            if "walls_data" in context and context["walls_data"]:
                elements = context["walls_data"]
                if isinstance(elements, list) and elements and elements[0].get("id"):
                    return elements[0]["id"]
        
        elif "space" in desc_lower:
            # Look for space elements
            if "spaces_data" in context and context["spaces_data"]:
                elements = context["spaces_data"]
                if isinstance(elements, list) and elements and elements[0].get("id"):
                    return elements[0]["id"]
        
        # Fallback: look for any elements in context
        for key, value in context.items():
            if key.endswith("_data") and isinstance(value, list) and value:
                if isinstance(value[0], dict) and value[0].get("id"):
                    return value[0]["id"]
        
        # Last resort: return a placeholder that indicates the issue
        return "NO_ELEMENT_ID_IN_CONTEXT"
    
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