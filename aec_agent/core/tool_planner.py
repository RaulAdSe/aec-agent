"""
Tool Planner - Maps tasks to optimal tool sequences.

This module analyzes tasks and determines the best sequence of tools
to execute for each task, considering dependencies and context using
deterministic pattern matching.
"""

from typing import Dict, Any, List, Optional

from .reasoning_utils import ReasoningUtils, Task
from .llm_guardrails import default_llm_retry

# Import LangSmith tracing
from langsmith import traceable

# Import LLM components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json


class ToolPlanner:
    """
    Maps tasks to optimal tool execution sequences.
    
    Uses LLM-based reasoning only - no fallback mechanisms.
    """
    
    def __init__(self, llm=None):
        """Initialize the tool planner."""
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        # Setup LLM for intelligent tool planning
        if llm is None:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1000
            )
        else:
            self.llm = llm
        
        # Tool dependency mapping
        self.tool_dependencies = {
            "get_all_elements": ["load_building_data"],
            "get_element_properties": ["load_building_data"],
            "query_elements": ["load_building_data"],
            "calculate_distances": ["load_building_data"],
            "calculate_areas": ["load_building_data"],
            "find_related_elements": ["load_building_data"],
            "validate_compliance_rule": ["load_building_data", "search_compliance_documents"]
        }
        
        # Tool capability descriptions with detailed parameter specifications
        self.tool_capabilities = {
            "load_building_data": {
                "purpose": "Load building data from IFC JSON files",
                "input": "String: File path to building data JSON file",
                "output": "Dict with status, data (loaded building data), and logs",
                "parameters": {
                    "data_path": "Required string: Path to the building data JSON file (e.g. 'data/out/FM-ARC_v2.json')"
                },
                "prerequisites": ["File must exist and be valid JSON"],
                "good_for": ["data loading", "initialization", "file access"],
                "examples": ['load_building_data("data/out/FM-ARC_v2.json")']
            },
            "get_all_elements": {
                "purpose": "Get all elements of a specific type from the building data",
                "input": "String: Element type (spaces, doors, walls, slabs, stairs)",
                "output": "Dict with status, data (list of elements), and logs",
                "parameters": {
                    "element_type": "Required string: One of 'spaces', 'doors', 'walls', 'slabs', 'stairs'"
                },
                "prerequisites": ["Building data must be loaded first using load_building_data"],
                "good_for": ["element discovery", "counting", "overview", "getting element IDs"],
                "examples": [
                    'get_all_elements("doors")',
                    'get_all_elements("spaces")',
                    'get_all_elements("walls")'
                ]
            },
            "get_element_properties": {
                "purpose": "Get all properties and quantities for a specific element",
                "input": "String: Element ID (obtained from get_all_elements or query_elements)",
                "output": "Dict with status, data (complete element properties), and logs", 
                "parameters": {
                    "element_id": "Required string: Unique identifier of the element (must be obtained from get_all_elements/query_elements first)"
                },
                "prerequisites": ["Building data loaded", "Valid element ID from get_all_elements or query_elements"],
                "good_for": ["detailed analysis", "property inspection", "discovering available properties", "understanding element structure"],
                "examples": [
                    'get_element_properties("door_123")',
                    'get_element_properties("space_456")'
                ],
                "workflow": "ALWAYS use get_all_elements() or query_elements() first to obtain element_id"
            },
            "query_elements": {
                "purpose": "Filter and search elements with specific criteria",
                "input": "Element type (string) and optional filters (dict)",
                "output": "Dict with status, data (filtered list of elements), and logs",
                "parameters": {
                    "element_type": "Required string: One of 'spaces', 'doors', 'walls', 'slabs', 'stairs'",
                    "filters": "Optional dict: Filter criteria using property paths (use get_element_properties first to discover paths)"
                },
                "prerequisites": ["Building data loaded", "For filters: use get_element_properties on sample element to discover property names"],
                "good_for": ["filtering", "finding specific elements", "complex searches", "conditional filtering"],
                "examples": [
                    'query_elements("doors", {"properties.IsExternal": True})',
                    'query_elements("spaces", {"quantities.NetFloorArea__gt": 50})',
                    'query_elements("doors", {"quantities.Width__gte": 0.8})'
                ],
                "workflow": "1. Use get_element_properties() on sample element to discover available property paths, 2. Use discovered paths in filters"
            },
            "calculate_distances": {
                "purpose": "Calculate distances between building elements or coordinates",
                "input": "Element IDs or coordinate points",
                "output": "Distance in meters",
                "parameters": {
                    "element1_id": "String: First element ID (for element-to-element distance)",
                    "element2_id": "String: Second element ID (for element-to-element distance)",
                    "point1": "List[float]: First coordinate [x, y] or [x, y, z] (for point-to-point distance)",
                    "point2": "List[float]: Second coordinate [x, y] or [x, y, z] (for point-to-point distance)"
                },
                "prerequisites": ["Building data loaded (for element distances)", "Element IDs from get_all_elements (if using elements)"],
                "good_for": ["distance", "distances", "spacing", "measure", "between", "apart", "spatial measurements", "proximity"],
                "examples": [
                    'calculate_distances(element1_id="door_123", element2_id="door_456")',
                    'calculate_distances(point1=[0, 0], point2=[10, 15])',
                    'calculate_distances(point1=[0, 0, 0], point2=[10, 15, 5])'
                ],
                "workflow": "Use for ANY distance measurement task - between elements or coordinates"
            },
            "calculate_areas": {
                "purpose": "Calculate areas and volumes of building elements",
                "input": "Element IDs and calculation type",
                "output": "Area in square meters or volume in cubic meters",
                "parameters": {
                    "element_ids": "List[str]: List of element IDs to calculate for",
                    "calculation_type": "String: 'area' (default) or 'volume'"
                },
                "prerequisites": ["Building data loaded", "Element IDs from get_all_elements"],
                "good_for": ["area", "areas", "volume", "volumes", "space", "size", "square meters", "cubic meters", "floor area"],
                "examples": [
                    'calculate_areas(element_ids=["room_001", "room_002"])',
                    'calculate_areas(element_ids=["space_123"], calculation_type="volume")'
                ],
                "workflow": "Use for area or volume calculations of spaces, rooms, or other elements"
            },
            "find_related_elements": {
                "purpose": "Find elements related to a specific element through spatial or logical relationships",
                "input": "Element ID (string) and relationship type (string) with optional parameters",
                "output": "Dict with status, data (list of related elements), and logs",
                "parameters": {
                    "element_id": "Required string: ID of the source element",
                    "relationship_type": "Required string: Type of relationship to find",
                    "**kwargs": "Relationship-specific parameters"
                },
                "relationship_types": {
                    "connected_spaces": "Find spaces connected by a door",
                    "same_level": "Find elements on the same building level",
                    "within_distance": "Find elements within specified distance (requires max_distance parameter)",
                    "connected_doors": "Find doors connected to a space",
                    "adjacent_spaces": "Find spaces adjacent to given space (optional tolerance parameter)",
                    "same_property": "Find elements with same property value (requires property_path parameter)"
                },
                "prerequisites": ["Building data loaded", "Valid element ID from get_all_elements or query_elements"],
                "good_for": ["spatial relationships", "connectivity", "adjacency", "finding connected elements"],
                "examples": [
                    'find_related_elements("door_123", "connected_spaces")',
                    'find_related_elements("space_456", "connected_doors")',
                    'find_related_elements("space_123", "within_distance", max_distance=5.0, element_types=["doors"])'
                ]
            },
            "validate_compliance_rule": {
                "purpose": "Validate elements against compliance rules and building codes",
                "input": "Rule type (string), element ID (string), and criteria (dict)",
                "output": "Dict with status, data (validation result with is_valid, message, actual_value), and logs",
                "parameters": {
                    "rule_type": "Required string: Type of validation rule",
                    "element_id": "Required string: ID of element to validate",
                    "criteria": "Required dict: Rule-specific validation criteria"
                },
                "rule_types": {
                    "min_width": {
                        "description": "Check minimum width requirement",
                        "required_criteria": {"min_value": "float: Minimum required width in meters"},
                        "optional_criteria": {"width_property": "string: Property path (default: 'quantities.Width')"},
                        "example": 'validate_compliance_rule("min_width", "door_123", {"min_value": 0.8})'
                    },
                    "min_area": {
                        "description": "Check minimum area requirement", 
                        "required_criteria": {"min_value": "float: Minimum required area in m²"},
                        "optional_criteria": {"area_property": "string: Property path (default: 'quantities.NetFloorArea')"},
                        "example": 'validate_compliance_rule("min_area", "room_456", {"min_value": 10.0})'
                    },
                    "accessibility_width": {
                        "description": "Check accessibility width requirements",
                        "required_criteria": {"min_width": "float: Minimum width for accessibility"},
                        "optional_criteria": {"door_type": "string: Type of door (default: 'general')"},
                        "example": 'validate_compliance_rule("accessibility_width", "door_123", {"min_width": 0.85, "door_type": "emergency_exit"})'
                    },
                    "fire_rating": {
                        "description": "Check fire rating requirements",
                        "required_criteria": {"required_rating": "string: Required fire rating (e.g., 'EI30')"},
                        "optional_criteria": {"rating_property": "string: Property path for fire rating"},
                        "example": 'validate_compliance_rule("fire_rating", "door_789", {"required_rating": "EI30"})'
                    }
                },
                "prerequisites": ["Building data loaded", "Valid element ID", "Use get_element_properties to discover available property paths for criteria"],
                "good_for": ["compliance checking", "validation", "rule enforcement", "building code compliance"],
                "examples": [
                    'validate_compliance_rule("min_width", "door_123", {"min_value": 0.8})',
                    'validate_compliance_rule("min_area", "room_456", {"min_value": 15.0})',
                    'validate_compliance_rule("fire_rating", "door_789", {"required_rating": "EI30"})'
                ]
            },
            "search_compliance_documents": {
                "purpose": "Search compliance documents and building regulations",
                "input": "String: Query text for document search",
                "output": "Dict with status, data (relevant compliance information), and logs",
                "parameters": {
                    "query_string": "Required string: Search query for regulations and standards"
                },
                "prerequisites": ["Knowledge base available"],
                "good_for": ["regulations", "standards", "compliance requirements", "building codes"],
                "examples": [
                    'search_compliance_documents("fire door width requirements")',
                    'search_compliance_documents("accessibility door standards")',
                    'search_compliance_documents("emergency exit regulations")'
                ]
            },
            "document_findings": {
                "purpose": "Document analysis results and findings in structured format",
                "input": "Dict: Content with title, type, sections, summary, recommendations",
                "output": "Dict with status, data (documentation with timestamp and ID), and logs",
                "parameters": {
                    "content": "Required dict with keys:",
                    "content.title": "Required string: Document title",
                    "content.type": "Required string: Document type (e.g., 'analysis_report', 'compliance_summary')",
                    "content.sections": "Required list: List of content sections",
                    "content.summary": "Required string: Executive summary",
                    "content.recommendations": "Required list: List of recommendations or findings"
                },
                "prerequisites": ["Analysis data to document"],
                "good_for": ["documentation", "summarizing results", "creating reports", "recording findings", "formatting results", "presenting data", "final output"],
                "examples": [
                    '''document_findings({
    "title": "Door Distance Analysis Report",
    "type": "distance_analysis",
    "sections": ["Door Overview", "Distance Calculations", "Analysis Results"],
    "summary": "Analysis of distances between all doors in the building",
    "recommendations": ["Doors are optimally spaced", "No accessibility concerns identified"]
})'''
                ]
            }
        }
        
    
    @traceable(name="tool_planning", metadata={"component": "tool_planner"})
    def plan_tools(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan the optimal tool sequence for a task with context-aware intelligence.
        
        Args:
            task: Task to plan tools for
            context: Current execution context (includes execution memory if available)
            
        Returns:
            Dict containing success status, tool sequence, and metadata
        """
        self.logger.info(f"Planning tools for task: {task.name}")
        
        try:
            # Check if we have execution history context for enhanced planning
            execution_context = context.get("execution_context", {})
            has_execution_history = bool(execution_context.get("recent_execution_steps", []))
            
            if has_execution_history:
                # Try context-aware LLM planning first
                context_aware_tools = self._context_aware_llm_planning(task, context, execution_context)
                if context_aware_tools:
                    self.logger.info(f"Used context-aware LLM planning: {task.name}")
                    return {
                        "success": True,
                        "tool_sequence": context_aware_tools,
                        "method": "context_aware_llm",
                        "metadata": {"context_aware": True, "execution_history_used": True}
                    }
            
            # Use ONLY LLM-based intelligent tool selection - NO FALLBACKS
            llm_tools = self._llm_plan_tools(task, context)
            if llm_tools:
                self.logger.info(f"Used LLM for tool planning: {task.name}")
                return {
                    "success": True,
                    "tool_sequence": llm_tools,
                    "method": "llm_reasoning",
                    "metadata": {"llm_planned": True}
                }
            
            # NO FALLBACKS - Fail explicitly if LLM planning fails
            raise RuntimeError(f"LLM tool planning failed for task: {task.name}. No fallback mechanisms available.")
            
        except Exception as e:
            self.logger.error(f"Tool planning failed for {task.name}: {e}")
            # NO FALLBACKS - Re-raise the exception
            raise RuntimeError(f"Tool planning failed for task: {task.name}. Error: {str(e)}")
    
    @traceable(name="context_aware_tool_planning")
    def plan_tools_with_execution_history(
        self, 
        task: Task, 
        context: Dict[str, Any], 
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhanced tool planning that leverages execution history for better decisions.
        
        Args:
            task: Task to plan tools for
            context: Current execution context
            execution_context: Rich execution context from execution memory
            
        Returns:
            Dict containing success status, tool sequence, and metadata
        """
        self.logger.info(f"Context-aware planning for task: {task.name}")
        
        # Enhance context with execution history insights
        enhanced_context = self._enhance_context_with_history(context, execution_context)
        
        # Try context-aware LLM planning
        context_tools = self._context_aware_llm_planning(task, enhanced_context, execution_context)
        if context_tools:
            return {
                "success": True,
                "tool_sequence": context_tools,
                "method": "context_aware_llm",
                "metadata": {
                    "context_aware": True,
                    "execution_insights_used": True,
                    "history_length": len(execution_context.get("recent_execution_steps", []))
                }
            }
        
        # NO FALLBACKS - Fail explicitly if context-aware planning fails
        raise RuntimeError(f"Context-aware LLM planning failed for task: {task.name}. No fallback mechanisms available.")
    
    @traceable(name="llm_tool_planning")
    def _llm_plan_tools(self, task: Task, context: Dict[str, Any]) -> Optional[List[str]]:
        """Use LLM to intelligently select optimal tools for a task."""
        
        # Create comprehensive tool planning prompt with detailed specifications
        tool_specs = self._format_tool_specifications_for_llm()
        
        tool_planning_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert tool selection specialist for AEC (Architecture, Engineering, Construction) compliance analysis.

{tool_specs}

Context: {{context}}

CRITICAL WORKFLOW REQUIREMENTS:
1. Building data MUST be loaded first before any other building-related operations
2. Element IDs MUST be obtained from get_all_elements() or query_elements() before using them in other tools
3. Property paths MUST be discovered using get_element_properties() before filtering or validation
4. Distance calculations require specific element IDs or coordinates
5. Documentation should be the final step after analysis

PARAMETER REQUIREMENTS:
- calculate_distances requires: element1_id="id1", element2_id="id2" OR point1=[x,y], point2=[x,y]
- calculate_areas requires: element_ids=["id1", "id2"], calculation_type="area" (or "volume")
- get_all_elements requires: element_type="doors" (or "spaces", "walls", etc.)
- document_findings requires: structured content dict with title, type, sections, summary, recommendations

Your job: Select the SINGLE BEST tool for this specific task.

SELECTION RULES:
1. Return ONLY the tool name (e.g., "get_all_elements")
2. Choose the tool that DIRECTLY accomplishes the task with proper parameters
3. Consider prerequisites - if task needs element IDs, ensure they can be obtained first
4. For distance calculations, ALWAYS use "calculate_distances"
5. For area/volume calculations, use "calculate_areas"
6. For documentation/formatting results, ALWAYS use "document_findings"
7. If task requires discovering available properties, use "get_element_properties"
8. If no tool can accomplish the task, return "none"

COMMON TASK PATTERNS:
- "Load data" → "load_building_data"
- "Get all [elements]" → "get_all_elements"
- "Calculate distance/distances" → "calculate_distances"
- "Measure distance" → "calculate_distances"
- "Calculate areas/volumes" → "calculate_areas"
- "Find properties" → "get_element_properties"
- "Filter elements" → "query_elements"
- "Document/format/present results" → "document_findings"
- "Check compliance" → "validate_compliance_rule"
- "Search regulations" → "search_compliance_documents"
"""),
            ("human", """Task: {task_name}
Description: {task_description}
Context: Building data loaded = {building_data_loaded}

Select the best tool:""")
        ])
        
        try:
            # Execute LLM tool selection with retry guardrail
            chain = tool_planning_prompt | self.llm | StrOutputParser()
            response = self._invoke_llm_with_retry(chain, {
                "task_name": task.name,
                "task_description": task.description,
                "building_data_loaded": context.get("building_data_loaded", False),
                "context": str(context) if context else "No additional context"
            })
            
            tool_name = response.strip().lower()
            self.logger.info(f"LLM raw response for task '{task.name}': '{response}' -> tool_name: '{tool_name}'")
            
            # Validate the tool exists
            available_tools = [
                "load_building_data", "get_all_elements", "get_element_properties",
                "query_elements", "calculate_distances", "calculate_areas", "find_related_elements", 
                "validate_compliance_rule", "search_compliance_documents", "document_findings"
            ]
            
            if tool_name in available_tools:
                self.logger.info(f"LLM selected tool '{tool_name}' for task '{task.name}'")
                return [tool_name]
            elif tool_name == "none":
                self.logger.warning(f"LLM indicated no suitable tool for task '{task.name}'")
                return None
            else:
                self.logger.warning(f"LLM selected invalid tool '{tool_name}' for task '{task.name}'")
                return None
                
        except Exception as e:
            self.logger.error(f"LLM tool planning failed for {task.name}: {e}")
            return None
    
    
    def _check_tool_prerequisites(self, tool: str, context: Dict[str, Any]) -> bool:
        """Check if tool prerequisites are satisfied."""
        dependencies = self.tool_dependencies.get(tool, [])
        
        for dep in dependencies:
            if dep == "load_building_data":
                if not context.get("building_data_loaded", False):
                    return False
            elif dep == "search_compliance_documents":
                if not context.get("knowledge_base_available", True):  # Assume available by default
                    return False
        
        return True
    
    def _get_missing_prerequisites(self, tool: str, context: Dict[str, Any]) -> List[str]:
        """Get list of prerequisite tools that need to be executed first."""
        missing = []
        dependencies = self.tool_dependencies.get(tool, [])
        
        for dep in dependencies:
            if dep == "load_building_data" and not context.get("building_data_loaded", False):
                missing.append("load_building_data")
            # Add other dependency checks as needed
        
        return missing
    
    @default_llm_retry
    def _invoke_llm_with_retry(self, chain, inputs: Dict[str, Any]) -> str:
        """Invoke LLM chain with retry guardrail."""
        return chain.invoke(inputs)
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific tool."""
        return self.tool_capabilities.get(tool_name, {})
    
    def list_available_tools(self) -> List[str]:
        """Get list of all available tools."""
        return list(self.tool_capabilities.keys())
    
    
    @traceable(name="context_aware_llm_planning")
    def _context_aware_llm_planning(
        self, 
        task: Task, 
        context: Dict[str, Any], 
        execution_context: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Use LLM with execution history for context-aware tool planning."""
        
        # Prepare execution history summary
        history_summary = self._prepare_execution_history_summary(execution_context)
        
        # Create enhanced tool planning prompt
        context_aware_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert tool selection specialist for AEC compliance analysis with access to execution history.

Available tools and their capabilities:
- load_building_data: Load IFC JSON building data files (input: file_path)
- get_all_elements: Get all elements of a specific type like spaces, doors, walls, slabs, stairs (input: element_type)
- get_element_properties: Get detailed properties of a specific element (input: element_id)
- query_elements: Filter elements with specific criteria (input: JSON with element_type and filters)
- calculate_metrics: Perform calculations like counts, areas, volumes, distances between elements (input: JSON with operation and parameters)
- find_related_elements: Find spatial relationships between elements (input: JSON with element_id and relationship_type)
- validate_compliance_rule: Check elements against compliance rules (input: JSON with rule_type, element_id, criteria)
- search_compliance_documents: Search building codes and regulations (input: query_string)
- document_findings: Document analysis results and findings (input: JSON with content, title, type, sections, summary, recommendations)

Context: {context}
Execution History: {execution_history}

Based on the task requirements AND execution history, select the SINGLE BEST tool.

Key insights from execution history:
1. Which tools have been successful vs unsuccessful recently
2. What patterns of failures suggest alternative approaches
3. What context has been discovered that could inform tool choice
4. Which tools might be redundant given recent successful executions

Rules:
1. Return ONLY the tool name (e.g., "get_all_elements")
2. Learn from recent failures - avoid tools that have consistently failed for similar tasks
3. Leverage successful patterns from execution history
4. Consider discovered context (e.g., if building data is already loaded, don't reload)
5. If execution history shows a clear successful path, prefer those tools
6. If no clear pattern, return "uncertain" for fallback to standard planning

Examples based on history:
- If load_building_data recently succeeded → prefer tools that work with loaded data
- If query_elements failed recently → try get_all_elements instead
- If building context discovered → leverage that context in tool choice"""),
            ("human", """Task: {task_name}
Description: {task_description}
Building data loaded: {building_data_loaded}

Select the best tool considering execution history:""")
        ])
        
        try:
            # Execute context-aware LLM tool selection with retry guardrail
            chain = context_aware_prompt | self.llm | StrOutputParser()
            response = self._invoke_llm_with_retry(chain, {
                "task_name": task.name,
                "task_description": task.description,
                "building_data_loaded": context.get("building_data_loaded", False),
                "context": str(context) if context else "No additional context",
                "execution_history": history_summary
            })
            
            tool_name = response.strip().lower()
            
            # Validate the tool exists
            available_tools = [
                "load_building_data", "get_all_elements", "get_element_properties",
                "query_elements", "calculate_distances", "calculate_areas", "find_related_elements", 
                "validate_compliance_rule", "search_compliance_documents", "document_findings"
            ]
            
            if tool_name in available_tools:
                self.logger.info(f"Context-aware LLM selected tool '{tool_name}' for task '{task.name}'")
                return [tool_name]
            elif tool_name == "uncertain":
                self.logger.info(f"Context-aware LLM indicated uncertainty for task '{task.name}'")
                return None
            else:
                self.logger.warning(f"Context-aware LLM selected invalid tool '{tool_name}' for task '{task.name}'")
                return None
                
        except Exception as e:
            self.logger.error(f"Context-aware LLM tool planning failed for {task.name}: {e}")
            return None
    
    def _enhance_context_with_history(self, context: Dict[str, Any], execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance planning context with insights from execution history."""
        
        enhanced_context = context.copy()
        
        # Add execution insights
        recent_steps = execution_context.get("recent_execution_steps", [])
        if recent_steps:
            # Identify successful vs failed tools
            successful_tools = [step["tool"] for step in recent_steps if step.get("success", False)]
            failed_tools = [step["tool"] for step in recent_steps if not step.get("success", False)]
            
            enhanced_context["recent_successful_tools"] = successful_tools
            enhanced_context["recent_failed_tools"] = failed_tools
            
            # Add success rates
            if successful_tools or failed_tools:
                total_recent = len(recent_steps)
                success_rate = len([s for s in recent_steps if s.get("success", False)]) / total_recent
                enhanced_context["recent_success_rate"] = success_rate
        
        # Add discovered context insights
        discovered_context = execution_context.get("discovered_context_summary", {})
        if discovered_context:
            enhanced_context["discovered_context"] = discovered_context
            
            # Check for specific context that affects tool choice
            if "building_data_loaded" in str(discovered_context).lower():
                enhanced_context["building_data_loaded"] = True
            
            if "total_elements" in discovered_context:
                enhanced_context["elements_available"] = True
                enhanced_context["element_count"] = discovered_context.get("total_elements", 0)
        
        # Add failure patterns
        recent_failures = execution_context.get("recent_failures", [])
        if recent_failures:
            failure_tools = [failure.get("tool", "unknown") for failure in recent_failures]
            enhanced_context["tools_to_avoid"] = list(set(failure_tools))
            
            # Extract common failure reasons
            failure_reasons = [failure.get("error", "") for failure in recent_failures]
            enhanced_context["recent_failure_patterns"] = failure_reasons
        
        return enhanced_context
    
    def _prepare_execution_history_summary(self, execution_context: Dict[str, Any]) -> str:
        """Prepare a concise execution history summary for LLM context."""
        
        summary_parts = []
        
        # Recent execution steps
        recent_steps = execution_context.get("recent_execution_steps", [])
        if recent_steps:
            summary_parts.append(f"Recent execution steps ({len(recent_steps)}):")
            for step in recent_steps[-5:]:  # Last 5 steps
                status = "✓" if step.get("success", False) else "✗"
                confidence = step.get("confidence", 0.0)
                summary_parts.append(f"  {status} {step.get('task', 'Unknown')} → {step.get('tool', 'unknown')} (conf: {confidence:.1%})")
        
        # Success patterns
        successful_tools = [step["tool"] for step in recent_steps if step.get("success", False)]
        if successful_tools:
            tool_counts = {}
            for tool in successful_tools:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            summary_parts.append(f"Successful tool patterns: {dict(tool_counts)}")
        
        # Failure patterns
        recent_failures = execution_context.get("recent_failures", [])
        if recent_failures:
            summary_parts.append(f"Recent failures ({len(recent_failures)}):")
            for failure in recent_failures[-3:]:  # Last 3 failures
                summary_parts.append(f"  • {failure.get('task', 'Unknown')}: {failure.get('error', 'Unknown error')[:50]}")
        
        # Discovered context
        discovered_context = execution_context.get("discovered_context_summary", {})
        if discovered_context:
            summary_parts.append("Discovered context:")
            for key, value in list(discovered_context.items())[:3]:  # Top 3 discoveries
                summary_parts.append(f"  • {key}: {str(value)[:40]}")
        
        # Plan confidence
        plan_confidence = execution_context.get("plan_confidence", 1.0)
        summary_parts.append(f"Current plan confidence: {plan_confidence:.1%}")
        
        return "\n".join(summary_parts) if summary_parts else "No execution history available"
    
    def analyze_tool_performance(self, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze tool performance based on execution history.
        
        Args:
            execution_context: Rich execution context from execution memory
            
        Returns:
            Dict with tool performance analysis
        """
        recent_steps = execution_context.get("recent_execution_steps", [])
        
        if not recent_steps:
            return {"message": "No execution history to analyze"}
        
        # Analyze tool success rates
        tool_stats = {}
        for step in recent_steps:
            tool = step.get("tool", "unknown")
            success = step.get("success", False)
            confidence = step.get("confidence", 0.0)
            
            if tool not in tool_stats:
                tool_stats[tool] = {"attempts": 0, "successes": 0, "total_confidence": 0.0}
            
            tool_stats[tool]["attempts"] += 1
            if success:
                tool_stats[tool]["successes"] += 1
            tool_stats[tool]["total_confidence"] += confidence
        
        # Calculate metrics
        performance_analysis = {}
        for tool, stats in tool_stats.items():
            attempts = stats["attempts"]
            successes = stats["successes"]
            success_rate = successes / attempts if attempts > 0 else 0.0
            avg_confidence = stats["total_confidence"] / attempts if attempts > 0 else 0.0
            
            performance_analysis[tool] = {
                "attempts": attempts,
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
                "reliability_score": success_rate * avg_confidence  # Combined metric
            }
        
        # Identify best and worst performing tools
        if performance_analysis:
            best_tool = max(performance_analysis.items(), key=lambda x: x[1]["reliability_score"])
            worst_tool = min(performance_analysis.items(), key=lambda x: x[1]["reliability_score"])
            
            return {
                "tool_performance": performance_analysis,
                "best_performing_tool": best_tool[0],
                "worst_performing_tool": worst_tool[0],
                "total_executions": len(recent_steps),
                "overall_success_rate": sum(1 for step in recent_steps if step.get("success", False)) / len(recent_steps)
            }
        
        return {"message": "No tool performance data available"}
    
    def recommend_tool_alternatives(self, failed_tool: str, task_description: str) -> List[str]:
        """
        Recommend alternative tools when a specific tool has failed.
        
        Args:
            failed_tool: Tool that failed
            task_description: Description of what the task is trying to accomplish
            
        Returns:
            List of recommended alternative tools
        """
        alternatives = []
        task_lower = task_description.lower()
        
        # Tool-specific alternatives based on failure patterns
        tool_alternatives = {
            "query_elements": ["get_all_elements", "get_element_properties"],
            "get_all_elements": ["query_elements", "calculate_metrics"],
            "calculate_metrics": ["get_all_elements", "query_elements"],
            "get_element_properties": ["query_elements", "get_all_elements"],
            "load_building_data": [],  # No alternatives for data loading
            "validate_compliance_rule": ["search_compliance_documents"],
            "search_compliance_documents": ["validate_compliance_rule"],
            "find_related_elements": ["query_elements", "get_element_properties"]
        }
        
        # Get direct alternatives
        direct_alternatives = tool_alternatives.get(failed_tool, [])
        
        # Filter based on task context
        for alternative in direct_alternatives:
            # Check if alternative matches task intent
            tool_capabilities = self.tool_capabilities.get(alternative, {})
            good_for = tool_capabilities.get("good_for", [])
            
            if any(purpose in task_lower for purpose in good_for):
                alternatives.append(alternative)
        
        return list(set(alternatives))[:3]  # Return top 3 unique alternatives
    
    def _format_tool_specifications_for_llm(self) -> str:
        """Format comprehensive tool specifications for LLM prompt with escaped braces."""
        
        def escape_braces(text: str) -> str:
            """Escape curly braces for ChatPromptTemplate."""
            return text.replace("{", "{{").replace("}", "}}")
        
        specs = ["AVAILABLE TOOLS WITH DETAILED SPECIFICATIONS:\n"]
        
        for tool_name, tool_info in self.tool_capabilities.items():
            specs.append(f"=== {tool_name.upper()} ===")
            specs.append(f"Purpose: {escape_braces(tool_info['purpose'])}")
            specs.append(f"Input: {escape_braces(tool_info['input'])}")
            specs.append(f"Output: {escape_braces(tool_info['output'])}")
            
            # Add parameters
            if 'parameters' in tool_info:
                specs.append("Parameters:")
                for param, desc in tool_info['parameters'].items():
                    specs.append(f"  - {param}: {escape_braces(desc)}")
            
            # Add operation types for calculate_metrics
            if 'operation_types' in tool_info:
                specs.append("Operation Types:")
                for op_type, op_info in tool_info['operation_types'].items():
                    specs.append(f"  - {op_type}: {escape_braces(op_info['description'])}")
                    if 'required_params' in op_info:
                        specs.append("    Required parameters:")
                        for param, desc in op_info['required_params'].items():
                            specs.append(f"      • {param}: {escape_braces(desc)}")
                    if 'example' in op_info:
                        specs.append(f"    Example: {escape_braces(op_info['example'])}")
            
            # Add rule types for validate_compliance_rule
            if 'rule_types' in tool_info:
                specs.append("Rule Types:")
                for rule_type, rule_info in tool_info['rule_types'].items():
                    specs.append(f"  - {rule_type}: {escape_braces(rule_info['description'])}")
                    if 'example' in rule_info:
                        specs.append(f"    Example: {escape_braces(rule_info['example'])}")
            
            # Add relationship types for find_related_elements
            if 'relationship_types' in tool_info:
                specs.append("Relationship Types:")
                for rel_type, rel_desc in tool_info['relationship_types'].items():
                    specs.append(f"  - {rel_type}: {escape_braces(rel_desc)}")
            
            # Add prerequisites
            if tool_info.get('prerequisites'):
                prereq_text = ', '.join(tool_info['prerequisites'])
                specs.append(f"Prerequisites: {escape_braces(prereq_text)}")
            
            # Add workflow notes
            if 'workflow' in tool_info:
                specs.append(f"Workflow: {escape_braces(tool_info['workflow'])}")
            
            # Add examples
            if 'examples' in tool_info:
                specs.append("Examples:")
                for example in tool_info['examples']:
                    specs.append(f"  {escape_braces(example)}")
            
            specs.append("")  # Empty line between tools
        
        return "\n".join(specs)