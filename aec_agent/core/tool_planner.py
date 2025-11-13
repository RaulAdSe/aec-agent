"""
Tool Planner - Maps tasks to optimal tool sequences.

This module analyzes tasks and determines the best sequence of tools
to execute for each task, considering dependencies and context using
deterministic pattern matching.
"""

from typing import Dict, Any, List, Optional

from .reasoning_utils import ReasoningUtils, Task

# Import LangSmith tracing
from langsmith import traceable

# Import LLM components
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import json


class ToolPlanner:
    """
    Maps tasks to optimal tool execution sequences.
    
    Uses LLM-based reasoning with fallback to deterministic pattern matching.
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
            "calculate_metrics": ["load_building_data"],
            "find_related_elements": ["load_building_data"],
            "validate_compliance_rule": ["load_building_data", "search_compliance_documents"]
        }
        
        # Tool capability descriptions
        self.tool_capabilities = {
            "load_building_data": {
                "purpose": "Load building data from IFC JSON files",
                "input": "File path to building data",
                "output": "Complete building data with element counts",
                "prerequisites": [],
                "good_for": ["data loading", "initialization", "file access"]
            },
            "get_all_elements": {
                "purpose": "Get all elements of a specific type",
                "input": "Element type (spaces, doors, walls, slabs, stairs)",
                "output": "List of all elements of that type",
                "prerequisites": ["building data loaded"],
                "good_for": ["element discovery", "counting", "overview"]
            },
            "get_element_properties": {
                "purpose": "Get detailed properties of a specific element",
                "input": "Element ID",
                "output": "Complete property data for the element",
                "prerequisites": ["building data loaded", "element ID known"],
                "good_for": ["detailed analysis", "property inspection", "validation"]
            },
            "query_elements": {
                "purpose": "Filter and search elements with specific criteria",
                "input": "Element type and filter criteria (JSON)",
                "output": "Filtered list of elements matching criteria",
                "prerequisites": ["building data loaded"],
                "good_for": ["filtering", "finding specific elements", "complex searches"]
            },
            "calculate_metrics": {
                "purpose": "Perform calculations on building elements",
                "input": "Operation type and parameters (JSON)",
                "output": "Calculated metrics and values",
                "prerequisites": ["building data loaded"],
                "good_for": ["measurements", "calculations", "quantitative analysis"]
            },
            "find_related_elements": {
                "purpose": "Find elements related to a specific element",
                "input": "Element ID and relationship type (JSON)",
                "output": "List of related elements",
                "prerequisites": ["building data loaded", "element ID known"],
                "good_for": ["spatial relationships", "connectivity", "adjacency"]
            },
            "validate_compliance_rule": {
                "purpose": "Validate elements against compliance rules",
                "input": "Rule type, element ID, and criteria (JSON)",
                "output": "Compliance validation results",
                "prerequisites": ["building data loaded", "compliance rules available"],
                "good_for": ["compliance checking", "validation", "rule enforcement"]
            },
            "search_compliance_documents": {
                "purpose": "Search compliance documents and regulations",
                "input": "Query text for document search",
                "output": "Relevant compliance information",
                "prerequisites": ["knowledge base available"],
                "good_for": ["regulations", "standards", "compliance requirements"]
            }
        }
        
        # Common task patterns
        self.task_patterns = {
            "load": ["load_building_data"],
            "identify": ["get_all_elements", "query_elements"],
            "find": ["query_elements", "find_related_elements"],
            "analyze": ["query_elements", "get_element_properties", "calculate_metrics"],
            "calculate": ["calculate_metrics"],
            "validate": ["search_compliance_documents", "validate_compliance_rule"],
            "check": ["validate_compliance_rule"],
            "retrieve": ["search_compliance_documents"],
            "search": ["search_compliance_documents"],
            "extract": ["get_all_elements"],
            "extract requested information": ["get_all_elements"],
            "present": ["get_all_elements"],
            "present results": ["get_all_elements"]
        }
    
    @traceable(name="tool_planning", metadata={"component": "tool_planner"})
    def plan_tools(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan the optimal tool sequence for a task.
        
        Args:
            task: Task to plan tools for
            context: Current execution context
            
        Returns:
            Dict containing success status, tool sequence, and metadata
        """
        self.logger.info(f"Planning tools for task: {task.name}")
        
        try:
            # First try LLM-based intelligent tool selection
            llm_tools = self._llm_plan_tools(task, context)
            if llm_tools:
                self.logger.info(f"Used LLM for tool planning: {task.name}")
                return {
                    "success": True,
                    "tool_sequence": llm_tools,
                    "method": "llm_reasoning",
                    "metadata": {"llm_planned": True}
                }
            
            # Fallback to pattern matching if LLM fails
            pattern_tools = self._try_pattern_matching(task, context)
            if pattern_tools:
                self.logger.info(f"Used pattern matching fallback for {task.name}")
                return {
                    "success": True,
                    "tool_sequence": pattern_tools,
                    "method": "pattern_fallback",
                    "metadata": {"pattern_matched": True}
                }
            
            # No planning possible
            self.logger.error(f"Both LLM and pattern matching failed for task: {task.name}")
            return {
                "success": False,
                "tool_sequence": [],
                "message": f"Tool planning failed for task: {task.name}",
                "available_patterns": list(self._get_available_patterns())
            }
            
        except Exception as e:
            self.logger.error(f"Tool planning failed for {task.name}: {e}")
            return {
                "success": False,
                "tool_sequence": [],
                "message": f"Tool planning failed: {str(e)}",
                "error": ReasoningUtils.extract_error_info(e)
            }
    
    @traceable(name="llm_tool_planning")
    def _llm_plan_tools(self, task: Task, context: Dict[str, Any]) -> Optional[List[str]]:
        """Use LLM to intelligently select optimal tools for a task."""
        
        # Create tool planning prompt
        tool_planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert tool selection specialist for AEC (Architecture, Engineering, Construction) compliance analysis.

Available tools and their capabilities:
- load_building_data: Load IFC JSON building data files (input: file_path)
- get_all_elements: Get all elements of a specific type like spaces, doors, walls, slabs, stairs (input: element_type)
- get_element_properties: Get detailed properties of a specific element (input: element_id)
- query_elements: Filter elements with specific criteria (input: JSON with element_type and filters)
- calculate_metrics: Perform calculations like counts, areas, volumes (input: JSON with operation and parameters)
- find_related_elements: Find spatial relationships between elements (input: JSON with element_id and relationship_type)
- validate_compliance_rule: Check elements against compliance rules (input: JSON with rule_type, element_id, criteria)
- search_compliance_documents: Search building codes and regulations (input: query_string)

Context: {context}
Task Dependencies: Building data must be loaded before other operations

Your job: Select the SINGLE BEST tool for this specific task.

Rules:
1. Return ONLY the tool name (e.g., "get_all_elements")
2. Choose the most direct tool that accomplishes the task
3. Consider dependencies (data must be loaded first)
4. Be specific - if task mentions doors, use tools that work with doors
5. If no tool fits perfectly, return "none"

Examples:
Task: "Load building data" → "load_building_data"
Task: "Get all doors" → "get_all_elements" 
Task: "Count spaces" → "calculate_metrics"
Task: "Find fire doors" → "query_elements"
Task: "Check compliance" → "validate_compliance_rule"
Task: "Search regulations" → "search_compliance_documents"
"""),
            ("human", """Task: {task_name}
Description: {task_description}
Context: Building data loaded = {building_data_loaded}

Select the best tool:""")
        ])
        
        try:
            # Execute LLM tool selection
            chain = tool_planning_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "task_name": task.name,
                "task_description": task.description,
                "building_data_loaded": context.get("building_data_loaded", False),
                "context": str(context) if context else "No additional context"
            })
            
            tool_name = response.strip().lower()
            
            # Validate the tool exists
            available_tools = [
                "load_building_data", "get_all_elements", "get_element_properties",
                "query_elements", "calculate_metrics", "find_related_elements", 
                "validate_compliance_rule", "search_compliance_documents"
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
    
    def _try_pattern_matching(self, task: Task, context: Dict[str, Any]) -> Optional[List[str]]:
        """Try to match task against known patterns."""
        task_name_lower = task.name.lower()
        task_description_lower = task.description.lower()
        
        # Check for direct pattern matches
        for pattern_key, tools in self.task_patterns.items():
            if pattern_key in task_name_lower or pattern_key in task_description_lower:
                # Filter tools based on context and dependencies
                filtered_tools = self._filter_tools_by_context(tools, context)
                if filtered_tools:
                    return filtered_tools
        
        # Check for AEC-specific patterns
        aec_tools = self._check_aec_patterns(task, context)
        if aec_tools:
            return aec_tools
        
        return None
    
    def _check_aec_patterns(self, task: Task, context: Dict[str, Any]) -> Optional[List[str]]:
        """Check for AEC-specific task patterns."""
        task_text = f"{task.name} {task.description}".lower()
        
        # Fire safety pattern
        if any(term in task_text for term in ["fire", "emergency", "exit", "safety"]):
            if "load" in task_text or "data" in task_text:
                return ["load_building_data"]
            elif "identify" in task_text or "find" in task_text:
                return ["query_elements"]
            elif "validate" in task_text or "check" in task_text:
                return ["search_compliance_documents", "validate_compliance_rule"]
        
        # Accessibility pattern
        if any(term in task_text for term in ["accessibility", "accessible", "disability", "ada"]):
            if "identify" in task_text:
                return ["query_elements"]
            elif "validate" in task_text:
                return ["search_compliance_documents", "validate_compliance_rule"]
        
        # Calculation pattern
        if any(term in task_text for term in ["calculate", "compute", "measure", "area", "volume"]):
            return ["calculate_metrics"]
        
        # Regulation pattern
        if any(term in task_text for term in ["regulation", "standard", "code", "requirement"]):
            return ["search_compliance_documents"]
        
        return None
    
    def _filter_tools_by_context(self, tools: List[str], context: Dict[str, Any]) -> List[str]:
        """Filter tools based on current context and dependencies."""
        # For deterministic behavior, return single best tool rather than sequences
        # Prerequisites should be handled by proper context, not tool sequences
        
        for tool in tools:
            # Check if prerequisites are met
            if self._check_tool_prerequisites(tool, context):
                return [tool]  # Return single tool if prerequisites met
        
        # If no tool has met prerequisites, return first tool and let execution handle it
        # This will fail explicitly if prerequisites aren't met, which is desired behavior
        return tools[:1] if tools else []
    
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
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific tool."""
        return self.tool_capabilities.get(tool_name, {})
    
    def list_available_tools(self) -> List[str]:
        """Get list of all available tools."""
        return list(self.tool_capabilities.keys())
    
    def _get_available_patterns(self) -> List[str]:
        """Get list of available task patterns."""
        patterns = []
        # Extract patterns from task mapping
        for pattern_list in self.task_patterns.values():
            patterns.extend(pattern_list)
        return sorted(set(patterns))