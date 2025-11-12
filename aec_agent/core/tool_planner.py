"""
Tool Planner - Maps tasks to optimal tool sequences.

This module analyzes tasks and determines the best sequence of tools
to execute for each task, considering dependencies and context.
"""

from typing import Dict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .reasoning_utils import ReasoningUtils, Task


class ToolPlanner:
    """
    Maps tasks to optimal tool execution sequences.
    
    Understands tool capabilities, dependencies, and prerequisites
    to create efficient execution plans.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize the tool planner."""
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1, 
            max_tokens=1500
        )
        self.logger = ReasoningUtils.setup_logger(__name__)
        
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
            "search": ["search_compliance_documents"]
        }
    
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
            # Try pattern-based planning first
            pattern_tools = self._try_pattern_matching(task, context)
            if pattern_tools:
                self.logger.info(f"Used pattern matching for {task.name}")
                return {
                    "success": True,
                    "tool_sequence": pattern_tools,
                    "method": "pattern",
                    "metadata": {"pattern_matched": True}
                }
            
            # Fall back to LLM-based planning
            llm_tools = self._llm_tool_planning(task, context)
            self.logger.info(f"Used LLM planning for {task.name}")
            return {
                "success": True,
                "tool_sequence": llm_tools,
                "method": "llm",
                "metadata": {"llm_planned": True}
            }
            
        except Exception as e:
            self.logger.error(f"Tool planning failed for {task.name}: {e}")
            return {
                "success": False,
                "tool_sequence": [],
                "message": f"Tool planning failed: {str(e)}",
                "error": ReasoningUtils.extract_error_info(e)
            }
    
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
        filtered_tools = []
        
        for tool in tools:
            # Check if prerequisites are met
            if self._check_tool_prerequisites(tool, context):
                filtered_tools.append(tool)
            else:
                # Add prerequisite tools first
                prereq_tools = self._get_missing_prerequisites(tool, context)
                filtered_tools.extend(prereq_tools)
                filtered_tools.append(tool)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tools = []
        for tool in filtered_tools:
            if tool not in seen:
                seen.add(tool)
                unique_tools.append(tool)
        
        return unique_tools
    
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
    
    def _llm_tool_planning(self, task: Task, context: Dict[str, Any]) -> List[str]:
        """Use LLM to plan tool sequence for a task."""
        
        # Create tool descriptions for the prompt
        tool_descriptions = []
        for tool_name, info in self.tool_capabilities.items():
            desc = f"- {tool_name}: {info['purpose']} (Input: {info['input']})"
            tool_descriptions.append(desc)
        
        tool_list = "\n".join(tool_descriptions)
        
        # Create context summary
        context_summary = self._create_context_summary(context)
        
        # Create planning prompt
        prompt = PromptTemplate.from_template("""
You are an expert at selecting the right tools for AEC compliance tasks.

TASK: {task_name}
DESCRIPTION: {task_description}

CURRENT CONTEXT:
{context_summary}

AVAILABLE TOOLS:
{tool_list}

TOOL DEPENDENCIES:
- Most tools (except search_compliance_documents) require building data to be loaded first
- Validation tools work best when compliance documents have been searched first
- Element-specific tools need element identification first

Select 1-3 tools that would best accomplish this task. Consider:
1. What data/information is needed
2. What operations need to be performed
3. Current context and what's already available
4. Prerequisites and dependencies

RESPOND WITH ONLY A JSON LIST:
["tool1", "tool2", "tool3"]

Examples:
- For loading data: ["load_building_data"]
- For finding doors: ["query_elements"] 
- For compliance checking: ["search_compliance_documents", "validate_compliance_rule"]
- For calculating areas: ["calculate_metrics"]
""")
        
        # Get LLM response
        formatted_prompt = prompt.format(
            task_name=task.name,
            task_description=task.description,
            context_summary=context_summary,
            tool_list=tool_list
        )
        
        response = self.llm.invoke(formatted_prompt)
        response_text = response.content.strip()
        
        # Parse JSON response
        import json
        try:
            tools = json.loads(response_text)
            if isinstance(tools, list):
                # Validate tools exist
                valid_tools = [t for t in tools if t in self.tool_capabilities]
                return valid_tools
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse LLM tool planning response: {response_text}")
        
        # Fallback to single generic tool
        return self._create_fallback_tools(task)
    
    def _create_context_summary(self, context: Dict[str, Any]) -> str:
        """Create context summary for LLM prompt."""
        summary_parts = []
        
        if context.get("building_data_loaded"):
            summary_parts.append("✓ Building data is loaded and available")
        else:
            summary_parts.append("✗ No building data loaded - will need to load first")
        
        building_context = context.get("building_context", {})
        if building_context:
            elements = building_context.get("available_element_types", {})
            if elements:
                element_summary = ", ".join([f"{k}: {v}" for k, v in elements.items() if v > 0])
                summary_parts.append(f"Available elements: {element_summary}")
        
        active_files = context.get("active_files", [])
        if active_files:
            summary_parts.append(f"Active files: {', '.join(active_files)}")
        
        return "\n".join(summary_parts) if summary_parts else "No specific context"
    
    def _create_fallback_tools(self, task: Task) -> List[str]:
        """Create fallback tool selection when other methods fail."""
        task_text = f"{task.name} {task.description}".lower()
        
        # Simple keyword matching
        if "load" in task_text or "data" in task_text:
            return ["load_building_data"]
        elif "search" in task_text or "find" in task_text or "query" in task_text:
            return ["query_elements"]
        elif "calculate" in task_text or "compute" in task_text:
            return ["calculate_metrics"]
        elif "validate" in task_text or "check" in task_text or "compliance" in task_text:
            return ["validate_compliance_rule"]
        else:
            # Default to element querying as most generic useful tool
            return ["query_elements"]
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific tool."""
        return self.tool_capabilities.get(tool_name, {})
    
    def list_available_tools(self) -> List[str]:
        """Get list of all available tools."""
        return list(self.tool_capabilities.keys())