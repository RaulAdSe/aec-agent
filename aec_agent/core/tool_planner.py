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
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
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
                "purpose": "Perform calculations on building elements including distances, areas, volumes, counts",
                "input": "Operation type and parameters (JSON)",
                "output": "Calculated metrics and values",
                "prerequisites": ["building data loaded"],
                "good_for": ["measurements", "calculations", "quantitative analysis", "distance calculations", "spatial metrics", "geometric analysis"]
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
            },
            "document_findings": {
                "purpose": "Document analysis results and findings",
                "input": "Content to document (JSON with content, title, type, sections, summary, recommendations)",
                "output": "Structured documentation with timestamp and document ID",
                "prerequisites": [],
                "good_for": ["documentation", "summarizing results", "creating reports", "recording findings", "formatting results", "presenting data", "final output"]
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
        
        # Create tool planning prompt
        tool_planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert tool selection specialist for AEC (Architecture, Engineering, Construction) compliance analysis.

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
Task: "Document findings" → "document_findings"
Task: "Create report" → "document_findings"
Task: "Calculate distances" → "calculate_metrics"
Task: "Calculate distance between doors" → "calculate_metrics"
Task: "Format results" → "document_findings"
Task: "Present data" → "document_findings"
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
                "query_elements", "calculate_metrics", "find_related_elements", 
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
                "query_elements", "calculate_metrics", "find_related_elements", 
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