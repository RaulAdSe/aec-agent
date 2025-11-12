"""
Goal Decomposer - Breaks complex goals into actionable subtasks.

This module analyzes high-level goals and creates structured task breakdowns
optimized for AEC compliance workflows.
"""

import uuid
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .reasoning_utils import ReasoningUtils, Task, Priority


class GoalDecomposer:
    """
    Breaks complex goals into actionable subtasks with dependencies.
    
    Specializes in AEC compliance workflows while handling general tasks.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize the goal decomposer."""
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1,
            max_tokens=2000
        )
        self.logger = ReasoningUtils.setup_logger(__name__)
        
        # AEC-specific task patterns
        self.aec_patterns = {
            "fire_safety": [
                "Load building data",
                "Identify fire-related elements",
                "Retrieve fire safety regulations", 
                "Validate fire safety compliance"
            ],
            "accessibility": [
                "Load building data",
                "Identify accessibility elements",
                "Retrieve accessibility standards",
                "Validate accessibility compliance"
            ],
            "structural": [
                "Load building data",
                "Identify structural elements",
                "Calculate structural metrics",
                "Validate structural requirements"
            ],
            "energy": [
                "Load building data",
                "Identify energy-related elements",
                "Calculate energy metrics",
                "Validate energy efficiency requirements"
            ]
        }
    
    def decompose(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompose a goal into actionable subtasks.
        
        Args:
            goal: High-level goal (e.g., "Analyze fire safety compliance")
            context: Current context (loaded data, session state, etc.)
            
        Returns:
            Dict containing success status, tasks list, and metadata
        """
        self.logger.info(f"Decomposing goal: {goal}")
        
        try:
            # Check for AEC pattern match first
            aec_tasks = self._try_aec_pattern_match(goal, context)
            if aec_tasks:
                self.logger.info(f"Used AEC pattern for decomposition")
                return {
                    "success": True,
                    "tasks": aec_tasks,
                    "method": "aec_pattern",
                    "message": f"Decomposed using AEC pattern into {len(aec_tasks)} tasks"
                }
            
            # Fall back to LLM-based decomposition
            llm_tasks = self._llm_decomposition(goal, context)
            self.logger.info(f"Used LLM for decomposition")
            return {
                "success": True,
                "tasks": llm_tasks,
                "method": "llm",
                "message": f"Decomposed using LLM into {len(llm_tasks)} tasks"
            }
            
        except Exception as e:
            self.logger.error(f"Goal decomposition failed: {e}")
            return {
                "success": False,
                "tasks": [],
                "message": f"Decomposition failed: {str(e)}",
                "error": ReasoningUtils.extract_error_info(e)
            }
    
    def _try_aec_pattern_match(self, goal: str, context: Dict[str, Any]) -> Optional[List[Task]]:
        """Try to match goal against known AEC patterns."""
        goal_lower = goal.lower()
        
        # Detect AEC domain
        if any(term in goal_lower for term in ["fire", "safety", "emergency", "exit"]):
            pattern = "fire_safety"
        elif any(term in goal_lower for term in ["accessibility", "accessible", "disability", "ada"]):
            pattern = "accessibility"
        elif any(term in goal_lower for term in ["structural", "load", "beam", "column"]):
            pattern = "structural"
        elif any(term in goal_lower for term in ["energy", "thermal", "efficiency", "hvac"]):
            pattern = "energy"
        else:
            return None
        
        # Get pattern and create tasks
        task_names = self.aec_patterns.get(pattern, [])
        if not task_names:
            return None
        
        tasks = []
        prev_task_id = None
        
        for i, name in enumerate(task_names):
            task_id = str(uuid.uuid4())
            
            # Create task with appropriate dependencies
            task = Task(
                id=task_id,
                name=name,
                description=self._generate_task_description(name, pattern, goal),
                priority=Priority.HIGH if i == 0 else Priority.MEDIUM,
                dependencies=[prev_task_id] if prev_task_id else [],
                metadata={
                    "pattern": pattern,
                    "original_goal": goal,
                    "sequence_order": i
                }
            )
            
            tasks.append(task)
            prev_task_id = task_id
        
        return tasks
    
    def _generate_task_description(self, task_name: str, pattern: str, goal: str) -> str:
        """Generate detailed description for a task."""
        descriptions = {
            "Load building data": f"Load and validate building data required for {pattern} analysis",
            "Identify fire-related elements": "Find all fire doors, exits, zones, and safety equipment",
            "Identify accessibility elements": "Find all doors, ramps, elevators, and accessibility features",
            "Identify structural elements": "Find all beams, columns, slabs, and structural components",
            "Identify energy-related elements": "Find all HVAC, lighting, and energy systems",
            "Retrieve fire safety regulations": "Search compliance documents for fire safety requirements",
            "Retrieve accessibility standards": "Search compliance documents for accessibility standards",
            "Calculate structural metrics": "Calculate loads, stresses, and structural performance metrics",
            "Calculate energy metrics": "Calculate energy consumption and efficiency metrics",
            "Validate fire safety compliance": "Check all fire elements against safety regulations",
            "Validate accessibility compliance": "Check all accessibility elements against standards",
            "Validate structural requirements": "Check all structural elements against building codes",
            "Validate energy efficiency requirements": "Check energy systems against efficiency standards"
        }
        
        return descriptions.get(task_name, f"Complete {task_name.lower()} for {goal}")
    
    def _llm_decomposition(self, goal: str, context: Dict[str, Any]) -> List[Task]:
        """Use LLM to decompose goal into tasks."""
        
        # Prepare context summary
        context_summary = self._summarize_context(context)
        
        # Create decomposition prompt
        prompt = PromptTemplate.from_template("""
You are an expert at breaking down complex AEC (Architecture, Engineering, Construction) compliance goals into actionable subtasks.

GOAL: {goal}

CURRENT CONTEXT:
{context_summary}

Break this goal into 3-7 specific, actionable subtasks that can be executed with tools. Each task should:
1. Be specific and measurable
2. Have clear dependencies (if any)  
3. Focus on one main action
4. Be achievable with available tools

AVAILABLE TOOL CATEGORIES:
- load_building_data: Load IFC building data
- query_elements: Find and filter building elements
- calculate_metrics: Perform calculations on building data
- search_compliance_docs: Search regulations and standards
- validate_compliance: Check compliance against rules

RESPOND WITH VALID JSON:
{{
    "tasks": [
        {{
            "name": "Task name",
            "description": "Detailed description of what this task accomplishes",
            "priority": "HIGH" | "MEDIUM" | "LOW",
            "dependencies": ["task_name_1", "task_name_2"],
            "estimated_tools": ["tool1", "tool2"]
        }}
    ]
}}

Remember:
- Keep tasks focused and specific
- First task usually loads data if needed
- Validation tasks come after data gathering
- Dependencies should form a logical sequence
""")
        
        # Get LLM response
        formatted_prompt = prompt.format(
            goal=goal,
            context_summary=context_summary
        )
        
        response = self.llm.invoke(formatted_prompt)
        response_text = response.content.strip()
        
        # Parse JSON response
        import json
        try:
            parsed_response = json.loads(response_text)
            task_defs = parsed_response.get("tasks", [])
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            # Fallback to simple decomposition
            return self._create_fallback_tasks(goal)
        
        # Convert to Task objects
        tasks = []
        task_name_to_id = {}
        
        for task_def in task_defs:
            task_id = str(uuid.uuid4())
            task_name = task_def.get("name", f"Task {len(tasks) + 1}")
            
            # Map priority
            priority_str = task_def.get("priority", "MEDIUM")
            priority = getattr(Priority, priority_str, Priority.MEDIUM)
            
            # Convert dependency names to IDs
            dep_names = task_def.get("dependencies", [])
            dep_ids = [task_name_to_id.get(dep) for dep in dep_names if dep in task_name_to_id]
            
            task = Task(
                id=task_id,
                name=task_name,
                description=task_def.get("description", ""),
                priority=priority,
                dependencies=dep_ids,
                metadata={
                    "original_goal": goal,
                    "estimated_tools": task_def.get("estimated_tools", []),
                    "llm_generated": True
                }
            )
            
            tasks.append(task)
            task_name_to_id[task_name] = task_id
        
        return tasks
    
    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """Create a summary of current context for the LLM."""
        summary_parts = []
        
        # Check for loaded building data
        if context.get("building_data_loaded"):
            summary_parts.append("✓ Building data is already loaded")
        else:
            summary_parts.append("✗ No building data loaded yet")
        
        # Check for active files
        active_files = context.get("active_files", [])
        if active_files:
            summary_parts.append(f"Active files: {', '.join(active_files)}")
        
        # Check for session goals
        session_goal = context.get("session_goal")
        if session_goal:
            summary_parts.append(f"Session goal: {session_goal}")
        
        # Check for available element counts
        building_context = context.get("building_context", {})
        if building_context:
            element_counts = building_context.get("available_element_types", {})
            if element_counts:
                count_summary = ", ".join([f"{k}: {v}" for k, v in element_counts.items() if v > 0])
                summary_parts.append(f"Available elements: {count_summary}")
        
        return "\n".join(summary_parts) if summary_parts else "No specific context available"
    
    def _create_fallback_tasks(self, goal: str) -> List[Task]:
        """Create simple fallback tasks when LLM decomposition fails."""
        self.logger.warning("Using fallback task decomposition")
        
        tasks = []
        
        # Always start with data loading if not loaded
        task1 = Task(
            id=str(uuid.uuid4()),
            name="Load required data",
            description=f"Load building data and documents needed for: {goal}",
            priority=Priority.HIGH,
            metadata={"fallback": True}
        )
        tasks.append(task1)
        
        # Generic analysis task
        task2 = Task(
            id=str(uuid.uuid4()),
            name="Analyze requirements",
            description=f"Analyze data and requirements for: {goal}",
            priority=Priority.MEDIUM,
            dependencies=[task1.id],
            metadata={"fallback": True}
        )
        tasks.append(task2)
        
        # Generic validation task
        task3 = Task(
            id=str(uuid.uuid4()),
            name="Validate results",
            description=f"Validate analysis results for: {goal}",
            priority=Priority.MEDIUM,
            dependencies=[task2.id],
            metadata={"fallback": True}
        )
        tasks.append(task3)
        
        return tasks